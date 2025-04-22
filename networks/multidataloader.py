import cv2
import random
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
from hist.Depth_Anything_V2.pipeline import load_depth_anything_model, infer_depth
from helper.colored_test import compute_surface_normals, colorize_normals, create_normal_weights
from weighted_hist import remove_far_objects, calculate_weighted_2d_histograms_optimized, tan_hyper, log_tran
from PIL import Image
import pandas as pd
import glob
from pathlib import Path

def linear_to_log(image):
    image_float = image.astype(np.float32) / 255.0
    image_log = np.log1p(image_float)
    image_log_scaled = (image_log / np.log1p(1.0)) * 255.0
    return image_log_scaled.astype(np.uint8)

def apply_fourier_transform(histogram, filter_type=None, cutoff=None):
    """
    Apply Fourier transform to histogram data with optional filtering.
    
    Args:
        histogram: Input histogram (2D numpy array or tensor)
        filter_type: Type of frequency filter to apply ('lowpass', 'highpass', or None)
        cutoff: Frequency cutoff for the filter (0-1 range, where 1 is the Nyquist frequency)
    
    Returns:
        Transformed histogram (same type as input)
    """
    # Convert tensor to numpy if needed
    is_tensor = isinstance(histogram, torch.Tensor)
    if is_tensor:
        device = histogram.device
        histogram_np = histogram.cpu().numpy()
    else:
        histogram_np = histogram
    
    # Apply 2D FFT
    fft_result = np.fft.fft2(histogram_np)
    fft_shifted = np.fft.fftshift(fft_result)  # Shift zero frequency to center
    
    # Apply frequency domain filtering if requested
    if filter_type and cutoff:
        rows, cols = histogram_np.shape
        crow, ccol = rows // 2, cols // 2
        mask = np.ones((rows, cols), dtype=np.float32)
        
        # Create filter mask based on requested type
        r = int(cutoff * min(crow, ccol))
        y, x = np.ogrid[:rows, :cols]
        mask_area = (y - crow) ** 2 + (x - ccol) ** 2 <= r * r
        
        if filter_type == 'lowpass':
            mask[~mask_area] = 0
        elif filter_type == 'highpass':
            mask[mask_area] = 0
        
        # Apply mask
        fft_shifted = fft_shifted * mask
    
    # Inverse FFT
    fft_ishifted = np.fft.ifftshift(fft_shifted)
    img_back = np.fft.ifft2(fft_ishifted)
    img_back = np.abs(img_back)  # Get magnitude
    
    # Convert back to tensor if the input was a tensor
    if is_tensor:
        return torch.tensor(img_back, device=device, dtype=torch.float32)
    return img_back

def get_fourier_spectrum(histogram):
    """
    Get the magnitude spectrum of the Fourier transform.
    Useful for visualization or feature extraction.
    
    Args:
        histogram: Input histogram (2D numpy array or tensor)
    
    Returns:
        Magnitude spectrum (log scale for better visualization)
    """
    # Convert tensor to numpy if needed
    if isinstance(histogram, torch.Tensor):
        histogram_np = histogram.cpu().numpy()
    else:
        histogram_np = histogram
    
    # Apply 2D FFT
    fft_result = np.fft.fft2(histogram_np)
    fft_shifted = np.fft.fftshift(fft_result)
    
    # Calculate magnitude spectrum (log scale for better visualization)
    magnitude_spectrum = 20 * np.log(np.abs(fft_shifted) + 1)
    
    return magnitude_spectrum



def add_gaussian_noise(histogram, mean=0, k=0.1):
    """
    Add Gaussian noise to histogram data.
    
    Args:
        histogram: Input histogram tensor
        mean: Mean of the Gaussian noise (default: 0)
        k: Noise scale factor (default: 0.1)
    
    Returns:
        Augmented histogram tensor
    """
    # Calculate standard deviation based on bin count
    bin_count = histogram.shape[-1]  # Assuming last dimension is bin count
    std = k * (bin_count)
    
    # Generate Gaussian noise
    noise = torch.randn_like(histogram) * std + mean
    
    # Add noise to histogram
    augmented_histogram = histogram + noise
    
    # Ensure values remain valid (non-negative for histograms)
    augmented_histogram = torch.clamp(augmented_histogram, min=0.0)
    # rand_val = random.random()
    rand_val = 0.1
    if rand_val > 1:
        return augmented_histogram
    return histogram

def parse_array_string(array_str):
    # Remove brackets and split by spaces
    return [float(x) for x in array_str.replace('[', '').replace(']', '').strip().split()]


class HistogramDataset(Dataset):
    def __init__(self, base_image_dir, base_depth_dir, base_normal_dir, csv_file, folders=None,plane=True,transform=None,training=False, fourier_transform=False, 
                 fourier_filter=None, fourier_cutoff=0.5):
        """
        Initialize the dataset with multiple folders.
        
        Args:
            base_image_dir (str): Base directory containing folder_1, folder_2, etc.
            base_depth_dir (str): Base directory for depth maps containing folder_1, folder_2, etc.
            base_normal_dir (str): Base directory for normal maps containing folder_1, folder_2, etc.
            csv_file (str): Path to CSV file containing annotations
            folders (list): List of folder names to include, e.g. ['folder_1', 'folder_2']
                           If None, will use all available folders
        """
        self.base_image_dir = "/work/SuperResolutionData/spectralRatio/data/hist_rg_for_training_log"
        self.base_depth_dir = base_depth_dir
        self.plane = plane
        self.base_normal_dir = base_normal_dir
        self.hist_rg_dir = "/work/SuperResolutionData/spectralRatio/data/hist_rg_for_training_log"
        self.hist_gb_dir = "/work/SuperResolutionData/spectralRatio/data/hist_gb_for_training_log"
        self.hist_br_dir = "/work/SuperResolutionData/spectralRatio/data/hist_br_for_training_log"
        self.hist_plane_dir = "/work/SuperResolutionData/spectralRatio/data/hist_plane_for_training_log"
        self.transform = transform
        self.check= True
        self.is_training = training
        self.fourier_transform = fourier_transform
        self.fourier_filter = fourier_filter
        self.fourier_cutoff = fourier_cutoff
        print(f"Using Fourier: {fourier_transform}")

        print(f'training ------------------{self.is_training}')
        
        # If folders not specified, detect available folders
        if folders is None:
            folders = [d for d in os.listdir(base_image_dir) if os.path.isdir(os.path.join(base_image_dir, d))]
            folders = sorted([f for f in folders if f.startswith('folder_')])
        
        self.folders = folders
        print(f"Using folders: {self.folders}")
        
        # Load CSV data and ensure filenames are lowercase
        self.csv_data = pd.read_csv(csv_file)
        self.csv_data["image_name"] = self.csv_data['image_name'].str.lower()
        
        # Store file paths and their corresponding folder
        self.image_files = []
        self.folder_map = {}  # Maps image file to its folder
        
        for folder in self.folders:
            img_dir = os.path.join(self.base_image_dir, folder)
            if not os.path.exists(img_dir):
                print(f"Warning: {img_dir} does not exist, skipping.")
                continue
                
            files = [f.lower() for f in os.listdir(img_dir) if f.lower().endswith(('.png'))]
            for file in sorted(files):
                # Check if the file has an entry in the CSV
                row = self.csv_data[self.csv_data['image_name'] == os.path.splitext(file)[0]]
                if row.empty:
                    # Try with folder as part of the path
                    alternative_name = os.path.join(folder, file)
                    row = self.csv_data[self.csv_data['image_name'] == alternative_name.lower()]
                
                # Only add images that have CSV entries
                if not row.empty:
                    self.image_files.append(file)
                    self.folder_map[file] = folder
                # else:
                #     c
                    # print(f"Dropping image {file} from dataset (no CSV entry found)")
        
        print(f"Kept {len(self.image_files)} images with CSV entries out of all images in folders")
        self.model = load_depth_anything_model()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        folder = self.folder_map[img_name]
        
        img_path = os.path.join(self.base_image_dir, folder, img_name)
        depth_name = os.path.splitext(img_name)[0] + ".png"
        normal_name = img_name
        depth_path = os.path.join(self.base_depth_dir, folder, depth_name)
        normal_path = os.path.join(self.base_normal_dir, folder, normal_name)
        rg_path = os.path.join(self.hist_rg_dir, folder, normal_name)
        gb_path = os.path.join(self.hist_gb_dir, folder, normal_name)
        br_path = os.path.join(self.hist_br_dir, folder, normal_name)
        plane_path = os.path.join(self.hist_plane_dir, folder, normal_name)

        # image = cv2.imread(img_path)


        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # image = linear_to_log(image)

        if os.path.exists(rg_path) and os.path.exists(plane_path):
            # depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
            # normal_map = cv2.imread(normal_path)
            normal_map = 2
        else:
            print(f"Error: Could not load image.{rg_path}")
            exit()

        if os.path.exists(rg_path) and os.path.exists(gb_path) and os.path.exists(br_path) and os.path.exists(plane_path):
            if self.transform=='log':
                if self.check:
                    # print('inside log')
                    self.check=False

                rg = log_tran(cv2.imread(rg_path, cv2.IMREAD_GRAYSCALE))
                gb = log_tran(cv2.imread(gb_path, cv2.IMREAD_GRAYSCALE))
                br = log_tran(cv2.imread(br_path, cv2.IMREAD_GRAYSCALE))
                plane = log_tran(cv2.imread(plane_path, cv2.IMREAD_GRAYSCALE))

            elif self.transform=='tan':
                if self.check:
                    # print('inside tan')
                    self.check=False
                rg = tan_hyper(cv2.imread(rg_path, cv2.IMREAD_GRAYSCALE))
                gb = tan_hyper(cv2.imread(gb_path, cv2.IMREAD_GRAYSCALE))
                br = tan_hyper(cv2.imread(br_path, cv2.IMREAD_GRAYSCALE))
                plane = tan_hyper(cv2.imread(plane_path, cv2.IMREAD_GRAYSCALE))

            else:
                if self.check:
                    # print('no transform')
                    self.check=False
                rg = cv2.imread(rg_path, cv2.IMREAD_GRAYSCALE)
                gb = cv2.imread(gb_path, cv2.IMREAD_GRAYSCALE)
                br = cv2.imread(br_path, cv2.IMREAD_GRAYSCALE)
                plane = cv2.imread(plane_path, cv2.IMREAD_GRAYSCALE)
        else:
            print(f"Error: Could not load hist.{img_path}")
            exit()
        # threshold = 10.0 
        # masked_image, mask = remove_far_objects(image, depth, threshold)

        # height, width, _ = image.shape
        # up = np.zeros((height, width), dtype=np.uint8)
        # front = np.zeros((height, width), dtype=np.uint8)
        # side = np.zeros((height, width), dtype=np.uint8)
        # up = (normal_map[...,1])
        # front = (normal_map[...,2])
        # side= (normal_map[...,0])

        # masked_up = side.copy()
        # masked_up[mask] = 0
        # masked_up = masked_up/255
        # rg, gb, br,plane= calculate_weighted_2d_histograms_optimized(image, masked_up, 128)
        # if self.transform=='log':
        #         if self.check:
        #             print('inside log')
        #             self.check=False
        #         rg = log_tran(rg)
        #         gb = log_tran(gb)
        #         br = log_tran(br)
        #         plane = log_tran(plane)
        # elif self.transform=='tan':
        #         if self.check:
        #             print('inside tan')
        #             self.check=False
        #         rg = tan_hyper(rg)
        #         gb = tan_hyper(gb)
        #         br = tan_hyper(br)
        #         plane = tan_hyper(plane)
        if self.fourier_transform:
            rg = apply_fourier_transform(rg, self.fourier_filter, self.fourier_cutoff)
            gb = apply_fourier_transform(gb, self.fourier_filter, self.fourier_cutoff)
            br = apply_fourier_transform(br, self.fourier_filter, self.fourier_cutoff)
            plane = apply_fourier_transform(plane, self.fourier_filter, self.fourier_cutoff)
            # cv2.imwrite('fourie.png',((rg/np.max(rg))*255).astype(np.uint8))
        if self.plane:
            
            if self.is_training:
                hist_tensor = torch.stack([add_gaussian_noise(torch.tensor(rg, dtype=torch.float32), k=0.3),
                                    add_gaussian_noise(torch.tensor(gb, dtype=torch.float32), k=0.3),
                                    add_gaussian_noise(torch.tensor(br, dtype=torch.float32), k=0.3),
                                    add_gaussian_noise(torch.tensor(plane, dtype=torch.float32), k=0.3)], dim=0)
            else:
                hist_tensor = torch.stack([torch.tensor(rg, dtype=torch.float32),
                                    torch.tensor(gb, dtype=torch.float32),
                                    torch.tensor(br, dtype=torch.float32),
                                    torch.tensor(plane, dtype=torch.float32)], dim=0)
        else:
            print(f"Error: Could not load plane hist.{img_path}")
            hist_tensor = torch.stack([torch.tensor(rg, dtype=torch.float32),
                                    torch.tensor(gb, dtype=torch.float32),
                                    torch.tensor(br, dtype=torch.float32)], dim=0)
        # print(hist_tensor.shape)
            

        # Get the image data from CSV (we already verified it exists during initialization)
        row = self.csv_data[self.csv_data['image_name'] == os.path.splitext(img_name)[0]]
        if row.empty:
            # Try with folder prefix
            print(self.csv_data['image_name'])
            alternative_name = os.path.join(folder, img_name)
            row = self.csv_data[self.csv_data['image_name'] == alternative_name.lower()]
        
        color_array = parse_array_string(row['augmented_isd'].values[0])
        r_avg = color_array[0] 
        g_avg = color_array[1]  
        b_avg = color_array[2]
        
#         print([r_avg, g_avg, b_avg])


        # Target tensor with average color values
        target_tensor = torch.tensor([r_avg, g_avg, b_avg], dtype=torch.float32)

        return hist_tensor, target_tensor

def extract_fourier_features(histogram, num_features=10):
    """
    Extract features from the Fourier spectrum of a histogram.
    
    Args:
        histogram: Input histogram (2D numpy array or tensor)
        num_features: Number of frequency components to extract
    
    Returns:
        Feature vector containing the magnitudes of the top frequency components
    """
    # Convert tensor to numpy if needed
    if isinstance(histogram, torch.Tensor):
        histogram_np = histogram.cpu().numpy()
    else:
        histogram_np = histogram
    
    # Apply 2D FFT
    fft_result = np.fft.fft2(histogram_np)
    magnitude_spectrum = np.abs(fft_result)
    
    # Flatten the spectrum and get top components
    flat_spectrum = magnitude_spectrum.flatten()
    top_indices = np.argsort(flat_spectrum)[-num_features:]
    features = flat_spectrum[top_indices]
    
    return features 

if __name__ == "__main__":
    base_image_dir = "/work/SuperResolutionData/spectralRatio/data/images_for_training"
    base_depth_dir = "/work/SuperResolutionData/spectralRatio/data/depth_for_training"
    base_normal_dir = "/work/SuperResolutionData/spectralRatio/data/surface_norm_for_training"
    csv_file = "/home/balamurugan.d/src/train.csv"

    csv_data = pd.read_csv(csv_file)
    print(csv_data.head())
    csv_data["file_name"] = csv_data['file_name'].str.lower() 
    print(csv_data.head())



    folders = [f"folder_{i}" for i in range(1, 10)] 
    
    dataset = HistogramDataset(base_image_dir, base_depth_dir, base_normal_dir, csv_file, folders=folders)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    # print("dataset shape:",dataset.shape)
    
    print(f"Dataset contains {len(dataset)} images")
    
    # Test the first batch
    for hist_batch, target_batch in dataloader:
        print("Histogram Batch shape:", hist_batch.shape)
        print("Target Batch shape:", target_batch.shape)
        print("Sample target values:", target_batch[0])
        break  # Just print the first batch