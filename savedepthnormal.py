import os
import sys
import cv2
import numpy as np
import pandas as pd
import torch
import tifffile
import argparse
from tqdm import tqdm

# Try to install imagecodecs if not present
try:
    import imagecodecs
except ImportError:
    print("Installing imagecodecs package...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "imagecodecs"])
    import imagecodecs

# Import your modules
from weighted_hist import remove_far_objects, calculate_weighted_2d_histograms_optimized

# Add the module directory to the path for importing AugmentISD_GPU
module_dir = "/home/massone.m/spectral_ratio/syn_data_pipeline/src"
sys.path.append(module_dir)
from isd_augmentation_xml_class import AugmentISD_GPU

def linear_to_log(image):
    """Converts a linear image to a log-encoded image."""
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    
    if image.dtype != np.uint8:
        # Assuming 16-bit TIFF inputs as in your original dataset
        image_float = image.astype(np.float32) / 65535.0
        image_uint8 = (image_float * 255.0).astype(np.uint8)
    else:
        image_uint8 = image
        
    image_float = image_uint8.astype(np.float32) / 255.0  # Normalize to [0, 1]
    image_log = np.log1p(image_float)  # Apply log1p (log(1 + x))
    image_log_scaled = (image_log / np.log1p(1.0)) * 255.0  # Scale back to [0, 255]
    return image_log_scaled.astype(np.uint8)

def str_to_array(s):
    """Convert a string representation of a list into a NumPy array."""
    return np.fromstring(s.strip('[]'), sep=' ')

def float_conversion(value):
    """Safely converts a value to float. If conversion fails, returns NaN."""
    try:
        return float(value)
    except ValueError:
        return np.nan

def safe_load_tiff(file_path):
    """
    Safely load a TIFF file handling the compression error.
    """
    try:
        # Try standard loading first
        return tifffile.imread(file_path)
    except Exception as e:
        if "requires the 'imagecodecs' package" in str(e):
            print(f"Handling compression for file: {file_path}")
            # Try with options for compressed files
            return tifffile.imread(file_path, imagej=True)
        else:
            # If it's a different error, raise it
            raise e

def generate_and_save_histograms(args,folders):
    # Parse arguments

    csv_file = args.csv
    folder = 'folder_'+folders
    print(folder)
    # Set up directories
    input_folder = f"/work/SuperResolutionData/spectralRatio/data/images_for_training/{folder}"
    depth_folder = f"/work/SuperResolutionData/spectralRatio/data/depth_for_training/{folder}"
    normal_folder = f"/work/SuperResolutionData/spectralRatio/data/surface_norm_for_training/{folder}"
    
    # Output folders for augmented histograms
    augmented_hist_rg_folder = f"/work/SuperResolutionData/spectralRatio/data/hist_rg_for_training_log/{folder}"
    augmented_hist_gb_folder = f"/work/SuperResolutionData/spectralRatio/data/hist_gb_for_training_log/{folder}"
    augmented_hist_br_folder = f"/work/SuperResolutionData/spectralRatio/data/hist_br_for_training_log/{folder}"
    augmented_hist_plane_folder = f"/work/SuperResolutionData/spectralRatio/data/hist_plane_for_training_log/{folder}"

    # Ensure output directories exist
    os.makedirs(augmented_hist_rg_folder, exist_ok=True)
    os.makedirs(augmented_hist_gb_folder, exist_ok=True)
    os.makedirs(augmented_hist_br_folder, exist_ok=True)
    os.makedirs(augmented_hist_plane_folder, exist_ok=True)
    
    # Initialize GPU augmentation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    augmenter = AugmentISD_GPU(device=device)
    
    # Read the CSV file with converters
    augmentations_csv = pd.read_csv(csv_file, converters={
        'starting_isd': str_to_array,
        'augmented_isd': str_to_array,
        'translation': str_to_array,
        'scaling_factor': float_conversion
    })
    
    print(f"Processing {len(augmentations_csv)} entries from CSV file...")
    
    # Process each entry in the CSV
    for idx in tqdm(range(len(augmentations_csv)), desc="Processing augmented images"):
        # Extract file info and augmentation parameters
        file_name = augmentations_csv.iloc[idx, 0]
        folder_name = augmentations_csv.iloc[idx, 2]
        aug_name = augmentations_csv.iloc[idx, 1]
        
        # Make sure we're processing files from the correct folder
        if folder_name != folder:
            continue
            
        starting_isd = augmentations_csv.iloc[idx, 3]
        augmented_isd = augmentations_csv.iloc[idx, 4]
        translation = augmentations_csv.iloc[idx, 5]
        scaling_factor = augmentations_csv.iloc[idx, 6]
        
        # Skip if any parameters are NaN
        if np.any(np.isnan(starting_isd)) or np.any(np.isnan(augmented_isd)) or \
           np.any(np.isnan(translation)) or np.isnan(scaling_factor):
            print(f"Skipping {file_name} due to NaN parameters")
            continue
        
        # Construct paths
        img_path = os.path.join(input_folder, file_name)
        depth_path = os.path.join(depth_folder, os.path.splitext(file_name)[0] + ".png")
        normal_path = os.path.join(normal_folder, os.path.splitext(file_name)[0] + ".png")
        
        if not (os.path.exists(img_path) and os.path.exists(depth_path) and os.path.exists(normal_path)):
            print(f"Skipping {file_name} - missing input, depth, or normal file")
            continue
        
        # Load image, depth, and normal maps
        try:
            # Load the original TIFF image with compression handling
            image = safe_load_tiff(img_path)
            # print('asdasdasdasd')
            # Ensure image is in a supported data type for PyTorch
            if image.dtype == np.uint16:
                # Convert uint16 to float32 for processing
                image = image.astype(np.float32)
            
            
            # Apply augmentation
            augmented_image = augmenter.transform(
                image=image,
                starting_isd=starting_isd,
                augmented_isd=augmented_isd,
                scaling_factor=scaling_factor,
                translation=translation,
                tensor_output=False
            )
            # Make sure augmented_image is in a valid format for further processing
            if isinstance(augmented_image, torch.Tensor):
                augmented_image = augmented_image.cpu().numpy()
            
            # Normalize to 8-bit for histogram calculation
            if augmented_image.dtype == np.float32 or augmented_image.dtype == np.float64:
                # Assuming the range is [0, 1] from the augmenter
                augmented_image_8bit = (np.clip(augmented_image * 255.0, 0, 255)).astype(np.uint8)
            elif augmented_image.dtype == np.uint16:
                # Normalize from uint16 [0, 65535] to uint8 [0, 255]
                augmented_image_8bit = (np.clip(augmented_image / 65535.0 * 255.0, 0, 255)).astype(np.uint8)
            else:
                # If it's already uint8, use as is
                augmented_image_8bit = augmented_image
            
            # Apply log transformation
            augmented_image_log = linear_to_log(augmented_image_8bit)
            
            # Ensure the image has 3 channels for histogram calculation
            if augmented_image_log.ndim == 2:
                augmented_image_log = np.stack([augmented_image_log] * 3, axis=2)
            elif augmented_image_log.shape[2] == 1:
                augmented_image_log = np.concatenate([augmented_image_log] * 3, axis=2)
            
            # Load depth and normal maps (these don't get augmented in color space)
            depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
            normal_map = cv2.imread(normal_path)
            augmented_image = cv2.cvtColor(augmented_image,cv2.COLOR_BGR2RGB)
            augmented_image = (augmented_image/(np.max(augmented_image)))*255
            cv2.imwrite('dep.png',depth)
            cv2.imwrite('nomal.png',normal_map)
            cv2.imwrite('image.png',augmented_image)
            # Process with depth information
            threshold = 10.0
            masked_image, mask = remove_far_objects(augmented_image, depth, threshold)
            print(np.max(augmented_image))
            # Extract normal map components
            height, width, _ = augmented_image.shape
            up = normal_map[..., 1]
            
            # Apply mask to the up component
            masked_up = up.copy()
            masked_up[mask] = 0
            masked_up = masked_up / 255
            
            # Calculate histograms
            rg_hist, gb_hist, br_hist, weighted_hist = calculate_weighted_2d_histograms_optimized(
                augmented_image, masked_up, 128
            )
            
            # Generate unique output filename based on the original and augmentation parameters
            base_name = os.path.splitext(file_name)[0]
            aug_suffix = f"_aug_{idx}"  # Use index to ensure uniqueness
            
            # Save histograms
            rg_hist_output_path = os.path.join(augmented_hist_rg_folder, f"{aug_name}.png")
            gb_hist_output_path = os.path.join(augmented_hist_gb_folder, f"{aug_name}.png")
            br_hist_output_path = os.path.join(augmented_hist_br_folder, f"{aug_name}.png")
            weight_hist_output_path = os.path.join(augmented_hist_plane_folder, f"{aug_name}.png")
            
            cv2.imwrite(rg_hist_output_path, rg_hist)
            cv2.imwrite(gb_hist_output_path, gb_hist)
            cv2.imwrite(br_hist_output_path, br_hist)
            cv2.imwrite(weight_hist_output_path, weighted_hist)
            
            print(f"Histograms saved for augmented image: {aug_name}")
            
        except Exception as e:
            print(f"Error processing {file_name}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("Processing complete. All augmented histograms saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate histograms for augmented spectral images')
    parser.add_argument('--folder', type=str, default='1', 
                        choices=['1', '2', '3', '4', '5', '6', '7', '8', '9', 'test', 'val'],
                        help='Folder name')
    parser.add_argument('--csv', type=str, default='/home/balamurugan.d/src/val_250310_10x.csv',
                        help='Path to CSV file with augmentation parameters')
    
    args = parser.parse_args()
    choices=['val']
    for i in choices:
        generate_and_save_histograms(args,folders=i)