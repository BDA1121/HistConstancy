import cv2
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
from hist.Depth_Anything_V2.pipeline import load_depth_anything_model, infer_depth
from colored_test import compute_surface_normals, colorize_normals, create_normal_weights
from weighted_hist import remove_far_objects,calculate_weighted_2d_histograms_optimized
from PIL import Image
import pandas as pd

def linear_to_log(image):
    image_float = image.astype(np.float32) / 255.0
    image_log = np.log1p(image_float)
    image_log_scaled = (image_log / np.log1p(1.0)) * 255.0
    return image_log_scaled.astype(np.uint8)

class HistogramDataset(Dataset):
    def __init__(self, image_dir, depth_dir, normal_dir, csv_file):
        self.image_dir = image_dir
        self.depth_dir = depth_dir
        self.normal_dir = normal_dir
        self.image_files = sorted([f.lower() for f in os.listdir(image_dir) if f.lower().endswith(('.tif'))])
        self.model = load_depth_anything_model()
        self.csv_data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        depth_name = os.path.splitext(img_name)[0] + ".png"
        normal_name = os.path.splitext(img_name)[0] + ".png"
        depth_path = os.path.join(self.depth_dir, depth_name)
        normal_path = os.path.join(self.normal_dir, normal_name)

        image = cv2.imread(img_path)
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # image = linear_to_log(image)

        if os.path.exists(depth_path) and os.path.exists(normal_path):
            depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
            normal_map = cv2.imread(normal_path)
        else:
            print(f"Error: Could not load image.{img_path}")
            exit()

        threshold = 10.0 
        masked_image, mask = remove_far_objects(image, depth, threshold)

        height, width, _ = image.shape
        up = np.zeros((height, width), dtype=np.uint8)
        front = np.zeros((height, width), dtype=np.uint8)
        side = np.zeros((height, width), dtype=np.uint8)
        up = (normal_map[...,1])
        front = (normal_map[...,2])
        side= (normal_map[...,0])

        masked_up = up.copy()
        masked_up[mask] =0
        masked_up=masked_up/255
        rg_hist, gb_hist, br_hist,_ = calculate_weighted_2d_histograms_optimized(image, masked_up, 64)

        row = self.csv_data[self.csv_data['filename'] == img_name]
        if not row.empty:
            r_avg = row['linear_sr_r_avg'].values[0]
            g_avg = row['linear_sr_g_avg'].values[0]
            b_avg = row['linear_sr_b_avg'].values[0]
        else:
            r_avg, g_avg, b_avg = 0, 0, 0

        # Concatenate histograms as 3 channels
        hist_tensor = torch.stack([torch.tensor(rg_hist, dtype=torch.float32),
                                    torch.tensor(gb_hist, dtype=torch.float32),
                                    torch.tensor(br_hist, dtype=torch.float32)], dim=0)

        # Target tensor with average color values
        target_tensor = torch.tensor([r_avg, g_avg, b_avg], dtype=torch.float32)

        return hist_tensor, target_tensor

if __name__ == "__main__":
    image_dir = "/work/SuperResolutionData/spectralRatio/data/images_for_training/folder_3"
    depth_dir = "/work/SuperResolutionData/spectralRatio/data/depth_for_training/folder_3"
    normal_dir = "/work/SuperResolutionData/spectralRatio/data/surface_norm_for_training/folder_3"
    csv_file = "/work/SuperResolutionData/spectralRatio/data/annotation/train_spectral_ratio.csv"

    dataset = HistogramDataset(image_dir, depth_dir, normal_dir, csv_file)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    print(f"Dataset contains {len(dataset)} images")

    for hist_batch, target_batch in dataloader:
        print("Histogram Batch shape:", hist_batch.shape)
        print("Target Batch shape:", target_batch[1])


# import cv2
# import numpy as np
# import os
# import torch
# from torch.utils.data import Dataset, DataLoader
# from hist.Depth_Anything_V2.pipeline import load_depth_anything_model, infer_depth
# from colored_test import compute_surface_normals, colorize_normals, create_normal_weights
# from PIL import Image
# from weighted_hist import plotting

# def linear_to_log(image):
#     """Converts a linear image to a log-encoded image."""
#     image_float = image.astype(np.float32) / 255.0
#     image_log = np.log1p(image_float)
#     image_log_scaled = (image_log / np.log1p(1.0)) * 255.0
#     return image_log_scaled.astype(np.uint8)

# def calculate_weighted_2d_histograms_optimized(image, weights, bins=64):
#     """Calculates weighted 2D histograms for RG, GB, and BR color pairs (optimized)."""
#     if image.shape[:2] != weights.shape:
#         raise ValueError("Image and weights must have the same height and width.")

#     r = image[:, :, 2]
#     g = image[:, :, 1]
#     b = image[:, :, 0]

#     r_bins = np.floor(r / (256/bins)).astype(int)
#     g_bins = np.floor(g / (256/bins)).astype(int)
#     b_bins = np.floor(b / (256/bins)).astype(int)

#     rg_hist = np.zeros((bins, bins), dtype=float)
#     gb_hist = np.zeros((bins, bins), dtype=float)
#     br_hist = np.zeros((bins, bins), dtype=float)

#     np.add.at(rg_hist, (r_bins, g_bins), weights)
#     np.add.at(gb_hist, (g_bins, b_bins), weights)
#     np.add.at(br_hist, (b_bins, r_bins), weights)

#     return rg_hist, gb_hist, br_hist

# class HistogramDataset(Dataset):
#     def __init__(self, image_dir, depth_dir, normal_dir):
#         self.image_dir = image_dir
#         self.depth_dir = depth_dir
#         self.normal_dir = normal_dir
#         self.image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.tif'))]) #only tif files
#         self.model = load_depth_anything_model()

#     def __len__(self):
#         return len(self.image_files)

#     def __getitem__(self, idx):
#         img_name = self.image_files[idx]
#         img_path = os.path.join(self.image_dir, img_name)
#         depth_name = os.path.splitext(img_name)[0] + ".png" #change extension
#         normal_name = os.path.splitext(img_name)[0] + ".png" #change extension
#         depth_path = os.path.join(self.depth_dir, depth_name)
#         print(depth_path)
#         normal_path = os.path.join(self.normal_dir, normal_name)

#         image = np.array(Image.open(img_path)) #open tiff using PIL
#         image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) #convert to bgr
#         image = linear_to_log(image)

#         if os.path.exists(depth_path) and os.path.exists(normal_path):
#             # Load existing depth and normal maps
#             depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
#             normal_map = cv2.imread(normal_path)
#             print(f"Loaded existing depth and normal maps for {img_name}")
#         else:
#             # Compute depth and normal maps
#             print(f"Computing depth and normal maps for {img_name}")
#             depth = infer_depth(self.model, image)
#             depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)
#             depth = depth.astype(np.float32)

#             normal_map = compute_surface_normals(depth)
#             colored_normals = colorize_normals(normal_map)

#             #save generated depth and normal maps.
#             cv2.imwrite(depth_path, depth.astype(np.uint8))
#             cv2.imwrite(normal_path, colored_normals.astype(np.uint8))

#         weights = create_normal_weights(normal_map)

#         rg_hist, gb_hist, br_hist = calculate_weighted_2d_histograms_optimized(image, weights, 64)

#         plotting(rg_hist, gb_hist, br_hist,"histDataloader.png")

#         rg_hist = torch.tensor(rg_hist, dtype=torch.float32)
#         gb_hist = torch.tensor(gb_hist, dtype=torch.float32)
#         br_hist = torch.tensor(br_hist, dtype=torch.float32)

#         return rg_hist, gb_hist, br_hist

# if __name__ == "__main__":
#     image_dir = "/work/SuperResolutionData/spectralRatio/data/images_for_annotation/split_annotation/split_images/folder_3"
#     depth_dir = "/work/SuperResolutionData/spectralRatio/data/images_for_annotation/split_annotation/split_images_depth/folder_3"
#     normal_dir = "/work/SuperResolutionData/spectralRatio/data/images_for_annotation/split_annotation/split_images_sn/folder_3"

#     dataset = HistogramDataset(image_dir, depth_dir, normal_dir)
#     dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

#     for rg_hist_batch, gb_hist_batch, br_hist_batch in dataloader:
#         print("RG Hist Batch shape:", rg_hist_batch.shape)
#         print("GB Hist Batch shape:", gb_hist_batch.shape)
#         print("BR Hist Batch shape:", br_hist_batch.shape)