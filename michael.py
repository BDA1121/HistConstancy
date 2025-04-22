""" 
Michael Massone  
MS Project - Spring 2025  
Created: 2025/02/18  
Updated: 2025/03/24  

This module defines a PyTorch `Dataset` for training models on augmented spectral image data with optional depth guidance.  
It supports GPU-accelerated ISD augmentations and integrates image/depth preprocessing transforms including cropping, flipping, 
and normalization. Images are read from 16-bit TIFFs and depth maps from 8-bit PNGs, and are cached in memory for fast access.

The dataset is compatible with models requiring both image and depth inputs (e.g., CNN-ViT fusion models), 
and reads transformation parameters from a CSV file.

Class Descriptions:
-------------------
• AugmentedSpectralDataset  
  Main dataset class for loading image-ISD-depth triplets from disk using CSV metadata. Supports optional ISD augmentation, 
  depth loading, image caching, and GPU-based normalization and transformation.

• ToTensor  
  Converts image, depth map, and ISD data from NumPy arrays to PyTorch tensors in channel-first format.

• RandomCropGPU 
  Randomly crops both image and depth map tensors to a target size on GPU.

• RandomHorizontalFlipGPU
  Randomly flips both image and depth map tensors horizontally with a configurable probability.

• ToLogRGB  
  Converts linear 16-bit RGB images to log RGB.
"""

####################################################################################################################
# Imports
import os
import sys
import glob
import cv2
import pandas as pd 
import numpy as np
import torch
import tifffile
import random
import logging
import concurrent.futures

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision.transforms.functional as F

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="divide by zero")

# Modules
module_dir = "/home/massone.m/spectral_ratio/syn_data_pipeline/src"
sys.path.append(module_dir)
from isd_augmentation_xml_class import AugmentISD_GPU

####################################################################################################################
# Transform Classes
####################################################################################################################
class ToTensor(object):
    """
    Convert NumPy ndarrays or PyTorch tensors in a sample to PyTorch tensors.

    This transformation is applied to a sample dictionary containing:
    - 'image': Ensures the image is a PyTorch tensor of shape (C, H, W). Also applied to 'depth_map', if it exists. 
    - 'isd': Ensures the ISD array is a PyTorch tensor.


    Usage:
        transform = ToTensor()
        sample = transform(sample)
    """
    def __call__(self, sample):
        """
        Args:
            sample (dict): A dictionary containing:
                - 'image': A NumPy array or PyTorch tensor of shape (H, W, C) or (C, H, W).
                - 'isd': A NumPy array or PyTorch tensor.
                - 'depth_map': A NumPy array or Pytorch tensor of shape (H, W, C) or (C, H, W).

        Returns:
            dict: A dictionary with:
                - 'image': A PyTorch tensor of shape (C, H, W).
                - 'isd': A PyTorch tensor.
                - 'depth_map': A PyTorch tensor of chape (C, H, W).
        """
        for key in ['image', 'isd', 'depth_map']:
            if key in sample and isinstance(sample[key], np.ndarray):
                sample[key] = torch.from_numpy(sample[key])

        if sample['image'].ndim == 3 and sample['image'].shape[-1] in {1, 3, 4}:
            sample['image'] = sample['image'].permute(2, 0, 1)

        if 'depth_map' in sample and sample['depth_map'].ndim == 3 and sample['depth_map'].shape[-1] in {1, 3, 4}:
            sample['depth_map'] = sample['depth_map'].permute(2, 0, 1)

        return sample

class RandomCropGPU(torch.nn.Module):
    """Randomly crop the image to target size on GPU."""

    def __init__(self, target_size=(516, 516)):
        super().__init__()
        self.target_size = target_size
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Initialized {self.__class__.__name__} with target_size={target_size}")

    def forward(self, sample):
        """
        Args:
            sample (dict): Dictionary with 'image', 'depth_map' and 'isd' tensors on GPU.
        
        Returns:
            dict: Cropped image and ISD, both on GPU.
        """
        img = sample['image']
        h, w = img.shape[-2:]

        if h < self.target_size[0] or w < self.target_size[1]:
            raise ValueError(f"Image size ({h}, {w}) is smaller than crop size {self.target_size}.")
        top = random.randint(0, h - self.target_size[0])
        left = random.randint(0, w - self.target_size[1])
        sample['image'] = img[..., top:top + self.target_size[0], left:left + self.target_size[1]]
        sample['isd'] = sample['isd']
        if 'depth_map' in sample:
            sample['depth_map'] = sample['depth_map'][..., top:top + self.target_size[0], left:left + self.target_size[1]]
        return sample

class ToLogRGB(torch.nn.Module):
    """Confvert 16-bit linear RGB image to log RGB."""

    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Initialized {self.__class__.__name__}")
    
    def forward(self, sample):
        """ 
        Args:
            sample (dict): Dictionary with 'image' and 'isd' tensors on GPU.
        Returns:
            dict: Log image and ISD, both on GPU.
        """
        augmented_image = sample['image'].float() * 65535.0
        zero_mask = (augmented_image == 0)
        log_img = torch.log(augmented_image)
        log_img[zero_mask] = 0
        log_max = torch.log(torch.tensor(65535.0))  
        log_img_normalized = torch.clamp(log_img / log_max, 0, 1)
        sample['image'] = log_img_normalized
        return sample

class RandomHorizontalFlipGPU(torch.nn.Module):
    """Randomly flip the image horizontally with a given probability on GPU."""

    def __init__(self, probability=0.5):
        super().__init__()
        self.probability = probability
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Initialized {self.__class__.__name__} with p={probability}")

    def forward(self, sample):
        """
        Args:
            sample (dict): Dictionary with 'image' and 'isd' tensors on GPU.
        
        Returns:
            dict: Flipped image and ISD (if applied), both on GPU.
        """
        if torch.rand(1, device=sample['image'].device) < self.probability:
            sample['image'] = torch.flip(sample['image'], dims=[-1])
            if 'depth_map' in sample:
                sample['depth_map'] = torch.flip(sample['depth_map'], dims=[-1])
        return sample


####################################################################################################################
# Dataset Class
####################################################################################################################

class AugmentedSpectralDataset(Dataset):
    """
    A PyTorch Dataset class to load and process augmented spectral image data.

    This dataset reads image file paths and augmentation parameters from a CSV file and 
    applies optional augmentations and transformations.

    Usage:
        dataset = AugmentedSpectralDataset(csv_file="data.csv", root_dir="images/", augment=True, image_transforms=None)
        sample = dataset[0]  # Access the first sample
    """
    def __init__(self, csv_file, root_dir, depth_dir=None, use_depth=False, augment=True, ds_device="cpu", image_transforms=None):
        """
        Args:
            csv_file (str): Path to the CSV file containing image paths and augmentation parameters.
            root_dir (str): Root directory where image files are stored.
            augment (bool): If True, applies ISD augmentations. Set to False for non-training splits.
            image_transforms (callable, optional): Optional image transformations (e.g., cropping, normalization).
        """
        self.augmentations_csv = pd.read_csv(csv_file, converters={
                                            'starting_isd': self.str_to_array,
                                            'augmented_isd': self.str_to_array,
                                            'translation': self.str_to_array,
                                            'scaling_factor': self.float_conversion
                                        })
        self.root_dir = root_dir
        self.depth_dir = depth_dir
        self.use_depth = use_depth
        self.device = torch.device(ds_device) if ds_device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize augmentation object if augment=True
        self.augment = AugmentISD_GPU(device=ds_device)

        self.image_transforms = image_transforms

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Initialized {self.__class__.__name__} on {self.device}")

        self.image_cache = {}  # Dictionary to store images in memory
        self.depth_cache = {}
        self._preload_images()

    def set_device(self, device):
        """Change the device."""
        self.device = device
        self.augment = AugmentISD_GPU(device=device)
    
    def _preload_images(self):
        """Preload all original images and matching depth maps into memory with a progress bar."""
        self.logger.info("Preloading images and depth maps into memory...")

        img_paths = list({
            os.path.join(self.root_dir, self.augmentations_csv.iloc[idx, 2], self.augmentations_csv.iloc[idx, 0])
            for idx in range(len(self.augmentations_csv))
        })

        for img_path in tqdm(img_paths, desc="Loading Images", unit="img"):
            if img_path not in self.image_cache:
                try:
                    # Load .tiff image
                    image = tifffile.imread(img_path)

                    if self.use_depth:
                        # Construct matching .png path for depth map
                        filename_no_ext = os.path.splitext(os.path.basename(img_path))[0]
                        folder = os.path.basename(os.path.dirname(img_path))
                        depth_path = os.path.join(self.depth_dir, folder, f"{filename_no_ext}.png")

                        # Load .png depth map (grayscale or RGB)
                        depth_map = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
                        if depth_map is None:
                            self.logger.warning(f"Missing or unreadable depth map: {depth_path}")
                            continue

                        if depth_map.ndim == 3 and depth_map.shape[2] == 3:
                            depth_map = cv2.cvtColor(depth_map, cv2.COLOR_BGR2RGB)

                        self.image_cache[img_path] = (image, depth_map)
                    else:
                        self.image_cache[img_path] = image
                except Exception as e:
                    self.logger.error(f"Error loading image/depth for {img_path}: {e}")

        self.logger.info(f"Loaded {len(self.image_cache)} image-depth pairs into memory.")


    @staticmethod
    def str_to_array(s):
        """
        Convert a string representation of a list into a NumPy array.

        Args:
            s (str): A string formatted as '[1.0, 2.0, 3.0]'.

        Returns:
            np.ndarray: A NumPy array containing the parsed values.
        """
        return np.fromstring(s.strip('[]'), sep=' ')

    @staticmethod
    def float_conversion(value):
        """
        Safely converts a value to float. If conversion fails, returns NaN.

        Args:
            value (str): The value to convert.

        Returns:
            float or np.nan: Converted float value, or NaN if conversion fails.
        """
        try:
            return float(value)
        except ValueError:
            return np.nan
    @staticmethod
    def move_dict_to_cpu(tensor_dict):
        """Moves all tensors in a dictionary to CPU."""
        return {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in tensor_dict.items()}

    def __len__(self):
        """
        Returns:
            int: The total number of samples in the dataset.
        """
        return len(self.augmentations_csv)

    def __getitem__(self, idx):
        """
        Args:
            idx (int or torch.Tensor): Index of the sample to retrieve.

        Returns:
            dict: A dictionary containing:
                - 'image': The processed image as a NumPy array.
                - 'isd': The ISD array.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Construct image path
        img_name = os.path.join(self.root_dir,
                                self.augmentations_csv.iloc[idx, 2],  # Folder name
                                self.augmentations_csv.iloc[idx, 0])  # File name

        if self.use_depth:
            image, depth_map = self.image_cache[img_name]
            depth_map = torch.tensor(depth_map, dtype=torch.float32, device=self.device)
            if depth_map.ndim == 2:
                depth_map = depth_map.unsqueeze(0)
            depth_map = depth_map / 255.0
        else:
            image = self.image_cache[img_name]

        # Load augmentation parameters
        starting_isd, augmented_isd, translation, scaling_factor = self.augmentations_csv.iloc[idx, 3:].values
        assert not np.any(np.isnan(starting_isd)), f"__getitem__: NaN found in 'starting_isd' = {starting_isd} at 'idx' = {idx}"
        isd = torch.tensor(starting_isd, dtype=torch.float32, device=self.device)

        # Apply ISD augmentation if enabled
        if self.augment:
            image = self.augment.transform(image=image, 
                                           starting_isd=starting_isd, 
                                           augmented_isd=augmented_isd, 
                                           scaling_factor=scaling_factor, 
                                           translation=translation,
                                           tensor_output=True)
            isd = torch.tensor(augmented_isd, dtype=torch.float32, device=self.device)                            
        image = image.to(self.device)

        # Normalize image (16-bit TIFF normalization)
        # image = np.clip(image.astype(np.float32) / 65535.0, 0, 1)
        image = torch.clamp(image.float() / 65535.0, min=0, max=1)

        # Create sample dictionary
        sample = {'image': image, 'isd': isd}
        if self.use_depth:
            sample['depth_map'] = depth_map
        # self.logger.debug(f"Image dtype before transforms: {sample['image'].dtype}")

        # Apply additional image transformations (e.g., cropping, flipping)
        if self.image_transforms:
            sample = self.image_transforms(sample)
        # self.logger.debug(f"Image dtype after transforms: {sample['image'].dtype}")
        
        # Move tensors back to CPU
        # sample = self.move_dict_to_cpu(sample)

        assert not torch.any(torch.isnan(sample['image'])), "Image Tensor contains NaN values!"
        assert not torch.any(torch.isnan(sample['isd'])), "ISD Tensor contains NaN values!"
        if self.use_depth:
            assert not torch.any(torch.isnan(sample['depth_map'])), "Depth Tensor contains NaN values!"

        return sample

####################################################################################################################
# End of Class
####################################################################################################################
if __name__ == "__main__":

    # Test class
    image_dir = "/work/SuperResolutionData/spectralRatio/data/images_for_training/"
    train_csv_path = "/home/massone.m/spectral_ratio/syn_data_pipeline/datasets/train_250304.csv"
    val_csv_path = "/home/massone.m/spectral_ratio/syn_data_pipeline/datasets/val_250304.csv"
    depth_dir = "/work/SuperResolutionData/spectralRatio/data/depth_for_training"
    use_depth=True
    ds_device = 'cuda'

    composed_transforms = transforms.Compose([
                                            ToTensor(),
                                            RandomCropGPU(target_size=(224, 224)),
                                            RandomHorizontalFlipGPU()
                                            ])        
    train_ds = AugmentedSpectralDataset(
                                        train_csv_path, 
                                        image_dir,
                                        depth_dir,
                                        use_depth=use_depth, 
                                        augment=True,
                                        ds_device=ds_device, 
                                        image_transforms=composed_transforms
                                        )      
    # val_ds = AugmentedSpectralDataset(
    #                                     val_csv_path, 
    #                                     image_dir,
    #                                     depth_dir,
    #                                     use_depth=use_depth, 
    #                                     augment=False,
    #                                     ds_device=ds_device,
    #                                     image_transforms=composed_transforms
    #                                     )


    # test train
    dataloader = DataLoader(train_ds, batch_size=32,
                            shuffle=True, num_workers=0)

    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched['image'].size(),
            sample_batched['depth_map'].size(),
            sample_batched['isd'].size())
        print(f"Depth map values: /n{sample_batched['depth_map']}")
        
        if i_batch == 5:
            print("Test with augmentation pass!")
            break
    
    # # Test val
    # dataloader = DataLoader(val_ds, batch_size=4,
    #                         shuffle=True, num_workers=0)

    # for i_batch, sample_batched in enumerate(dataloader):
    #     print(i_batch, sample_batched['image'].size(),
    #         sample_batched['depth_map'].size(),
    #         sample_batched['isd'].size())
        
    #     if i_batch == 5:
    #         print("Test without augmentation pass!")
    #         break