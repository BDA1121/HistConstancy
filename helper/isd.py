"""
ISD Augmentation Class

*This version uses the direct ISD sampling method. 


Michael Massone

Created: 2025/01/27
Updated: 2025/04/21

Classes:
--------
- AugmentISD_GPU: Performs random augmentations or defined transformations of an inputs images. Designed for use on GPU with Pytorch. 

Usage:
------
- This class processes images by augmenting their Illuminant Spectral Direction (ISD) to create synthetic datasets.
- The augmentation process involves computing ISD vectors from lit and shadow regions of an image, projecting them to a chromaticity plane, and applying transformations such as rotation and scaling to generate new images.
- The augmented images are reconstructed with augmented ISD values and are returned along with the transformation metadata (ISD, rotation, scaling, translation).
- THe 'transform' method can be used to reapply augmentations to original iamges given the image, augmented isd, scaling factor, and translation factor. 

Configuration:
--------------
- distribution: Choose the distribution to draw random rotation values from: 'uniform' or 'normal'.
- rotation_lower_bound: Minimum rotation angle (in degrees) for random rotations.
- rotation_upper_bound: Maximum rotation angle (in degrees) for random rotations.
- initial_anchor_point: Initial anchor point for the chromaticity plane.
- augmented_anchor_point: Anchor point for the reconstructed image.
- seed: Optional seed for random number generation.

Key Features:
--------------
- ISD Computation: The class computes the Illuminant Spectral Direction (ISD) vector by calculating the spectral ratio between lit and shadow regions in log RGB space.
- Log Chromaticity: Converts input images into log chromaticity space before augmenting the ISD, which is then projected and rotated.
- Rotation & Scaling: Randomyl samples a new ISD from the selected dsitributions and construructs a rotation matrix to transform projected image.
- Image Reconstruction: The image is projected and rotated back into 3D space, and the augmented image is reconstructed.
- Saturation Threshold: Percent of saturated pixels is checkted to esnure below threshold of 0.001%.

Important Notes:
----------------
- Ensure the `AugmentISD_GPU` class has been implemented correctly for your specific dataset and augmentation requirements.
- The input image must be in 16-bit format (dtype=np.uint16) for proper processing.
- The input annotations should provide pixel coordinates for lit and shadow regions, which are used to compute the ISD vector.
- The random rotations applied during augmentation are constrained to the specified bounds (`lower_bound` and `upper_bound`).
- The class provides options to either perform random transformations or use predefined rotation angles for augmentation.
 
"""
import cv2
import gc
import pandas as pd
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging


import warnings
# Suppress specific warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="divide by zero")


#########################################################################################################################################################

class AugmentISD_GPU:
    """
    GPU-optimized ISD Augmentation Class using PyTorch.

    This class processes images by converting them to log space, computing the ISD vector from lit and shadow regions,
    projecting the image onto a chromaticity plane, applying rotation and scaling, and reconstructing the image.
    All heavy computations are performed on the GPU.

    Configuration:
      - distribution: 'uniform' or 'normal'
      - initial_anchor_point: initial anchor point for the chromaticity plane
      - augmented_anchor_point: anchor point for the reconstructed image
      - seed: Optional seed for random number generation
    """
    def __init__(self, distribution='uniform', initial_anchor_point=10.4, augmented_anchor_point=10.4, seed=None, device=None):
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.distribution = distribution
        # Store anchor points as torch tensors on the chosen device.
        self.initial_anchor_point = torch.full((3,), initial_anchor_point, device=self.device, dtype=torch.float32)
        self.augmented_anchor_point = torch.full((3,), augmented_anchor_point, device=self.device, dtype=torch.float32)
        if seed is not None:
            self.rng = torch.Generator(device=self.device)
            self.rng.manual_seed(seed)

        
        self._reset_state()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.debug(f"Initialized AugmentISD_GPU on {self.device}")

    def _reset_state(self):
        # Input state
        self.image = None          # Expected to be a torch.Tensor on self.device
        self.annotations = None    # Remains as provided (pandas Series or dict)
        self.filename = None

        # ISD state
        self.lit_pixels = None     # List of tuples (row, col)
        self.shadow_pixels = None  # List of tuples (row, col)
        self.patch_size = None     # Tuple (h, w)
        self.starting_isd = None   # torch.Tensor of shape (3,)
        self.augmented_isd = None  # torch.Tensor of shape (3,)

        # Transformation state
        self.rotation_matrix = None   # torch.Tensor shape (3,3)
        self.distance_scaler = 1
        self.scaling_factor = None
        self.translation = None

        # Transformation constraints (as torch tensors)
        eps = 0.1
        self.lower_bounds = torch.full((3,), 0 + eps, device=self.device, dtype=torch.float32)
        self.upper_bounds = torch.full((3,), 11.1 - eps, device=self.device, dtype=torch.float32)
        self.saturation_threshold = 0.001

        # Image processing state
        self.logRGB_image = None      # torch.Tensor of shape (H, W, 3)
        self.projected_image = None   # torch.Tensor of shape (H, W, 3)
        self.distance_map = None      # torch.Tensor of shape (H, W)
        self.rotated_projection = None  # torch.Tensor of shape (H, W, 3)
        self.reconstructed_image = None  # torch.Tensor of shape (H, W, 3)

    def set_image(self, image):
        # Convert numpy array to torch tensor if necessary and move to device
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).float()
        self.image = image.to(self.device)

    def get_augmented_image(self):
        if self.reconstructed_image is None:
            return None
        return self.log_to_linear(self.reconstructed_image)

    def get_starting_isd(self):
        return self.starting_isd.cpu().numpy()

    def get_augmented_isd(self):
        return self.augmented_isd.cpu().numpy()

    @property
    def rotation_angles(self):
        # Not implemented – can be computed from the rotation matrix if needed.
        return None

    def convert_img_to_log_space(self):
        """
        Converts a 16-bit image (as a torch tensor) to log space.
        """
        if self.image is None:
            raise ValueError("Image not set.")
        eps = 1e-6
        zero_mask = (self.image == 0)
        log_img = torch.log(self.image + eps)
        log_img[zero_mask] = 0
        # Optionally warn if values are out of the expected range.
        if log_img.min() < 0 or log_img.max() > 11.1:
            self.logger.warning("Log image values are outside expected range [0, 11.1].")
        self.logRGB_image = log_img

    @staticmethod
    def log_to_linear(log_image):
        return torch.exp(log_image)

    @staticmethod
    def convert_16bit_to_8bit(image):
        # Using torch operations
        img_normalized = torch.clamp(image / 255, 0, 255)
        return img_normalized.to(torch.uint8)

    def get_lit_shadow_pixel_coordinates(self):
        """
        Extracts lit and shadow pixel locations from annotations.
        If annotations is a pandas Series, expects columns like:
          "lit_row1", "lit_col1", ..., "shad_row1", "shad_col1", etc.
        If annotations is a dict, expects a structure with a "clicks" key.
        Converts pixel coordinates to integers.
        """
        if isinstance(self.annotations, pd.Series):
            self.lit_pixels = [
                (int(self.annotations[f"lit_row{i}"]), int(self.annotations[f"lit_col{i}"]))
                for i in range(1, 7)
            ]
            self.shadow_pixels = [
                (int(self.annotations[f"shad_row{i}"]), int(self.annotations[f"shad_col{i}"]))
                for i in range(1, 7)
            ]
        elif isinstance(self.annotations, dict):
            self.lit_pixels = []
            self.shadow_pixels = []
            for click in self.annotations.get("clicks", []):
                if "lit" in click and "shadow" in click:
                    lit_tuple = tuple(int(v) for v in click["lit"].values())
                    shadow_tuple = tuple(int(v) for v in click["shadow"].values())
                    self.lit_pixels.append(lit_tuple)
                    self.shadow_pixels.append(shadow_tuple)
            self.patch_size = tuple(map(int, self.annotations.get('patch_size').strip("()").split(", ")))
        else:
            raise ValueError("Annotations are in unreadable format")

    def compute_isd(self):
        """
        Computes the ISD vector using the mean difference between lit and shadow pixel values in log space.
        """
        if self.logRGB_image is None:
            raise ValueError("Log-transformed image is not set.")
        if self.lit_pixels is None or self.shadow_pixels is None:
            raise ValueError("Lit or shadow pixel coordinates not set.")
        # Convert lists of pixel coordinates to tensors
        lit_tensor = torch.tensor(self.lit_pixels, device=self.device, dtype=torch.long)
        shadow_tensor = torch.tensor(self.shadow_pixels, device=self.device, dtype=torch.long)
        # Assume self.logRGB_image shape is (H, W, 3)
        lit_values = self.logRGB_image[lit_tensor[:, 0], lit_tensor[:, 1]]  # shape (n, 3)
        shadow_values = self.logRGB_image[shadow_tensor[:, 0], shadow_tensor[:, 1]]  # shape (n, 3)
        pixel_diff = lit_values - shadow_values
        mean_pixel_diff = torch.mean(pixel_diff, dim=0)
        self.starting_isd = mean_pixel_diff / torch.norm(mean_pixel_diff)
    
    # def extract_patch_mean(self, coords, img_tensor, patch_size):
    #         """Extracts mean patch values around each coordinate in a tensor."""
    #         H, W, C = img_tensor.shape
    #         pad_H, pad_W = patch_size[0] // 2, patch_size[1] // 2
    #         # Pad the image to avoid out-of-bounds issues
    #         padded_img = F.pad(img_tensor.permute(2, 0, 1), (pad_W, pad_W, pad_H, pad_H), mode='reflect')
    #         padded_H, padded_W = padded_img.shape[1], padded_img.shape[2]
    #         # Adjust coordinates due to padding
    #         coords = coords + torch.tensor([pad_H, pad_W], device=self.device)
    #         # Extract patches
    #         patches = torch.stack([
    #             padded_img[:, y - pad_H:y + pad_H + 1, x - pad_W:x + pad_W + 1]
    #             for y, x in coords
    #         ])  # Shape: (N, C, H_patch, W_patch)
    #         # Compute mean of each patch along spatial dimensions
    #         return patches.mean(dim=(2, 3))  # Shape: (N, C)
    
    # def compute_isd(self):
    #     """
    #     Computes the ISD vector using the mean difference between lit and shadow pixel values in log space.
    #     Uses a patch-based mean around each pixel, with a patch size defined by self.patch_size (H, W).
    #     """
    #     if self.logRGB_image is None:
    #         raise ValueError("Log-transformed image is not set.")
    #     if self.lit_pixels is None or self.shadow_pixels is None:
    #         raise ValueError("Lit or shadow pixel coordinates not set.")
    #     if not hasattr(self, 'patch_size'):
    #         raise ValueError("Patch size (H, W) must be defined in self.patch_size")

    #     # Convert lists of pixel coordinates to tensors
    #     lit_tensor = torch.tensor(self.lit_pixels, device=self.device, dtype=torch.long)
    #     shadow_tensor = torch.tensor(self.shadow_pixels, device=self.device, dtype=torch.long)
    #     # Compute mean values for lit and shadow regions using patch averaging
    #     lit_values = self.extract_patch_mean(lit_tensor, self.logRGB_image, self.patch_size)
    #     shadow_values = self.extract_patch_mean(shadow_tensor, self.logRGB_image, self.patch_size)
    #     # Compute mean difference
    #     pixel_diff = lit_values - shadow_values
    #     mean_pixel_diff = torch.mean(pixel_diff, dim=0)
    #     self.starting_isd = mean_pixel_diff / torch.norm(mean_pixel_diff)

    def project_to_plane(self):
        """
        Projects the log RGB image onto the chromaticity plane defined by the starting ISD vector.
        """
        if self.logRGB_image is None:
            raise ValueError("Log image not set.")
        if self.starting_isd is None:
            raise ValueError("Starting ISD not computed.")
        shifted_log_rgb = self.logRGB_image - self.initial_anchor_point.view(1,1,3)
        dot_product = torch.einsum('ijk,k->ij', shifted_log_rgb, self.starting_isd)
        dot_product_reshaped = dot_product.unsqueeze(-1)
        projection = dot_product_reshaped * self.starting_isd.view(1,1,3)
        self.projected_image = shifted_log_rgb - projection
        self.distance_map = torch.einsum('ijk,k->ij', projection, self.starting_isd)
        self.projected_image += self.initial_anchor_point.view(1,1,3)
    
    def fit_distribution_to_bounds(self, points):
        """
        Scales and translates a 3D point distribution (points) to fit within specified bounds.
        Args:
            points (torch.Tensor): shape (n, 3)
        Returns:
            torch.Tensor: Transformed points (n, 3)
        """
        unscaled_min, _ = torch.min(points, dim=0)
        unscaled_max, _ = torch.max(points, dim=0)
        scale_factors = (self.upper_bounds - self.lower_bounds) / (unscaled_max - unscaled_min)
        self.scaling_factor = torch.clamp(scale_factors.min(), max=1.0)
        dot = torch.matmul(points, self.augmented_isd)
        parallel_components = dot.unsqueeze(1) * self.augmented_isd.unsqueeze(0)
        orthogonal_components = points - parallel_components
        scaled_parallel_components = self.scaling_factor * parallel_components
        scaled_points = scaled_parallel_components + orthogonal_components
        if self.translation is None:
            scaled_max, _ = torch.max(scaled_points, dim=0)
            scaled_min, _ = torch.min(scaled_points, dim=0)
            scaled_midpoint = (scaled_max + scaled_min) / 2
            target_midpoint = (self.upper_bounds + self.lower_bounds) / 2
            shift_magnitude = torch.dot(target_midpoint - scaled_midpoint, self.augmented_isd)
            self.translation = shift_magnitude * self.augmented_isd
        final_points = scaled_points + self.translation
        return final_points
    
    def rotate_image(self):
        """
        Rotates the projected image using the rotation matrix.
        """
        if self.projected_image is None:
            raise ValueError("Projected image not set.")
        H, W, _ = self.projected_image.shape
        reshaped_img = self.projected_image.view(-1, 3)
        translated_points = reshaped_img - self.initial_anchor_point.view(1,3)
        rotated_points = torch.matmul(translated_points, self.rotation_matrix.t())
        augmented_anchor_point = self.augmented_anchor_point.expand_as(rotated_points)
        rotated_img = rotated_points + augmented_anchor_point
        self.rotated_projection = rotated_img.view(H, W, 3)
    
    def reconstruct_image(self):
        """
        Reconstructs the augmented image by combining the rotated projection and the scaled distance map,
        then fitting the pixel distribution within 16-bit bounds.
        """
        if self.rotated_projection is None or self.distance_map is None:
            raise ValueError("Rotated projection or distance map not set.")
        scaled_distance_map = self.distance_map * self.distance_scaler
        augmented_image = self.rotated_projection + (scaled_distance_map.unsqueeze(-1) * self.augmented_isd.view(1,1,3))
        H, W, _ = augmented_image.shape
        flat_points = augmented_image.view(-1, 3)
        scaled_flat = self.fit_distribution_to_bounds(flat_points)
        self.reconstructed_image = scaled_flat.view(H, W, 3)
    
    def get_random_isd(self, power=1, eps=1e-1):
        """
        Generates a random ISD vector using the specified distribution.
        Returns:
            torch.Tensor: A normalized ISD vector of shape (3,)
        """
        if self.distribution == 'normal':
            Z_hat = torch.normal(mean=0.5, std=0.2, size=(3,), generator=self.rng, device=self.device)
            Z_hat = torch.clamp(Z_hat, min=epsilon, max=1-epsilon)
        elif self.distribution == 'uniform':
            Z_hat = eps + (1 - 2 * eps) * torch.rand(3, device=self.device)
            # for comparing methods
            # Z_hat = self.rng.uniform(0 + eps, 1 - eps, size=3)
            # Z_hat = torch.tensor(Z_hat, dtype=torch.float32, device=self.device)     

        else:
            raise ValueError("Unsupported distribution type. Use 'normal' or 'uniform'.")
        Z_hat = Z_hat / torch.norm(Z_hat)
        return Z_hat
    
    def generate_rotation_matrix(self):
        """
        Generates a rotation matrix based on the augmented ISD vector.
        Returns:
            torch.Tensor: Rotation matrix of shape (3,3)
        """
        if self.augmented_isd is None:
            self.augmented_isd = self.get_random_isd()
        Y_hat = torch.tensor([0, 1, 0], device=self.device, dtype=torch.float32)
        if torch.allclose(torch.linalg.cross(Y_hat, self.augmented_isd), torch.zeros(3, device=self.device)):
            Y_hat = torch.tensor([1, 0, 0], device=self.device, dtype=torch.float32)
        X_hat = torch.linalg.cross(Y_hat, self.augmented_isd)
        X_hat = X_hat / torch.norm(X_hat)
        Y_prime = torch.linalg.cross(self.augmented_isd, X_hat)
        Y_prime = Y_prime / torch.norm(Y_prime)
        self.rotation_matrix = torch.stack([X_hat, Y_prime, self.augmented_isd], dim=0)
        return self.rotation_matrix
    
    def is_saturated(self):
        """
        Checks if the reconstructed image's pixel values (in log space) are out of the expected bounds.
        Returns:
            bool: True if the percentage of saturated pixels exceeds the threshold.
        """

        if self.reconstructed_image is None:
            raise ValueError("Reconstructed image not set.")
        H, W, _ = self.reconstructed_image.shape
        total_pixels = H * W
        # Create a mask for saturation (per pixel, not per channel)
        mask = (self.reconstructed_image < 0) | (self.reconstructed_image > 11.1)
        # Count pixels where any channel is saturated
        total_saturated = torch.sum(mask.any(dim=-1)).item()
        # Compute percentage of saturated pixels
        percent_saturated = (total_saturated / total_pixels) * 100
        return percent_saturated >= self.saturation_threshold
    
    def move_isd_values_midpoint(self, r=0.5):
        """
        Contracts the augmented ISD vector toward the starting ISD vector by a factor r and updates the rotation matrix.
        """
        Z_hat = self.starting_isd + r * (self.augmented_isd - self.starting_isd)
        Z_hat = Z_hat / torch.norm(Z_hat)
        self.augmented_isd = Z_hat
        Y_hat = torch.tensor([0, 1, 0], device=self.device, dtype=torch.float32)
        X_hat = torch.linalg.cross(Y_hat, Z_hat)
        X_hat = X_hat / torch.norm(X_hat)
        Y_prime = torch.linalg.cross(Z_hat, X_hat)
        Y_prime = Y_prime / torch.norm(Y_prime)
        self.rotation_matrix = torch.stack([X_hat, Y_prime, Z_hat], dim=0)
    
    def saturation_check(self, random=True, n=100, m=10):
        if random:
            outer_count = 0
            inner_count = 0
            while self.is_saturated() and outer_count < n:
                if outer_count == n - 1:
                    while inner_count < m:
                        self.move_isd_values_midpoint()
                        self.rotate_image()
                        self.reconstruct_image()
                        if not self.is_saturated():
                            break
                        inner_count += 1
                    if inner_count == m:
                        return False
                else:
                    self.generate_rotation_matrix()
                    self.rotate_image()
                    self.reconstruct_image()
                outer_count += 1
        else:
            if self.is_saturated():
                self.logger.debug("Unable to maintain image within 16-bit range with specified rotation.")
                return False
        return True
    
    def augment(self, image: np.ndarray, annotations, filename=None):
        """
        Augments the given image using ISD transformations.
        Args:
            image (np.ndarray): A 16-bit RGB image (H, W, 3).
            annotations (pd.Series or dict): Annotation data for lit and shadow pixel coordinates.
            filename (str): Optional filename.
        Returns:
            tuple: (augmented_image, augmented_isd, scaling_factor, translation, rotation_matrix)
        """
        self._validate_inputs(image, annotations)
        self._reset_state()
        # Convert image to torch tensor if needed
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).float()
        self.image = image.to(self.device)
        self.annotations = annotations  # Remains as provided (pandas Series or dict)
        self.filename = filename

        # Processing steps
        self.convert_img_to_log_space()
        self.get_lit_shadow_pixel_coordinates()
        self.compute_isd()
        self.project_to_plane()
        self.generate_rotation_matrix()
        self.rotate_image()
        self.reconstruct_image()
        if not self.saturation_check():
            self.logger.debug(f"No suitable rotation found for {filename}")
            return (None, None, None, None, None)
        self.augmented_image = self.log_to_linear(self.reconstructed_image)

        return (
            self.augmented_image.cpu().numpy(),
            self.augmented_isd.cpu().numpy(),
            float(self.scaling_factor),
            self.translation.cpu().numpy(),
            self.rotation_matrix.cpu().numpy()
            )    

    def _validate_inputs(self, image, annotations):
        if not isinstance(image, np.ndarray) or image.ndim != 3:
            raise ValueError("Image must be a 3D NumPy array")
        if image.dtype != np.uint16:
            raise ValueError("Image must be 16-bit (dtype=np.uint16)")
        if not (isinstance(annotations, pd.Series) or isinstance(annotations, dict)):
            raise ValueError(f"Annotations must be a pandas Series or dict. Got {type(annotations)}")

    
    def transform(self, image, starting_isd, augmented_isd, scaling_factor, translation, tensor_output=False):
        """
        Given the image and its augmentation parameters, returns the augmented version.
        """
        self._reset_state()
        
        # Convert NumPy arrays to torch tensors and move to self.device.
        self.image = torch.from_numpy(image).float().to(self.device)
        self.starting_isd = torch.from_numpy(starting_isd).float().to(self.device)
        self.augmented_isd  = torch.from_numpy(augmented_isd).float().to(self.device)
        self.translation = torch.from_numpy(translation).float().to(self.device)
        self.scaling_factor = scaling_factor  # remains as float

        # IF the starting_isd == augmented_isd (no augmentation), return original image
        if torch.allclose(self.starting_isd, self.augmented_isd, atol=1e-6):
            self.augmented_image = self.image
            return self.image.cpu().numpy() if not tensor_output else self.image

        # Process through augmentation steps
        self.convert_img_to_log_space()
        self.project_to_plane()
        self.generate_rotation_matrix()
        self.rotate_image()
        self.reconstruct_image()
        
        self.augmented_image = self.log_to_linear(self.reconstructed_image)
        if tensor_output:
            return self.augmented_image
        return self.augmented_image.cpu().numpy()
    
    def resample_isd(self, image, annotations):
        "Rechecks the augmented isd and image validity by resampling the augmented image and computing isd form the original annotations."

        self._validate_inputs(image, annotations)
        self._reset_state()
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).float()
        self.annotations = annotations
        self.image = image.to(self.device)

        self.convert_img_to_log_space()
        self.get_lit_shadow_pixel_coordinates()
        self.compute_isd()
        return self.starting_isd.cpu().numpy()

if __name__ == "__main__":
    pass
    