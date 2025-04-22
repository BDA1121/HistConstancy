import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats
import time
from tdhist import build_planar_axes, get_min_max_projected_boundaries, get_hist_upper_bound, get_plane_projected_histogram
from hist.Depth_Anything_V2.pipeline import load_depth_anything_model, infer_depth 
from colored_test import compute_surface_normals,colorize_normals,create_normal_weights

def linear_to_log(image):
    """Converts a linear image to a log-encoded image."""
    image_float = image.astype(np.float32) / 255.0  # Normalize to [0, 1]
    image_log = np.log1p(image_float)  # Apply log1p (log(1 + x))
    image_log_scaled = (image_log / np.log1p(1.0)) * 255.0  # Scale back to [0, 255]
    return image_log_scaled.astype(np.uint8)

def calculate_2d_histograms(image,):
    """Calculates 2D histograms for RG, GB, and BR color pairs.

    Args:
        image: The input color image (NumPy array, BGR format).

    Returns:
        A tuple containing the 2D histograms for RG, GB, and BR.
    """

    r = image[:, :, 2]  # Red channel
    g = image[:, :, 1]  # Green channel
    b = image[:, :, 0]  # Blue channel

    rg_hist = np.histogram2d(r.flatten(), g.flatten(), bins=128, range=[[0, 256], [0, 256]])[0]
    gb_hist = np.histogram2d(g.flatten(), b.flatten(), bins=128, range=[[0, 256], [0, 256]])[0]
    br_hist = np.histogram2d(b.flatten(), r.flatten(), bins=128, range=[[0, 256], [0, 256]])[0]

    return rg_hist, gb_hist, br_hist

def calculate_weighted_2d_histograms(image, weights,bins=64):
    """Calculates weighted 2D histograms for RG, GB, and BR color pairs.

    Args:
        image: The input color image (NumPy array, BGR format).
        weights: A NumPy array of the same shape as the image, 
                 representing the weights for each pixel.

    Returns:
        A tuple containing the weighted 2D histograms for RG, GB, and BR.
    """

    if image.shape[:2] != weights.shape:  # Check only the 2D shape
        raise ValueError("Image and weights must have the same height and width.")

    r = image[:, :, 2]
    g = image[:, :, 1]
    b = image[:, :, 0]

    rg_hist = np.zeros((bins, bins), dtype=float)
    gb_hist = np.zeros((bins, bins), dtype=float)
    br_hist = np.zeros((bins, bins), dtype=float)

  # Calculate bin indices for each pixel
    r_bins = np.floor(r / 4).astype(int)
    g_bins = np.floor(g / 4).astype(int)
    b_bins = np.floor(b / 4).astype(int)

  # Accumulate weighted counts into the histograms
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rg_hist[r_bins[i, j], g_bins[i, j]] += weights[i, j]
            gb_hist[g_bins[i, j], b_bins[i, j]] += weights[i, j]
            br_hist[b_bins[i, j], r_bins[i, j]] += weights[i, j]

    return rg_hist, gb_hist, br_hist

def calculate_weighted_2d_histograms_optimized(image, weights, bins=64):
    """Calculates weighted 2D histograms for RG, GB, and BR color pairs (optimized).

    Args:
        image: The input color image (NumPy array, BGR format).
        weights: A NumPy array of the same shape as the image, 
                 representing the weights for each pixel.

    Returns:
        A tuple containing the weighted 2D histograms for RG, GB, and BR.
    """

    if image.shape[:2] != weights.shape:  # Check only the 2D shape
        raise ValueError("Image and weights must have the same height and width.")

    r = image[:, :, 2]
    g = image[:, :, 1]
    b = image[:, :, 0]
    # print(np.max(r))

    # Calculate bin indices for each channel using vectorized operations
    r_bins = np.floor(r / (256/bins)).astype(int)
    g_bins = np.floor(g / (256/bins)).astype(int)
    b_bins = np.floor(b / (256/bins)).astype(int)

    # Initialize the histograms
    rg_hist = np.zeros((bins, bins), dtype=float)
    gb_hist = np.zeros((bins, bins), dtype=float)
    br_hist = np.zeros((bins, bins), dtype=float)
    hist_3d = np.zeros((bins, bins, bins), dtype=float)

    np.add.at(rg_hist, (r_bins, g_bins), weights)
    np.add.at(gb_hist, (g_bins, b_bins), weights)
    np.add.at(br_hist, (b_bins, r_bins), weights)
    np.add.at(hist_3d, (r_bins, g_bins, b_bins), weights)
    weighted_hist = calculate_weighted_2d_histograms_projected(image,weights,bins)

    return rg_hist, gb_hist, br_hist, weighted_hist

def calculate_weighted_2d_histograms_projected(image, weights, bins=64):
    """Calculates weighted 2D histograms for RG, GB, and BR color pairs (optimized).

    Args:
        image: The input color image (NumPy array, BGR format).
        weights: A NumPy array of the same shape as the image, 
                 representing the weights for each pixel.

    Returns:
        A tuple containing the weighted 2D histograms for RG, GB, and BR.
    """
    plane = np.array([1, 1, 1])
    origin = np.array([0, 0, 0])
    if image.shape[:2] != weights.shape:  # Check only the 2D shape
        raise ValueError("Image and weights must have the same height and width.")
    
    hist_ub = get_hist_upper_bound(np.max(image))
    
    u_axis, v_axis = build_planar_axes(plane, origin, hist_ub)
    hist_lb =0

    # Get the points that bound the histogram
    u_min_projected, u_max_projected = get_min_max_projected_boundaries(u_axis, hist_lb, hist_ub)
    v_min_projected, v_max_projected = get_min_max_projected_boundaries(v_axis, hist_lb, hist_ub)

    # Project the image into the 111 space
    u_indexes = np.dot(image, u_axis)
    v_indexes = np.dot(image, v_axis)
    projected_image = np.stack((u_indexes, v_indexes), axis=2).astype(np.float32)
    # print(projected_image.shape)

    g = projected_image[:, :, 1]
    b = projected_image[:, :, 0]
    # print(np.max(g),np.max(b))

    # Calculate bin indices for each channel using vectorized operations
    u_range = u_max_projected - u_min_projected
    v_range = v_max_projected - v_min_projected
    
    # Map values from their range to bin indices (0 to bins-1)
    u_bins = np.floor((u_indexes - u_min_projected) / u_range * bins).astype(int)
    v_bins = np.floor((v_indexes - v_min_projected) / v_range * bins).astype(int)
    weighted_hist = np.zeros((bins, bins), dtype=float)
    np.add.at(weighted_hist, (u_bins, v_bins), weights)

    return  weighted_hist

def plotting(rg_hist, gb_hist, br_hist,filename,image,weights=None):
    plt.figure(figsize=(20, 10))

    plt.subplot(231)
    plt.imshow(rg_hist.T, origin='lower', extent=[0, 128, 0, 128], cmap='jet')
    plt.xlabel("R")
    plt.ylabel("G")
    plt.title("RG Histogram")

    plt.subplot(232)
    plt.imshow(gb_hist.T, origin='lower', extent=[0, 128, 0, 128], cmap='jet')
    plt.xlabel("G")
    plt.ylabel("B")
    plt.title("GB Histogram")

    plt.subplot(233)
    plt.imshow(br_hist.T, origin='lower', extent=[0, 128, 0, 128], cmap='jet')
    plt.xlabel("B")
    plt.ylabel("R")
    plt.title("BR Histogram")

    r = image[:, :, 2].flatten()
    g = image[:, :, 1].flatten()
    b = image[:, :, 0].flatten()

    plt.subplot(212)
    if weights is None:
        plt.hist(r, bins=256, color='red', alpha=0.7, label='Red')
        plt.hist(g, bins=256, color='green', alpha=0.7, label='Green')
        plt.hist(b, bins=256, color='blue', alpha=0.7, label='Blue')
        plt.title('Color Channel Histograms')
        plt.legend()
    else:
        weights_f =weights.flatten()
        plt.hist(r, bins=256, color='red', alpha=0.7, label='Red',weights=weights_f)
        plt.hist(g, bins=256, color='green', alpha=0.7, label='Green',weights=weights_f)
        plt.hist(b, bins=256, color='blue', alpha=0.7, label='Blue',weights=weights_f)
        plt.title('Color Channel Histograms')
        plt.legend()

    plt.tight_layout()
    plt.show()
    plt.savefig(filename)

def plot_depth_value_distribution(depth, filename):
    """Plots the distribution of depth values."""
    plt.figure(figsize=(10, 6))
    plt.hist(depth.flatten(), bins=256, color='skyblue', edgecolor='black')
    plt.xlabel("Depth Value")
    plt.ylabel("Number of Pixels")
    plt.title("Depth Value Distribution")
    plt.savefig(filename)
    plt.close()

def remove_far_objects(image, depth, threshold):
    """Removes portions of the image that are too far based on depth."""
    mask = depth < threshold
    masked_image = image.copy()
    masked_image[mask] = 0  
    return masked_image, mask

import numpy as np
import matplotlib.pyplot as plt

def plot_bin_index_vs_value(hist, hist_name, filename, include_transformations=True):
    """
    Plots bin index vs. value for a 2D histogram with optional value transformations.
    
    Args:
        hist: 2D histogram array
        hist_name: Name of the histogram for the plot title
        filename: Base filename for saving the plots
        include_transformations: Whether to include tanh and log transformations
    """
    # Create the base figure for original values
    plt.figure(figsize=(10, 8))
    rows, cols = hist.shape
    bin_indices = []
    bin_values = []
    
    for i in range(rows):
        for j in range(cols):
            bin_index = i * cols + j  # Flatten 2D index to 1D
            bin_indices.append(bin_index)
            bin_values.append(hist[i, j])
    
    # Plot original values
    plt.plot(bin_indices, bin_values)
    plt.xlabel("Bin Index (Flattened)")
    plt.ylabel("Bin Value")
    plt.title(f"{hist_name} - Bin Index vs. Value (Original)")
    plt.savefig(filename)
    plt.close()
    print(f"Sum of original bin values: {np.sum(bin_values)}")
    
    if include_transformations:
        # Plot with hyperbolic tangent transformation
        plt.figure(figsize=(10, 8))
        tanh_values = np.tanh(bin_values)
        max_val = np.max(bin_values) # Avoid division by zero
        
    # Apply scaled tanh transformation
        normalized = bin_values / (max_val/6)  # Scale values for tanh input
        transformed = np.tanh(normalized)
    
    # Rescale back to original range
        tanh_values = transformed 
        plt.plot(bin_indices, tanh_values)
        plt.xlabel("Bin Index (Flattened)")
        plt.ylabel("Bin Value (tanh transformed)")
        plt.title(f"{hist_name} - Bin Index vs. Value (tanh transform)")
        plt.savefig(f"{filename.rsplit('.', 1)[0]}_tanh.{filename.rsplit('.', 1)[1]}")
        plt.close()
        print(f"Sum of tanh transformed bin values: {np.sum(tanh_values)}")
        
        # Plot with log transformation (adding small constant to avoid log(0))
        plt.figure(figsize=(10, 8))
        epsilon = 1e-10  # Small constant to avoid log(0)
        log_values = np.log1p(bin_values)  # log(1+x) to handle zeros gracefully
        plt.plot(bin_indices, log_values)
        plt.xlabel("Bin Index (Flattened)")
        plt.ylabel("Bin Value (log transformed)")
        plt.title(f"{hist_name} - Bin Index vs. Value (log transform)")
        plt.savefig(f"{filename.rsplit('.', 1)[0]}_log.{filename.rsplit('.', 1)[1]}")
        plt.close()
        print(f"Sum of log transformed bin values: {np.sum(log_values)}")

def tan_hyper(bin_values):
    max_val = np.max(bin_values) # Avoid division by zero
        
    # Apply scaled tanh transformation
    normalized = bin_values / (max_val/6)  # Scale values for tanh input
    transformed = np.tanh(normalized)
    
    # Rescale back to original range
    return transformed 
def log_tran(bin_values):
    log_values = np.log1p(bin_values)
    return log_values

if __name__ == "__main__":
    model = load_depth_anything_model()
    # /work/SuperResolutionData/spectralRatio/data/images_for_training/folder_5/huang_xingrui_011.tif
    image = cv2.imread("/home/balamurugan.d/src/hist2d/tif2.png")
    cv2.imwrite("tif2.png",image)
    # image = linear_to_log(image)
    # cv2.imwrite("log.png",image)
    depth = infer_depth(model,image)
    # depth = cv2.imread("/home/balamurugan.d/src/hist2d/dep.png")
    depth = cv2.cvtColor(depth,cv2.COLOR_BGR2GRAY)
    depth = depth.astype(np.float32)
    plot_depth_value_distribution(depth, "depth_distribution.png")
    threshold = 10.0 
    masked_image, mask = remove_far_objects(image, depth, threshold)

    print(image.shape)
    if image is None:
        print("Error: Could not load image.")
        exit()

    height, width, _ = image.shape
    color_normals= cv2.imread("/home/balamurugan.d/src/hist2d/up2.png")
    up = np.zeros((height, width), dtype=np.uint8)
    front = np.zeros((height, width), dtype=np.uint8)
    side = np.zeros((height, width), dtype=np.uint8)
    up = (color_normals[...,1])
    front = (color_normals[...,2])
    side= (color_normals[...,0])

    masked_up = up.copy()
    masked_up[mask] =0
    # cv2.imwrite("maskedup2.png",masked_up)
    up=up/255
    masked_up=masked_up/255

    rg_hist, gb_hist, br_hist = calculate_2d_histograms(image)
    wup_rg_hist, wup_gb_hist, wup_br_hist, hist_3d = calculate_weighted_2d_histograms_optimized(image,up,128)
    wup_rg_hist1, wup_gb_hist1, wup_br_hist1,_ = calculate_weighted_2d_histograms_optimized(image,masked_up,128)
    wup_rg_hist_proj = calculate_weighted_2d_histograms_projected(image,masked_up,128)
    proj_hist = get_plane_projected_histogram(image, buckets=128)


    wup_rg_hist1_tan = tan_hyper(wup_rg_hist1)
    wup_gb_hist1_tan = tan_hyper(wup_gb_hist1)
    wup_br_hist1_tan = tan_hyper(wup_br_hist1)

    wup_rg_hist1_log = log_tran(wup_rg_hist1)
    wup_gb_hist1_log = log_tran(wup_gb_hist1)
    wup_br_hist1_log = log_tran(wup_br_hist1)

    # plotting(rg_hist, gb_hist, br_hist,"2Dhist.png",image)
    # # plotting(w_rg_hist, w_gb_hist, w_br_hist,"weighted2Dhist.png",weights)

    # plotting(wup_rg_hist, wup_gb_hist, wup_br_hist,"l_weightedUpHist.png",image, up)
    # plotting(wup_rg_hist1, wup_gb_hist1, wup_br_hist1,"l_weightedUpHistMasked.png",image, masked_up)
    # plotting(wup_rg_hist_proj, wup_gb_hist1, wup_br_hist1,"l_weightedUpHist_weightplane.png",image, masked_up)
    # plotting(proj_hist, wup_gb_hist1, wup_br_hist1,"l_weightedUpHist_plane.png",image, masked_up)

    # # plotting(wup_rg_hist1_tan, wup_gb_hist1_tan, wup_br_hist1_tan,"l_weightedUpHistMasked_tan.png",image, masked_up)
    # # plotting(wup_rg_hist1_log, wup_gb_hist1_log, wup_br_hist1_log,"l_weightedUpHistMasked_log.png",image, masked_up)
    # plot_bin_index_vs_value(rg_hist, "Weighted Up RG Histogram", "rg_bin_index_vs_value.png")

    # plot_bin_index_vs_value(wup_rg_hist, "Weighted Up RG Histogram", "wup_rg_bin_index_vs_value.png")
    # plot_bin_index_vs_value(wup_rg_hist1, "Weighted Masked Up RG Histogram", "wup_rg_bin_index_vs_valuemasked.png")
    plot_bin_index_vs_value(wup_rg_hist1_tan, "Weighted Masked Up RG Histogram", "wup_rg_bin_index_vs_valuemasked_tan.png")

    # # plotting(wfront_rg_hist, wfront_gb_hist, wfront_br_hist,"weightedFrontHist.png",image, front)
    # # plotting(wside_rg_hist, wside_gb_hist, wside_br_hist,"weightedSideHist.png",image, side)

