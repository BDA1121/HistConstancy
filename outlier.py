# # import cv2
# # import numpy as np
# # import matplotlib.pyplot as plt
# # from mpl_toolkits.mplot3d import Axes3D
# # from scipy import stats
# # import time
# # from hist.Depth_Anything_V2.pipeline import load_depth_anything_model, infer_depth 
# # from colored_test import compute_surface_normals,colorize_normals,create_normal_weights

# # def linear_to_log(image):
# #     """Converts a linear image to a log-encoded image."""
# #     image_float = image.astype(np.float32) / 255.0  # Normalize to [0, 1]
# #     image_log = np.log1p(image_float)  # Apply log1p (log(1 + x))
# #     image_log_scaled = (image_log / np.log1p(1.0)) * 255.0  # Scale back to [0, 255]
# #     return image_log_scaled.astype(np.uint8)

# # def calculate_2d_histograms(image,):
# #     """Calculates 2D histograms for RG, GB, and BR color pairs.

# #     Args:
# #         image: The input color image (NumPy array, BGR format).

# #     Returns:
# #         A tuple containing the 2D histograms for RG, GB, and BR.
# #     """

# #     r = image[:, :, 2]  # Red channel
# #     g = image[:, :, 1]  # Green channel
# #     b = image[:, :, 0]  # Blue channel

# #     rg_hist = np.histogram2d(r.flatten(), g.flatten(), bins=64, range=[[0, 256], [0, 256]])[0]
# #     gb_hist = np.histogram2d(g.flatten(), b.flatten(), bins=64, range=[[0, 256], [0, 256]])[0]
# #     br_hist = np.histogram2d(b.flatten(), r.flatten(), bins=64, range=[[0, 256], [0, 256]])[0]

# #     return rg_hist, gb_hist, br_hist

# # def calculate_weighted_2d_histograms_optimized(image, weights, bins=64):
# #     """Calculates weighted 2D histograms for RG, GB, and BR color pairs (optimized).

# #     Args:
# #         image: The input color image (NumPy array, BGR format).
# #         weights: A NumPy array of the same shape as the image, 
# #                     representing the weights for each pixel.

# #     Returns:
# #         A tuple containing the weighted 2D histograms for RG, GB, and BR.
# #     """

# #     if image.shape[:2] != weights.shape:  # Check only the 2D shape
# #         raise ValueError("Image and weights must have the same height and width.")

# #     r = image[:, :, 2]
# #     g = image[:, :, 1]
# #     b = image[:, :, 0]

# #     # Calculate bin indices for each channel using vectorized operations
# #     r_bins = np.floor(r / (256/bins)).astype(int)
# #     g_bins = np.floor(g / (256/bins)).astype(int)
# #     b_bins = np.floor(b / (256/bins)).astype(int)

# #     # Initialize the histograms
# #     rg_hist = np.zeros((bins, bins), dtype=float)
# #     gb_hist = np.zeros((bins, bins), dtype=float)
# #     br_hist = np.zeros((bins, bins), dtype=float)

# #     # Use advanced indexing to accumulate weights into the histograms
# #     np.add.at(rg_hist, (r_bins, g_bins), weights)
# #     np.add.at(gb_hist, (g_bins, b_bins), weights)
# #     np.add.at(br_hist, (b_bins, r_bins), weights)

# #     return rg_hist, gb_hist, br_hist


# # def plot_histogram_details(hist, hist_name, filename):
# #     plt.figure(figsize=(12, 6))

# #     flat_hist = hist.flatten()
# #     bins = np.arange(len(flat_hist))

# #     plt.plot(bins, flat_hist)
# #     plt.xlabel("Bin Index")
# #     plt.ylabel("Bin Value")
# #     plt.title(f"{hist_name} Bin Values")

# #     counts_256 = np.sum(flat_hist > 256)
# #     counts_512 = np.sum(flat_hist > 512)
# #     counts_1024 = np.sum(flat_hist > 1024)

# #     plt.text(0.95, 0.95, f">256: {counts_256}\n>512: {counts_512}\n>1024: {counts_1024}",
# #              verticalalignment='top', horizontalalignment='right',
# #              transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8))

# #     plt.savefig(filename)
# #     plt.close()

# # if __name__ == "__main__":
# #     model = load_depth_anything_model()
# #     image = cv2.imread("/home/balamurugan.d/src/tests/tif2.png")
# #     image = linear_to_log(image)
# #     cv2.imwrite("log.png", image)
# #     depth = infer_depth(model, image)
# #     depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)

# #     depth = depth.astype(np.float32)
# #     cv2.imwrite("dep.png", depth)
# #     print(image.shape)
# #     if image is None:
# #         print("Error: Could not load image.")
# #         exit()

# #     normal_map = compute_surface_normals(depth)
# #     colored_normals = colorize_normals(normal_map)
# #     _, up, front, side = colored_normals

# #     rg_hist, gb_hist, br_hist = calculate_2d_histograms(image)
# #     wup_rg_hist, wup_gb_hist, wup_br_hist = calculate_weighted_2d_histograms_optimized(image, up, 64)
# #     wfront_rg_hist, wfront_gb_hist, wfront_br_hist = calculate_weighted_2d_histograms_optimized(image, front, 64)
# #     wside_rg_hist, wside_gb_hist, wside_br_hist = calculate_weighted_2d_histograms_optimized(image, side, 64)


# #     plot_histogram_details(rg_hist, "RG Histogram", "histdata/rg_hist_details.png")
# #     plot_histogram_details(gb_hist, "GB Histogram", "histdata/gb_hist_details.png")
# #     plot_histogram_details(br_hist, "BR Histogram", "histdata/br_hist_details.png")

# #     plot_histogram_details(wup_rg_hist, "Weighted Up RG Histogram", "histdata/wup_rg_hist_details.png")
# #     plot_histogram_details(wup_gb_hist, "Weighted Up GB Histogram", "histdata/wup_gb_hist_details.png")
# #     plot_histogram_details(wup_br_hist, "Weighted Up BR Histogram", "histdata/wup_br_hist_details.png")

# #     plot_histogram_details(wfront_rg_hist, "Weighted Front RG Histogram", "histdata/wfront_rg_hist_details.png")
# #     plot_histogram_details(wfront_gb_hist, "Weighted Front GB Histogram", "histdata/wfront_gb_hist_details.png")
# #     plot_histogram_details(wfront_br_hist, "Weighted Front BR Histogram", "histdata/wfront_br_hist_details.png")

# #     plot_histogram_details(wside_rg_hist, "Weighted Side RG Histogram", "histdata/wside_rg_hist_details.png")
# #     plot_histogram_details(wside_gb_hist, "Weighted Side GB Histogram", "histdata/wside_gb_hist_details.png")
# #     plot_histogram_details(wside_br_hist, "Weighted Side BR Histogram", "histdata/wside_br_hist_details.png")

# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from scipy import stats
# import time
# from hist.Depth_Anything_V2.pipeline import load_depth_anything_model, infer_depth 
# from colored_test import compute_surface_normals,colorize_normals,create_normal_weights

# def linear_to_log(image):
#     """Converts a linear image to a log-encoded image."""
#     image_float = image.astype(np.float32) / 255.0
#     image_log = np.log1p(image_float)
#     image_log_scaled = (image_log / np.log1p(1.0)) * 255.0
#     return image_log_scaled.astype(np.uint8)

# def calculate_2d_histograms(image):
#     """Calculates 2D histograms for RG, GB, and BR color pairs."""
#     r = image[:, :, 2]
#     g = image[:, :, 1]
#     b = image[:, :, 0]
#     rg_hist = np.histogram2d(r.flatten(), g.flatten(), bins=64, range=[[0, 256], [0, 256]])[0]
#     gb_hist = np.histogram2d(g.flatten(), b.flatten(), bins=64, range=[[0, 256], [0, 256]])[0]
#     br_hist = np.histogram2d(b.flatten(), r.flatten(), bins=64, range=[[0, 256], [0, 256]])[0]
#     return rg_hist, gb_hist, br_hist

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

# def plot_2d_histogram_scatter(hist, hist_name, filename):
#     """Plots a 2D histogram as a scatter plot."""
#     plt.figure(figsize=(10, 8))
#     rows, cols = hist.shape
#     x, y, z = [], [], []
#     for i in range(rows):
#         for j in range(cols):
#             if hist[i, j] > 0:  # Plot only non-zero bins
#                 x.append(j)
#                 y.append(i)
#                 z.append(hist[i, j])
#     plt.scatter(x, y, s=z, alpha=0.7)  # Size proportional to bin value
#     plt.xlabel(hist_name.split()[0][0])  # Extract first letter of x-axis label
#     plt.ylabel(hist_name.split()[0][1])  # Extract second letter of y-axis label
#     plt.title(f"{hist_name} Scatter Plot")
#     plt.colorbar(label="Bin Value")
#     plt.savefig(filename)
#     plt.close()

# if __name__ == "__main__":
#     model = load_depth_anything_model()
#     image = cv2.imread("/home/balamurugan.d/src/tests/tif2.png")
#     image = linear_to_log(image)
#     cv2.imwrite("log.png", image)
#     depth = infer_depth(model, image)
#     depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)
#     depth = depth.astype(np.float32)
#     cv2.imwrite("dep.png", depth)
#     print(image.shape)
#     if image is None:
#         print("Error: Could not load image.")
#         exit()
#     normal_map = compute_surface_normals(depth)
#     colored_normals = colorize_normals(normal_map)
#     _, up, front, side = colored_normals
#     rg_hist, gb_hist, br_hist = calculate_2d_histograms(image)
#     wup_rg_hist, wup_gb_hist, wup_br_hist = calculate_weighted_2d_histograms_optimized(image, up, 64)
#     wfront_rg_hist, wfront_gb_hist, wfront_br_hist = calculate_weighted_2d_histograms_optimized(image, front, 64)
#     wside_rg_hist, wside_gb_hist, wside_br_hist = calculate_weighted_2d_histograms_optimized(image, side, 64)

#     plot_2d_histogram_scatter(rg_hist, "RG Histogram", "histdata/rg_scatter.png")
#     plot_2d_histogram_scatter(gb_hist, "GB Histogram", "histdata/gb_scatter.png")
#     plot_2d_histogram_scatter(br_hist, "BR Histogram", "histdata/br_scatter.png")
#     plot_2d_histogram_scatter(wup_rg_hist, "Weighted Up RG Histogram", "histdata/wup_rg_scatter.png")
#     plot_2d_histogram_scatter(wup_gb_hist, "Weighted Up GB Histogram", "histdata/wup_gb_scatter.png")
#     plot_2d_histogram_scatter(wup_br_hist, "Weighted Up BR Histogram", "histdata/wup_br_scatter.png")
#     plot_2d_histogram_scatter(wfront_rg_hist, "Weighted Front RG Histogram", "histdata/wfront_rg_scatter.png")
#     plot_2d_histogram_scatter(wfront_gb_hist, "Weighted Front GB Histogram", "histdata/wfront_gb_scatter.png")
#     plot_2d_histogram_scatter(wfront_br_hist, "Weighted Front BR Histogram", "histdata/wfront_br_scatter.png")
#     plot_2d_histogram_scatter(wside_rg_hist, "Weighted Side RG Histogram", "histdata/wside_rg_scatter.png")
#     plot_2d_histogram_scatter(wside_gb_hist, "Weighted Side GB Histogram", "histdata/wside_gb_scatter.png")
#     plot_2d_histogram_scatter(wside_br_hist, "Weighted Side BR Histogram", "histdata/wside_br_scatter.png")

import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats
import time
from hist.Depth_Anything_V2.pipeline import load_depth_anything_model, infer_depth 
from colored_test import compute_surface_normals,colorize_normals,create_normal_weights

def linear_to_log(image):
    """Converts a linear image to a log-encoded image."""
    image_float = image.astype(np.float32) / 255.0
    image_log = np.log1p(image_float)
    image_log_scaled = (image_log / np.log1p(1.0)) * 255.0
    return image_log_scaled.astype(np.uint8)

def calculate_2d_histograms(image):
    """Calculates 2D histograms for RG, GB, and BR color pairs."""
    r = image[:, :, 2]
    g = image[:, :, 1]
    b = image[:, :, 0]
    rg_hist = np.histogram2d(r.flatten(), g.flatten(), bins=64, range=[[0, 256], [0, 256]])[0]
    gb_hist = np.histogram2d(g.flatten(), b.flatten(), bins=64, range=[[0, 256], [0, 256]])[0]
    br_hist = np.histogram2d(b.flatten(), r.flatten(), bins=64, range=[[0, 256], [0, 256]])[0]
    return rg_hist, gb_hist, br_hist

def calculate_weighted_2d_histograms_optimized(image, weights, bins=64):
    """Calculates weighted 2D histograms for RG, GB, and BR color pairs (optimized)."""
    if image.shape[:2] != weights.shape:
        raise ValueError("Image and weights must have the same height and width.")
    r = image[:, :, 2]
    g = image[:, :, 1]
    b = image[:, :, 0]
    r_bins = np.floor(r / (256/bins)).astype(int)
    g_bins = np.floor(g / (256/bins)).astype(int)
    b_bins = np.floor(b / (256/bins)).astype(int)
    rg_hist = np.zeros((bins, bins), dtype=float)
    gb_hist = np.zeros((bins, bins), dtype=float)
    br_hist = np.zeros((bins, bins), dtype=float)
    np.add.at(rg_hist, (r_bins, g_bins), weights)
    np.add.at(gb_hist, (g_bins, b_bins), weights)
    np.add.at(br_hist, (b_bins, r_bins), weights)
    return rg_hist, gb_hist, br_hist

def plot_bin_index_vs_value(hist, hist_name, filename):
    """Plots bin index vs. value for a 2D histogram."""
    plt.figure(figsize=(10, 8))
    rows, cols = hist.shape
    bin_indices = []
    bin_values = []
    for i in range(rows):
        for j in range(cols):
            bin_index = i * cols + j  # Flatten 2D index to 1D
            bin_indices.append(bin_index)
            bin_values.append(hist[i, j])
    plt.plot(bin_indices, bin_values)
    plt.xlabel("Bin Index (Flattened)")
    plt.ylabel("Bin Value")
    plt.title(f"{hist_name} - Bin Index vs. Value")
    plt.savefig(filename)
    plt.close()

if __name__ == "__main__":
    model = load_depth_anything_model()
    image = cv2.imread("/home/balamurugan.d/src/tests/tif2.png")
    image = linear_to_log(image)
    cv2.imwrite("log.png", image)
    depth = infer_depth(model, image)
    depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)
    depth = depth.astype(np.float32)
    cv2.imwrite("dep.png", depth)
    print(image.shape)
    if image is None:
        print("Error: Could not load image.")
        exit()
    normal_map = compute_surface_normals(depth)
    colored_normals = colorize_normals(normal_map)
    _, up, front, side = colored_normals
    rg_hist, gb_hist, br_hist = calculate_2d_histograms(image)
    wup_rg_hist, wup_gb_hist, wup_br_hist = calculate_weighted_2d_histograms_optimized(image, up, 64)
    wfront_rg_hist, wfront_gb_hist, wfront_br_hist = calculate_weighted_2d_histograms_optimized(image, front, 64)
    wside_rg_hist, wside_gb_hist, wside_br_hist = calculate_weighted_2d_histograms_optimized(image, side, 64)

    plot_bin_index_vs_value(rg_hist, "RG Histogram", "histdata/rg_bin_index_vs_value.png")
    plot_bin_index_vs_value(gb_hist, "GB Histogram", "histdata/gb_bin_index_vs_value.png")
    plot_bin_index_vs_value(br_hist, "BR Histogram", "histdata/br_bin_index_vs_value.png")
    plot_bin_index_vs_value(wup_rg_hist, "Weighted Up RG Histogram", "histdata/wup_rg_bin_index_vs_value.png")
    plot_bin_index_vs_value(wup_gb_hist, "Weighted Up GB Histogram", "histdata/wup_gb_bin_index_vs_value.png")
    plot_bin_index_vs_value(wup_br_hist, "Weighted Up BR Histogram", "histdata/wup_br_bin_index_vs_value.png")
    plot_bin_index_vs_value(wfront_rg_hist, "Weighted Front RG Histogram", "histdata/wfront_rg_bin_index_vs_value.png")
    plot_bin_index_vs_value(wfront_gb_hist, "Weighted Front GB Histogram", "histdata/wfront_gb_bin_index_vs_value.png")
    plot_bin_index_vs_value(wfront_br_hist, "Weighted Front BR Histogram", "histdata/wfront_br_bin_index_vs_value.png")
    plot_bin_index_vs_value(wside_rg_hist, "Weighted Side RG Histogram", "histdata/wside_rg_bin_index_vs_value.png")
    plot_bin_index_vs_value(wside_gb_hist, "Weighted Side GB Histogram", "histdata/wside_gb_bin_index_vs_value.png")
    plot_bin_index_vs_value(wside_br_hist, "Weighted Side BR Histogram", "histdata/wside_br_bin_index_vs_value.png")