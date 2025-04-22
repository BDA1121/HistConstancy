import cv2
import numpy as np
from hist.Depth_Anything_V2.pipeline import load_depth_anything_model, infer_depth 



def compute_surface_normals(depth):
    # Apply bilateral filter to smooth depth

    for _ in range(5):
        depth = cv2.bilateralFilter(depth, d=7, sigmaColor=75, sigmaSpace=70)
    # depth = cv2.bilateralFilter(depth, d=9, sigmaColor=75, sigmaSpace=70)
    
    # Compute Sobel gradients (tangent vectors)
    sobel_x = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=5) / 4  
    sobel_y = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=5) / 4  
    # Create normal map
    # [1,0,dx]x[0,1,dy] = [-dx,-dy,1]
    normal = np.zeros((depth.shape[0], depth.shape[1], 3), dtype=np.float32)
    normal[..., 0] = -sobel_x  # Nx
    normal[..., 1] = sobel_y # Ny
    normal[..., 2] = 1        # Nz 
    norm = np.sqrt(sobel_x**2 + sobel_y**2 + normal[..., 2]**2)
    normal[..., 0] = normal[..., 0]/norm
    normal[..., 1] = normal[..., 1]/norm
    normal[..., 2] = normal[..., 2]/norm


    return normal

def colorize_normals(normal):
    height, width, _ = normal.shape
    color_normals = np.zeros((height, width, 3), dtype=np.uint8)
    color_normals= normal
    up = np.zeros((height, width), dtype=np.uint8)
    front = np.zeros((height, width), dtype=np.uint8)
    side = np.zeros((height, width), dtype=np.uint8)
    up = (normal[...,1])
    front = (normal[...,2])
    side= (normal[...,0])

    return (color_normals+1)/2,(up+1)/2,(front+1)/2,(side+1)/2

def create_normal_weights(normals):
    """
    Creates weights based on surface normals, prioritizing up, then front, then sideways.

    Args:
        normals (numpy.ndarray): A NumPy array representing the surface normals.
                                 Shape: (height, width, 3)
                                 Channel 0: Sideways normal component (x)
                                 Channel 1: Upward normal component (y)
                                 Channel 2: Forward normal component (z)

    Returns:
        numpy.ndarray: A NumPy array of weights with the same shape as the image (height, width).
    """

    height, width, _ = normals.shape

    # normals = cv2.cvtColor(normals,cv2.COLOR_gr2BGR)
    # Extract normal components
    sideways = normals[:, :, 0]
    up = normals[:, :, 1]
    front = normals[:, :, 2]

    # Create weights based on normal components
    # Prioritize up, then front, then sideways.
    abs_sideways = np.abs(sideways)
    abs_up = np.abs(up)
    abs_front = np.abs(front)

    weights = np.ones((height, width)) * 1.0  # Initialize weights to 1 (sideways default)

    # Apply weights based on dominant component
    weights[abs_up > abs_sideways] = 2.0
    weights[abs_up > abs_front] = 2.0

    weights[np.logical_and(abs_front > abs_sideways, abs_front > abs_up)] = 0.5
    # cv2.imwrite("weights.png",((weights/2)*255).astype(np.uint8))
    
    return weights

# normal = cv2.imread("/home/balamurugan.d/src/colored_normals.png")
# Compute normals and colorize them

# models = load_depth_anything_model()
if __name__ == "__main__":
    model = load_depth_anything_model()
    image = cv2.imread("/work/SuperResolutionData/spectralRatio/data/images_for_training/folder_1/dhoka_jain_chirag_000.tif", cv2.IMREAD_UNCHANGED)
    depth = infer_depth(model,image)
    print("--------------------")

    depth = cv2.cvtColor(depth,cv2.COLOR_BGR2GRAY)
    print("--------------------")

    # cv2.imwrite("dep.png",depth)
    # depth = cv2.imread("/home/balamurugan.d/src/hist/Depth-Anything-V2/vis_depth/tif.png",cv2.IMREAD_GRAYSCALE)


    depth = depth.astype(np.float32)
    normal_map = compute_surface_normals(depth)
    colored_normals,up,front,side = colorize_normals(normal_map)
    # weights = create_normal_weights(colored_normals)
    print("--------------------")

    # Save the result
    # colored_normals,up,front,side = cv2.cvtColor(colored_normals,cv2.COLOR_BGR2RGB)
    cv2.imwrite("colored_normals1.png", (colored_normals*255).astype(np.uint8))
    cv2.imwrite("up.png", (up*255).astype(np.uint8))
    cv2.imwrite("front.png", (front*255).astype(np.uint8))
    cv2.imwrite("side_side.png", (side*255).astype(np.uint8))
    print("--------------------")

