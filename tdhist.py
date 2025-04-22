import cv2
import numpy as np
import sys

MAX_8BIT_INT = 2**8 - 1
MAX_16BIT_INT = 2**16 - 1

def get_hist_upper_bound(max_image_value):
    """
    Uses the size of the maximum image value to determine how big the upper bound of the histogram should be,
    implicitly determining if stretching is needed.
    8-bit maximum unsigned integer if 1 < value <= 8-bit, 16-bit maximum unsigned integer if 8-bit < value < 16-bit max
    or if the image value is 1, because images stored as floats between [0, 1] should use 16 bits
    :param max_image_value: The maximum value possible for the image
    :return: The upper bound of the histogram range
    """
    # Upper bound has 1 added because OpenCV calcHist uses exclusive upper bound
    if 1 < max_image_value <= MAX_8BIT_INT:
        return MAX_8BIT_INT + 1
    elif MAX_8BIT_INT < max_image_value <= MAX_16BIT_INT or max_image_value == 1:
        return MAX_16BIT_INT + 1
    else:
        print("ERROR: Invalid maximum image value! Should be between 1 and {}".format(MAX_16BIT_INT))
        sys.exit(-1)


def scale_to_upper_bound(image, max_image_value, hist_ub):
    """
    Normalizes image to [0, 1] via max value, then scales to the proper depth
    :param image:           The image to normalize and scale
    :param max_image_value: The maximum value to normalize the image with
    :param hist_ub:         Histogram upper bound to scale to (EXCLUSIVE)
    :return: The image scaled to 16 bits
    """
    image_upper_bound = hist_ub - 1  # upper bound will be exclusive, and therefore one bigger than actual image
    normalized_image = np.clip(image.astype(np.float32) / max_image_value, 0, 1)
    scaled_image = normalized_image * image_upper_bound
    return scaled_image

def build_planar_axes(plane, origin, upper_bound):
    """
    Builds the planar axis from a plane, an origin point, and the maximum value of the image to project
    :param plane:       The plane to project onto
    :param origin:      The origin of the plane
    :param upper_bound: The upper bound of the image to project onto the plane
    :return: The u and v axes of the projected plane
    """
    normal_plane = plane / np.linalg.norm(plane)
    green_axis_point = np.array([0, np.log(upper_bound), 0])

    v_axis_unnormalized = project_3d_point(green_axis_point, origin, normal_plane) - origin
    v_axis = v_axis_unnormalized / np.linalg.norm(v_axis_unnormalized)

    u_axis = np.cross(v_axis, normal_plane)

    return u_axis, v_axis

def get_min_max_projected_boundaries(axis, lower_bound, upper_bound):
    """
    Calculates the minimum and maximum for the axis components on the project surface
    :param axis:        The axis to calculate the bounds for
    :param lower_bound: Lower bound of image to project
    :param upper_bound: Upper bound of image to project
    :return: The minimum and maximum values along the axis
    """
    # Get the cube boundaries
    cube_boundaries = get_cube_boundaries(lower_bound, upper_bound)

    # Get the dot product of all cube boundaries with the axis
    dot_products = []
    for boundary_point in cube_boundaries:
        dot_products.append(np.dot(boundary_point, axis))
    return np.min(dot_products), np.max(dot_products)

def get_cube_boundaries(lower_bound, upper_bound):
    """
    Gets the boundary positions for the projection onto a plane
    :param lower_bound: The lower bound of values in the image to project
    :param upper_bound: The upper bound of values in the image to project
    :return:
    """
    # Define all possible points involving 0 and 1, and return as numpy array
    boundaries = []
    values = [lower_bound, upper_bound]
    for x in values:
        for y in values:
            for z in values:
                boundaries.append((x, y, z))
    return np.array(boundaries)

def get_plane_projected_histogram(image, buckets):
    """
    Gets the histogram of an image projected onto a plane
    :param image:   The image to get the projected histogram of
    :param plane:   The plane to project the histogram onto
    :param origin:  The origin of the plane
    :param hist_lb: The lower bound of the values within the image
    :param hist_ub: The upper bound of the values within the image
    :param buckets: The number of buckets in the histogram
    :return: The projected histogram
    """
    plane = np.array([1, 1, 1])
    origin = np.array([0, 0, 0])
    hist_lb =0
    hist_ub = get_hist_upper_bound(np.max(image))
    # Get the u and v axes
    u_axis, v_axis = build_planar_axes(plane, origin, hist_ub)

    # Get the points that bound the histogram
    u_min_projected, u_max_projected = get_min_max_projected_boundaries(u_axis, hist_lb, hist_ub)
    v_min_projected, v_max_projected = get_min_max_projected_boundaries(v_axis, hist_lb, hist_ub)

    # Project the image into the 111 space
    u_indexes = np.dot(image, u_axis)
    v_indexes = np.dot(image, v_axis)
    projected_image = np.stack((u_indexes, v_indexes), axis=2).astype(np.float32)

    # Get the projected histogram
    projected_histogram = cv2.calcHist(images=[projected_image], channels=[0, 1], mask=None,
                                       histSize=[buckets, buckets],
                                       ranges=[u_min_projected, u_max_projected, v_min_projected, v_max_projected])

    return projected_histogram

def project_3d_point(point_3d, origin, normal_plane):
    """
    Projects a 3d point onto a plane
    :param point_3d:     The 3d point to project
    :param origin:       The origin of the plane to project onto
    :param normal_plane: The normal plane of the plane to project onto
    :return:
    """
    difference = origin - point_3d
    dot_product = np.dot(difference, normal_plane)
    normal_magnitude_square = np.dot(normal_plane, normal_plane)
    projected_point = point_3d - (dot_product / normal_magnitude_square) * normal_magnitude_square
    return projected_point

if __name__ == '__main__':
    # example_usage()
    print("")