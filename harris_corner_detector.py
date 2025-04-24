from Image import Image
from image_processor import edge_detection
import numpy as np
import cv2


sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
prewitt_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])


def apply_harris_changes(K, gradient_method, block_size, img_arr_in_gray: Image):
    """    
        return binary image array and overlay image array
    """
    Gx, Gy = apply_gradient(gradient_method, img_arr_in_gray.image)

    Ixx = Gx ** 2
    Iyy = Gy ** 2
    Ixy = Gx * Gy

    # Ixx = cv2.GaussianBlur(Ixx, (block_size, block_size), 0)
    # Iyy = cv2.GaussianBlur(Iyy, (block_size, block_size), 0)
    # Ixy = cv2.GaussianBlur(Ixy, (block_size, block_size), 0)

    # Harris response matrix
    det_M = (Ixx * Iyy) - (Ixy ** 2)
    trace_M = Ixx + Iyy
    harris_response = det_M - K * (trace_M ** 2)

    # Threshold the Harris response
    threshold = choose_threshold_from_histogram(harris_response)
    binary_img_arr = (harris_response > threshold).astype(np.uint8)

    # Step 6: Visualize the corners
    overlay_img_arr = display_corners(img_arr_in_gray.image, binary_img_arr)

    return binary_img_arr, overlay_img_arr


def apply_gradient(method, img_arr):
    """ 
    Returns->\n  
    -->Gx (ndarray): Gradient in the x-direction.\n
    -->Gy (ndarray): Gradient in the y-direction.
    """
    processor = edge_detection(img_arr)
    if method == "Sobel":
        Gx = processor.apply_kernel(sobel_x, img_arr)
        Gy = processor.apply_kernel(sobel_y, img_arr)
    elif method == "Prewitt":
        Gx = processor.apply_kernel(prewitt_x, img_arr)
        Gy = processor.apply_kernel(prewitt_y, img_arr)
    else:
        raise ValueError("Invalid gradient method. Choose 'Sobel' or 'Prewitt'.")
    return Gx, Gy


def choose_threshold_from_histogram(harris_response):
    """
    Choose a threshold based on the histogram of the Harris response.
    
    Args:
        harris_response (ndarray): Harris response matrix.
    
    Returns:
        threshold (float): Selected threshold value.
    """
    # Flatten the Harris response and compute the histogram
    flat_response = harris_response.flatten()
    hist, bins = np.histogram(flat_response, bins=256, range=(flat_response.min(), flat_response.max()))
    threshold = 0.1 * flat_response.max()
    print(f"Selected threshold: {threshold}")
    return threshold


def display_corners(og_image, corners):
    """
    Display the corners on the original image.
    
    Args:
        image (ndarray): Original grayscale image.
        corners (ndarray): Binary image with corners marked as 1.
    """
    # Overlay corners on the original image
    overlay_image_arr = np.stack((og_image,) * 3, axis=-1)  # Convert grayscale to RGB
    overlay_image_arr[corners > 0] = [255, 0, 0]  # Mark corners in red

    return overlay_image_arr
    
        