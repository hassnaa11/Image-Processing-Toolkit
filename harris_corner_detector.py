from Image import Image
from image_processor import edge_detection
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import maximum_filter


sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
prewitt_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])


def apply_harris_changes(K, gradient_method, block_size, img_arr_in_gray: Image):

    Gx, Gy = apply_gradient(gradient_method, img_arr_in_gray.image)

    Ixx = Gx ** 2
    Iyy = Gy ** 2
    Ixy = Gx * Gy
    
    
    if Ixx.dtype != np.float32:
        Ixx = Ixx.astype(np.float32)
    if Iyy.dtype != np.float32:
        Iyy = Iyy.astype(np.float32)
    if Ixy.dtype != np.float32:
        Ixy = Ixy.astype(np.float32)

    Ixx = cv2.GaussianBlur(Ixx, (block_size, block_size), 0)
    Iyy = cv2.GaussianBlur(Iyy, (block_size, block_size), 0)
    Ixy = cv2.GaussianBlur(Ixy, (block_size, block_size), 0)

    det_M = (Ixx * Iyy) - (Ixy ** 2)
    trace_M = Ixx + Iyy
    harris_response = det_M - K * (trace_M ** 2)

    # Harris response Normalization
    harris_response = (harris_response - harris_response.min()) / (harris_response.max() - harris_response.min())

    harris_response = non_maximum_suppression(harris_response)

    threshold = choose_threshold_from_histogram(harris_response)
    binary_img_arr = (harris_response > threshold).astype(np.uint8)

    overlay_img_arr = display_corners(img_arr_in_gray.image, binary_img_arr)

    return binary_img_arr, overlay_img_arr


def non_maximum_suppression(harris_response, window_size=3):
    # Apply a maximum filter to find local maxima
    local_max = maximum_filter(harris_response, size=window_size)
    suppressed_response = np.where(harris_response == local_max, harris_response, 0)
    return suppressed_response


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
    # Flatten the Harris response and compute the histogram
    flat_response = harris_response.flatten()
    hist, bins = np.histogram(flat_response, bins=256, range=(flat_response.min(), flat_response.max()))

    # Find peaks in the histogram
    peaks, _ = find_peaks(hist)
    
    # Find valleys (local minima) between peaks
    valleys = []
    for i in range(len(peaks) - 1):
        start, end = peaks[i], peaks[i + 1]
        valley_index = np.argmin(hist[start:end]) + start
        valleys.append(valley_index)

    # Choose the first valley as the threshold (or customize this logic)
    if valleys:
        average_valley_index = int(np.mean(valleys))  # Compute the average valley index
        threshold = bins[average_valley_index]
    else:
        # Fallback: Use a percentage of the maximum response
        threshold = 0.01 * flat_response.max()

    print(f"Selected threshold from valley: {threshold}")
    return threshold


def display_corners(og_image, corners):
    # Overlay corners on the original image
    overlay_image_arr = np.stack((og_image,) * 3, axis=-1)  # Convert grayscale to RGB
    overlay_image_arr[corners > 0] = [255, 0, 0]  # Mark corners in red

    return overlay_image_arr
    
        