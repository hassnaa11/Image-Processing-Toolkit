from Image import Image
from image_processor import edge_detection
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
from skimage.measure import EllipseModel
from scipy.ndimage import label

def line_hough_transform(canny_filtered_img: np.ndarray, theta_step=15):
    """
    Implements Hough transform for line detection

    Args:
        canny_filtered_img: Binary edge image (output from Canny edge detector)
        theta_step: Angular step size in degrees

    Returns:
        accumulator: Hough accumulator array
        rhos: Array of rho values
        thetas: Array of theta values
    """
    height, width = canny_filtered_img.shape

    # Maximum distance possible in the image is the diagonal
    diag_len = int(np.sqrt(height ** 2 + width ** 2))

    # Range of rho values: -diag_len to +diag_len
    rho_range = 2 * diag_len

    # Initialize parameters
    thetas = np.deg2rad(np.arange(-90, 90, theta_step))  # Theta values in radians
    rhos = np.arange(-diag_len, diag_len+1)  # Rho values

    # Initialize accumulator array
    accumulator = np.zeros((len(rhos), len(thetas)), dtype=np.uint64)

    # Find edge points
    edge_points = np.argwhere(canny_filtered_img == 255)

    # Calculate sin and cos values once for all thetas
    cos_thetas = np.cos(thetas)
    sin_thetas = np.sin(thetas)

    # For each edge point
    for y, x in edge_points:
        # For each theta value
        for theta_idx, (cos_t, sin_t) in enumerate(zip(cos_thetas, sin_thetas)):
            # Calculate rho = x*cos(theta) + y*sin(theta)
            rho = int(x * cos_t + y * sin_t)

            # Shift rho to ensure it's positive (for array indexing)
            rho_idx = rho + diag_len

            # Increment accumulator
            if 0 <= rho_idx < len(rhos):
                accumulator[rho_idx, theta_idx] += 1

    return accumulator, rhos, thetas, cos_thetas, sin_thetas


def detect_lines(canny_filtered_img_arr: np.ndarray, threshold_ratio=0.5, step_sz=20):
    """
    Detects lines from Hough accumulator array

    Args:
        canny_filtered_img_arr: Binary edge image (output from Canny edge detector)
        threshold_ratio: Ratio of max accumulator value to use as threshold
        min_line_length: Minimum line length to consider

    Returns:
        lines: List of detected lines in (rho, theta) format
    """
    # Apply Hough transform
    accumulator, rhos, thetas, cosines, sines = line_hough_transform(canny_filtered_img_arr, step_sz)

    # Calculate threshold
    threshold = threshold_ratio * np.max(accumulator)

    # Find peaks in accumulator
    lines = []
    height, width = canny_filtered_img_arr.shape
    min_line_length = 5

    for rho_idx in range(len(rhos)):
        for theta_idx in range(len(thetas)):
            if accumulator[rho_idx, theta_idx] > threshold:
                rho = rhos[rho_idx]
                theta = thetas[theta_idx]

                # Convert (rho, theta) to line endpoints for visualization
                if sines[theta_idx] != 0:  # Non-vertical line
                    x1, y1 = 0, int(round(rho / sines[theta_idx]))
                    x2, y2 = width - 1, int((rho - (width - 1) * cosines[theta_idx]) / sines[theta_idx])
                else:  # Vertical line
                    x1, y1 = int(rho), 0
                    x2, y2 = int(rho), height - 1

                # Calculate line length
                line_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

                # Add to list if line is long enough
                if line_length >= min_line_length:
                    lines.append((rho, theta, (x1, y1, x2, y2)))

    return lines


def draw_lines(original_img_arr, canny_filtered_img_arr, threshold_ratio, step_sz):
    """
    Draw detected lines on the image

    Args:
        image: Input image
        lines: List of lines in (rho, theta, (x1, y1, x2, y2)) format

    Returns:
        image_with_line_arr: Image with lines drawn
    """
    lines = detect_lines(canny_filtered_img_arr, threshold_ratio, step_sz)
    
    # Create a copy of the input image
    image_with_line_arr = original_img_arr.copy()

    # If the image is grayscale, convert to RGB
    if len(image_with_line_arr.shape) == 2:
        image_with_line_arr = cv2.cvtColor(image_with_line_arr, cv2.COLOR_GRAY2RGB)

    # Draw each line
    for rho, theta, (x1, y1, x2, y2) in lines:
        # Draw the line
        cv2.line(image_with_line_arr, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return image_with_line_arr


def circle_hough_transform(canny_filtered_img: np.ndarray, step_sz=20):
    height, width = canny_filtered_img.shape
    small_dim = min(height, width)
    max_r = small_dim // 2
    min_r = 5
    radius_values = np.arange(min_r, max_r + 1)
    accumulator = np.zeros((height, width, len(radius_values)), dtype=np.uint64)
    edge_points = np.argwhere(canny_filtered_img == 255)
    
    thetas = np.deg2rad(np.arange(-90, 90, step_sz))
    cos_thetas, sin_thetas = np.cos(thetas), np.sin(thetas)
    
    max_accum = 0
    for idx, r in enumerate(radius_values):
        for x, y in edge_points:
            for cos_theta, sin_theta in zip(cos_thetas, sin_thetas):
                a, b = int(x - r * cos_theta), int(y - r * sin_theta)
                if 0 <= a < height and 0 <= b < width:
                    accumulator[a, b, idx] += 1
                    max_accum = max(max_accum, accumulator[a, b, idx])
    return accumulator, radius_values, max_accum

def detect_circles(canny_filtered_img_arr: np.ndarray, threshold_ratio=0.7, step_sz=20):
    accumulator, radius_values, max_accum = circle_hough_transform(canny_filtered_img_arr, step_sz)
    threshold = threshold_ratio * max_accum
    circles = [(b, a, radius_values[r_idx])
               for r_idx in range(len(radius_values))
               for a, b in np.argwhere(accumulator[:, :, r_idx] > threshold)]
    return circles

def draw_circles_on_image(original_img_arr, canny_filtered_img_arr, threshold_ratio, step_sz):
    circles = detect_circles(canny_filtered_img_arr, threshold_ratio, step_sz)
    image_with_circles_arr = np.copy(original_img_arr)
    for x, y, r in circles:
        cv2.circle(image_with_circles_arr, (int(x), int(y)), int(r), (0, 0, 255), 2)
    return image_with_circles_arr

# def detect_ellipses(canny_filtered_img_arr: np.ndarray, min_ellipse_size=10):
#     contours, _ = cv2.findContours(canny_filtered_img_arr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     ellipses = []
#     for contour in contours:
#         if len(contour) >= 5:
#             ellipse = cv2.fitEllipse(contour)
#             (x, y), (major, minor), angle = ellipse
#             if min_ellipse_size < major < canny_filtered_img_arr.shape[1] and min_ellipse_size < minor < canny_filtered_img_arr.shape[0]:
#                 ellipses.append((int(x), int(y), int(major // 2), int(minor // 2), int(angle)))
#     return ellipses

def detect_ellipses(canny_filtered_img_arr: np.ndarray, min_ellipse_size=10):
    labeled_img, num_features = label(canny_filtered_img_arr)  # Find connected components
    ellipses = []

    for i in range(1, num_features + 1):  
        # Get the indices (y, x) of the contour points
        contour_points = np.column_stack(np.where(labeled_img == i))

        if len(contour_points) >= 5:  # Minimum 5 points needed to fit an ellipse
            ellipse = EllipseModel()
            if ellipse.estimate(contour_points):  # Fit the model
                xc, yc, a, b, theta = ellipse.params  # Center, semi-axes, rotation angle
                
                # Ensure valid ellipse size
                if (min_ellipse_size < a < canny_filtered_img_arr.shape[1] // 2 and
                    min_ellipse_size < b < canny_filtered_img_arr.shape[0] // 2 and
                    a > b):  # Ensure major > minor axis
                    ellipses.append((int(xc), int(yc), int(a), int(b), int(np.degrees(theta))))

    return ellipses


# def draw_ellipses_on_image(original_img_arr, canny_filtered_img_arr):
#     ellipses = detect_ellipses(canny_filtered_img_arr)
#     image_with_ellipses_arr = np.copy(original_img_arr)
#     for x, y, major, minor, angle in ellipses:
#         cv2.ellipse(image_with_ellipses_arr, (x, y), (major, minor), angle, 0, 360, (255, 0, 0), 2)
#     return image_with_ellipses_arr



def draw_ellipses_on_image(original_img_arr, canny_filtered_img_arr,elipse_step_size):
   
    ellipses = detect_ellipses(canny_filtered_img_arr)
    image_with_ellipses_arr = np.copy(original_img_arr)
    # Draw detected contours before ellipse fitting
    # contours, _ = cv2.findContours(canny_filtered_img_arr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(image_with_ellipses_arr, contours, -1, (0, 255, 0), 1)

    for x_center, y_center, major_axis, minor_axis, angle in ellipses:
        cv2.ellipse(image_with_ellipses_arr, (x_center, y_center), (major_axis // 2, minor_axis // 2), angle, 0, 360, (255, 0, 0), 2)

    return image_with_ellipses_arr


def detect_shapes(og_img_arr: np.ndarray, canny_filtered_img_arr: np.ndarray, detect_lines, detect_ellipses, detect_circles, threshold_ratio, circle_step_sz, line_step_sz):
    new_img_arr = np.copy(og_img_arr)
    if detect_circles:
        new_img_arr = draw_circles_on_image(new_img_arr, canny_filtered_img_arr, threshold_ratio, circle_step_sz)
    if detect_ellipses:
        new_img_arr = draw_ellipses_on_image(new_img_arr, canny_filtered_img_arr)
    if detect_lines:
        new_img_arr = draw_lines(new_img_arr, canny_filtered_img_arr, threshold_ratio, line_step_sz)
    return Image(new_img_arr).display_image()

def canny_filter(img_arr, sigma=1, T_low=50, T_high=100, kernel_sz=3):
    edge_detection_processor = edge_detection(img_arr)
    return edge_detection_processor.apply_edge_detection_filter("Canny", T_low, T_high, sigma, kernel_sz)
