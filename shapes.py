from Image import Image
from image_processor import edge_detection
import numpy as np
from math import floor
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import math
from image_processor import edge_detection

def circle_hough_transform(canny_filtered_img: np.ndarray, step_sz=20):
    height, width = canny_filtered_img.shape
    
    small_dim = height if height<width else width
    max_r = small_dim // 2
    min_r = 5
    
    radius_values = np.arange(min_r, max_r + 1)
    accumulator = np.zeros((height, width, len(radius_values)), dtype=np.uint64)

    edge_points = np.argwhere(canny_filtered_img == 255)
    print("Pixel Wide Edges Found")

    # Precompute Thetas to avoid heavy repetetive calling of deg2rad
    thetas = np.deg2rad(np.arange(-90, 90, step_sz))
    cos_thetas = np.cos(thetas)
    sin_thetas = np.sin(thetas)
    
    print("Thetas Precomputed")
    
    max = 0
    for idx, r in enumerate(radius_values):
        for x, y in edge_points:
            for cos_theta, sin_theta in zip(cos_thetas, sin_thetas): 
                a = int(x - r * cos_theta)
                b = int(y - r * sin_theta)
                if 0 <= a < height and 0 <= b < width:
                    accumulator[a, b, idx] += 1
                    if accumulator[a, b, idx] > max: max = accumulator[a, b, idx]

    print("accummulator formed")
    return accumulator, radius_values, max


def detect_circles(canny_filtered_img_arr: np.ndarray, threshold_ratio=0.7, step_sz=20):
    accumulator, radius_values, max = circle_hough_transform(canny_filtered_img_arr, step_sz)
    threshold = threshold_ratio * max
    
    circles = []
    h, w, r_len = accumulator.shape
    for r_idx in range(r_len):
        acc_slice = accumulator[:, :, r_idx]
        centers = np.argwhere(acc_slice > threshold)
        for a, b in centers:
            circles.append((b, a, radius_values[r_idx]))  # (x_center, y_center, radius)
    
    print("Circles Detected")
    return circles
    

def draw_circles_on_image(original_img_arr, canny_filtered_img_arr, threshold_ratio, step_sz):
    
    circles = detect_circles(canny_filtered_img_arr, threshold_ratio, step_sz)
    image_with_circles_arr = np.copy(original_img_arr)
    
    for x_center, y_center, radius in circles:
        cv2.circle(image_with_circles_arr, (int(x_center), int(y_center)), int(radius), (0, 0, 255), 2)
        
    print("Circles Drawn")
    return image_with_circles_arr       
    

def line_hough_transform(canny_filtered_img: np.ndarray, theta_step=1):
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
    thetas = np.deg2rad(np.arange(0, 180, theta_step))  # Theta values in radians
    rhos = np.arange(-diag_len, diag_len)  # Rho values

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

    return accumulator, rhos, thetas


def detect_lines(canny_filtered_img_arr: np.ndarray, threshold_ratio=0.5, min_line_length=20):
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
    accumulator, rhos, thetas = line_hough_transform(canny_filtered_img_arr)

    # Calculate threshold
    threshold = threshold_ratio * np.max(accumulator)

    # Find peaks in accumulator
    lines = []
    height, width = canny_filtered_img_arr.shape

    for rho_idx in range(len(rhos)):
        for theta_idx in range(len(thetas)):
            if accumulator[rho_idx, theta_idx] > threshold:
                rho = rhos[rho_idx]
                theta = thetas[theta_idx]

                # Convert (rho, theta) to line endpoints for visualization
                if sin(theta) != 0:  # Non-vertical line
                    x1, y1 = 0, int(rho / sin(theta))
                    x2, y2 = width - 1, int((rho - (width - 1) * cos(theta)) / sin(theta))
                else:  # Vertical line
                    x1, y1 = int(rho), 0
                    x2, y2 = int(rho), height - 1

                # Calculate line length
                line_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

                # Add to list if line is long enough
                if line_length >= min_line_length:
                    lines.append((rho, theta, (x1, y1, x2, y2)))

    return lines


def draw_lines(image, lines):
    """
    Draw detected lines on the image

    Args:
        image: Input image
        lines: List of lines in (rho, theta, (x1, y1, x2, y2)) format

    Returns:
        result_image: Image with lines drawn
    """
    # Create a copy of the input image
    result_image = image.copy()

    # If the image is grayscale, convert to RGB
    if len(result_image.shape) == 2:
        result_image = cv2.cvtColor(result_image, cv2.COLOR_GRAY2RGB)

    # Draw each line
    for rho, theta, (x1, y1, x2, y2) in lines:
        # Draw the line
        cv2.line(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return result_image


def display_results(original_image, edge_image, lines):
    """
    Display original image, edge image, and detected lines

    Args:
        original_image: Original input image
        edge_image: Canny edge detected image
        lines: List of detected lines
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Show original image
    axes[0].imshow(original_image, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # Show edge image
    axes[1].imshow(edge_image, cmap='gray')
    axes[1].set_title('Edge Image')
    axes[1].axis('off')

    # Show image with detected lines
    result_image = draw_lines(original_image, lines)
    axes[2].imshow(result_image)
    axes[2].set_title('Detected Lines')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()


def detect_shapes(og_img_arr: np.ndarray, canny_filtered_img_arr: np.ndarray, detect_lines, detect_ellipses, detect_circles, threshold_ratio, circle_step_sz):
    if detect_circles: 
        new_img_arr = draw_circles_on_image(og_img_arr, canny_filtered_img_arr, threshold_ratio, circle_step_sz)
        new_img = Image(new_img_arr)
        scene = new_img.display_image()
        return scene


def canny_filter(img_arr , sigma = 1, T_low: int = 50, T_high: int = 100):
    edge_detection_processor = edge_detection(img_arr)
    canny_filtered_img_arr = edge_detection_processor.apply_edge_detection_filter("Canny", T_low, T_high, sigma)
    
    return canny_filtered_img_arr