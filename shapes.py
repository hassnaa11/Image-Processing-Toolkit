from Image import Image
from image_processor import edge_detection
import numpy as np
from math import floor
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import math
from collections import defaultdict

def circle_hough_transform(og_img_arr, canny_filtered_img: np.ndarray, step_sz=20, min_r=1, max_r=100):
    height, width = canny_filtered_img.shape
    max_r = max_r
    min_r = min_r
    
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


def detect_circles(canny_filtered_img_arr: np.ndarray, threshold_ratio=0.7, step_sz=20, min_r=1, max_r=100):
    accumulator, radius_values, max = circle_hough_transform(canny_filtered_img_arr, step_sz, min_r, max_r)
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
    

def draw_circles_on_image(original_img_arr, canny_filtered_img_arr, threshold_ratio, step_sz, min_r, max_r):
    
    circles = detect_circles(canny_filtered_img_arr, threshold_ratio, step_sz, min_r, max_r)
    image_with_circles_arr = np.copy(original_img_arr)
    
    for x_center, y_center, radius in circles:
        cv2.circle(image_with_circles_arr, (int(x_center), int(y_center)), int(radius), (0, 0, 255), 2)
        
    print("Circles Drawn")
    return image_with_circles_arr       
    

import numpy as np
import cv2
from collections import defaultdict

def line_hough_transform(canny_filtered_img: np.ndarray, orig_img_arr=None, theta_step=15):
    """
    Implements Hough transform for line detection.

    Args:
        canny_filtered_img: Binary edge image (output from Canny edge detector).
        orig_img_arr: Original image (RGB or grayscale) for gradient info (optional).
        theta_step: Angular step size in degrees.

    Returns:
        accumulator: Hough accumulator (dict).
        rhos: Array of rho values.
        thetas: Array of theta values.
        cos_thetas, sin_thetas: Precomputed cos and sin values.
    """
    height, width = canny_filtered_img.shape
    diag_len = int(np.sqrt(height ** 2 + width ** 2))
    thetas = np.deg2rad(np.arange(-90, 90, theta_step))
    rhos = np.arange(-diag_len, diag_len + 1)
    accumulator = defaultdict(int)  # Sparse accumulator

    edge_points = np.argwhere(canny_filtered_img == 255)
    cos_thetas = np.cos(thetas)
    sin_thetas = np.sin(thetas)

    # Optional gradient filtering
    if orig_img_arr is not None:
        grad_img = cv2.cvtColor(orig_img_arr, cv2.COLOR_RGB2GRAY).astype(float) if len(orig_img_arr.shape) == 3 else orig_img_arr.astype(float)
        Gy, Gx = np.gradient(grad_img)

    for y, x in edge_points:
        if orig_img_arr is not None and Gx[y, x] == 0 and Gy[y, x] == 0:
            continue  # Skip weak edges
        for theta_idx, (cos_t, sin_t) in enumerate(zip(cos_thetas, sin_thetas)):
            rho = int(x * cos_t + y * sin_t)
            rho_idx = rho + diag_len
            if 0 <= rho_idx < len(rhos):
                accumulator[(rho_idx, theta_idx)] += 1

    return accumulator, rhos, thetas, cos_thetas, sin_thetas


def detect_lines(canny_filtered_img_arr: np.ndarray, orig_img_arr=None, threshold_ratio=0.5, theta_step=15, min_line_length=20):
    """
    Detects lines from Hough accumulator.

    Args:
        canny_filtered_img_arr: Binary edge image.
        orig_img_arr: Original image for gradients (optional).
        threshold_ratio: Ratio of max accumulator value for threshold.
        theta_step: Angular step size in degrees.
        min_line_length: Minimum line length in pixels.

    Returns:
        lines: List of (rho, theta, (x1, y1, x2, y2)).
    """
    accumulator, rhos, thetas, cosines, sines = line_hough_transform(canny_filtered_img_arr, orig_img_arr, theta_step)
    threshold = threshold_ratio * max(accumulator.values(), default=0)
    if threshold < 5:  # Minimum reasonable threshold
        return []

    height, width = canny_filtered_img_arr.shape
    lines = []
    for (rho_idx, theta_idx), votes in accumulator.items():
        if votes > threshold:
            rho = rhos[rho_idx]
            theta = thetas[theta_idx]
            if sines[theta_idx] != 0:  # Non-vertical line
                x1, y1 = 0, int(round(rho / sines[theta_idx]))
                x2, y2 = width - 1, int((rho - (width - 1) * cosines[theta_idx]) / sines[theta_idx])
            else:  # Vertical line
                x1, y1 = int(rho), 0
                x2, y2 = int(rho), height - 1
            line_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            if line_length >= min_line_length:
                lines.append((rho, theta, (x1, y1, x2, y2)))

    return lines


def draw_lines(original_img_arr, canny_filtered_img_arr, threshold_ratio=0.5, theta_step=15, min_line_length=20):
    """
    Draw detected lines on the image.

    Args:
        original_img_arr: Input image (RGB or grayscale).
        canny_filtered_img_arr: Binary edge image.
        threshold_ratio: Ratio of max accumulator value for threshold.
        theta_step: Angular step size in degrees.
        min_line_length: Minimum line length in pixels.

    Returns:
        image_with_line_arr: Image with lines drawn.
    """
    lines = detect_lines(canny_filtered_img_arr, original_img_arr, threshold_ratio, theta_step, min_line_length)
    image_with_line_arr = original_img_arr.copy()

    # Convert grayscale to RGB for drawing
    if len(image_with_line_arr.shape) == 2:
        image_with_line_arr = cv2.cvtColor(image_with_line_arr, cv2.COLOR_GRAY2RGB)

    for rho, theta, (x1, y1, x2, y2) in lines:
        cv2.line(image_with_line_arr, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return image_with_line_arr


def detect_shapes(og_img_arr: np.ndarray, canny_filtered_img_arr: np.ndarray, detect_lines, detect_ellipses, detect_circles, min_line_length, threshold_ratio, theta_step_sz, min_r, max_r, minor_step, major_step):
    if detect_circles: 
        new_img_arr = draw_circles_on_image(og_img_arr, canny_filtered_img_arr, threshold_ratio, theta_step_sz, min_r, max_r)
        
    
    elif detect_lines:
        new_img_arr = draw_lines(og_img_arr, canny_filtered_img_arr, threshold_ratio, theta_step_sz, min_line_length)
    
    elif detect_ellipses:
        center_range = max_r - min_r 
        ellipses =  ellipse_hough_transform(canny_filtered_img_arr, og_img_arr, 
        threshold_ratio, major_step, minor_step, theta_step_sz, center_range)
        new_img_arr = draw_ellipses(og_img_arr, ellipses)
        
    
    new_img = Image(new_img_arr)
    scene = new_img.display_image()
    return scene    



def canny_filter(img_arr , sigma = 1, T_low: int = 50, T_high: int = 100, kernel_sz=3):
    edge_detection_processor = edge_detection(img_arr)
    canny_filtered_img_arr = edge_detection_processor.apply_edge_detection_filter("Canny", T_low, T_high, sigma, kernel_sz)
    
    return canny_filtered_img_arr


def ellipse_hough_transform(edge_img_arr, orig_img_arr, threshold_ratio=0.6, step_a=5, step_b=5, step_theta=5, center_range=20):
    height, width = edge_img_arr.shape
    edge_points = np.argwhere(edge_img_arr == 255)

    # Handle RGB or grayscale input directly with NumPy/OpenCV
    if len(orig_img_arr.shape) == 3:  # RGB
        grad_img = cv2.cvtColor(orig_img_arr, cv2.COLOR_RGB2GRAY).astype(float)
    else:  # Grayscale
        grad_img = orig_img_arr.astype(float)

    Gy, Gx = np.gradient(grad_img)
    accumulator = defaultdict(int)

    for (y, x) in edge_points:
        if Gx[y, x] == 0 and Gy[y, x] == 0:
            continue

        gradient_angle = np.arctan2(Gy[y, x], Gx[y, x])
        gradient_angle_deg = np.rad2deg(gradient_angle)
        
        # Precompute theta values
        theta_start = max(0, gradient_angle_deg - 45)
        theta_end = min(180, gradient_angle_deg + 45)
        theta_vals = np.arange(theta_start, theta_end, step_theta)
        theta_rads = np.deg2rad(theta_vals)
        cos_ts = np.cos(theta_rads)
        sin_ts = np.sin(theta_rads)

        for dx in range(-center_range, center_range + 1, 2):
            for dy in range(-center_range, center_range + 1, 2):
                x_c = x + dx
                y_c = y + dy
                if not (0 <= x_c < width and 0 <= y_c < height):
                    continue

                for theta_idx, theta in enumerate(theta_vals):
                    cos_t = cos_ts[theta_idx]
                    sin_t = sin_ts[theta_idx]

                    for a in range(10, min(width, height) // 4, step_a):
                        a_sq = a ** 2
                        for b in range(5, a, step_b):
                            b_sq = b ** 2
                            x_diff = x - x_c
                            y_diff = y - y_c
                            term1 = (x_diff * cos_t + y_diff * sin_t) ** 2 / a_sq
                            term2 = (x_diff * sin_t - y_diff * cos_t) ** 2 / b_sq
                            ellipse_eq = term1 + term2

                            if 0.9 <= ellipse_eq <= 1.1:
                                accumulator[(x_c, y_c, a, b, theta)] += 1

    max_votes = max(accumulator.values(), default=0)
    if max_votes < 5:  # Minimum vote threshold
        return []
    threshold = max(threshold_ratio * max_votes, 5)  # Ensure at least 5 votes
    ellipses = [(x_c, y_c, a, b, theta) for (x_c, y_c, a, b, theta), votes in accumulator.items() 
                if votes >= threshold]

    return ellipses


def draw_ellipses(orig_img_arr, ellipses):
    """
    Draw detected ellipses on the original image and return the new image as an ndarray.

    Args:
        orig_img_arr: Original image (RGB or grayscale NumPy array).
        ellipses: List of (x_c, y_c, a, b, theta) tuples from ellipse_hough_transform.

    Returns:
        img_draw: New image with ellipses drawn (ndarray, same type as orig_img_arr).
    """
    # Create a copy of the image to draw on
    img_draw = orig_img_arr.copy()

    # Check if image is RGB or grayscale
    is_rgb = len(orig_img_arr.shape) == 3

    for (x_c, y_c, a, b, theta) in ellipses:
        # OpenCV uses (center, (major, minor), angle) format
        # Convert theta (degrees) to OpenCV format (clockwise from x-axis)
        angle = -theta  # Negate to convert counterclockwise to clockwise
        
        # OpenCV expects (width, height) as full axes, but a, b are semi-axes
        axes = (int(a), int(b))  # Use semi-axes directly
        
        # Draw ellipse
        color = (0, 255, 0) if is_rgb else 255  # Green for RGB, white for grayscale
        thickness = 2
        img_draw = cv2.ellipse(img_draw, 
                              center=(int(x_c), int(y_c)), 
                              axes=axes, 
                              angle=angle, 
                              startAngle=0, 
                              endAngle=360, 
                              color=color, 
                              thickness=thickness)

    return img_draw

# Example usage
# edge_img = cv2.Canny(orig_img_arr, 100, 200)
# ellipses = ellipse_hough_transform(edge_img, orig_img_arr, threshold_ratio=0.6)
# new_img = draw_ellipses(orig_img_arr, ellipses)
# cv2.imwrite("output.png", new_img)  # Optional: save it
# cv2.imshow("Ellipses", new_img); cv2.waitKey(0)  # Optional: display it

# Example usage
# edge_img = cv2.Canny(orig_img_arr, 100, 200)
# ellipses = ellipse_hough_transform(edge_img, orig_img_arr, threshold_ratio=0.6)
# draw_ellipses(orig_img_arr, ellipses, "output_with_ellipses.png")
