from Image import Image
from image_processor import edge_detection
import numpy as np
from math import floor
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import math
from image_processor import edge_detection
from collections import defaultdict

def circle_hough_transform(canny_filtered_img: np.ndarray, step_sz=20, min_r=1, max_r=100):
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


def detect_shapes(og_img_arr: np.ndarray, canny_filtered_img_arr: np.ndarray, detect_lines, detect_ellipses, detect_circles, threshold_ratio, theta_step_sz, min_r, max_r, minor_step, major_step):
    if detect_circles: 
        new_img_arr = draw_circles_on_image(og_img_arr, canny_filtered_img_arr, threshold_ratio, theta_step_sz, min_r, max_r)
        
    
    elif detect_lines:
        new_img_arr = draw_lines(og_img_arr, canny_filtered_img_arr, threshold_ratio, theta_step_sz)
    
    elif detect_ellipses:
        center_range = max_r - min_r 
        ellipses =  ellipse_hough_transform(canny_filtered_img_arr, og_img_arr, 
        threshold_ratio, major_step, minor_step, theta_step_sz, center_range)
        
    
    new_img = Image(new_img_arr)
    scene = new_img.display_image()
    return scene    

def detect_ellipses(canny_filtered_img_arr: np.ndarray, min_ellipse_size=10, min_r=1, max_r=100):
  
    contours, _ = cv2.findContours(canny_filtered_img_arr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    ellipses = []
    for contour in contours:
        if len(contour) >= 5:  # Minimum points required to fit an ellipse
            ellipse = cv2.fitEllipse(contour)
            (x, y), (major, minor), angle = ellipse
            
            # Filter small ellipses
            if major > min_ellipse_size and minor > min_ellipse_size:
                ellipses.append((int(x), int(y), int(major), int(minor), int(angle)))

    return ellipses

def draw_ellipses_on_image(original_img_arr, canny_filtered_img_arr,elipse_step_size):
   
    ellipses = detect_ellipses(canny_filtered_img_arr)
    image_with_ellipses_arr = np.copy(original_img_arr)

    for x_center, y_center, major_axis, minor_axis, angle in ellipses:
        cv2.ellipse(image_with_ellipses_arr, (x_center, y_center), (major_axis // 2, minor_axis // 2), angle, 0, 360, (255, 0, 0), 2)

    return image_with_ellipses_arr


def canny_filter(img_arr , sigma = 1, T_low: int = 50, T_high: int = 100, kernel_sz=3):
    edge_detection_processor = edge_detection(img_arr)
    canny_filtered_img_arr = edge_detection_processor.apply_edge_detection_filter("Canny", T_low, T_high, sigma, kernel_sz)
    
    return canny_filtered_img_arr


def ellipse_hough_transform(edge_img_arr, orig_img_arr, threshold_ratio, step_a=5, step_b=5, step_theta=5, center_range=20):
    """
    Detects ellipses using an improved Hough Transform.

    Args:
        edge_img: Binary edge image (e.g., from Canny detector).
        orig_img: Original grayscale image for gradient computation (optional).
        step_a: Step size for semi-major axis 'a' in pixels.
        step_b: Step size for semi-minor axis 'b' in pixels.
        step_theta: Step size for orientation angle 'theta' in degrees.
        center_range: Range (in pixels) to search for ellipse centers around each edge point.

    Returns:
        ellipses: List of detected ellipses (x_c, y_c, a, b, theta).
    """
    height, width = edge_img_arr.shape
    edge_points = np.argwhere(edge_img_arr == 255)

    # Use original image for gradient if provided, otherwise use edge image
    Gx, Gy = np.gradient(orig_img_arr)
    accumulator = defaultdict(int)

    # Iterate over edge points
    for (y, x) in edge_points:
        # Skip if gradient is zero
        if Gx[y, x] == 0 and Gy[y, x] == 0:
            continue

        # Compute gradient direction (normal to edge)
        gradient_angle = np.arctan2(Gy[y, x], Gx[y, x])
        gradient_angle_deg = np.rad2deg(gradient_angle)

        # Search for possible centers in a small grid around (x, y)
        for dx in range(-center_range, center_range + 1, 2):  # Step by 2 for efficiency
            for dy in range(-center_range, center_range + 1, 2):
                x_c = x + dx
                y_c = y + dy
                if not (0 <= x_c < width and 0 <= y_c < height):
                    continue

                # Limit theta to values near gradient direction (±45°)
                theta_start = max(0, gradient_angle_deg - 45)
                theta_end = min(180, gradient_angle_deg + 45)
                for theta in np.arange(theta_start, theta_end, step_theta):
                    theta_rad = np.deg2rad(theta)

                    # Try different a and b values
                    for a in range(10, min(width, height) // 4, step_a):
                        for b in range(5, a, step_b):  # b < a
                            # Rotated ellipse equation
                            cos_t = np.cos(theta_rad)
                            sin_t = np.sin(theta_rad)
                            x_diff = x - x_c
                            y_diff = y - y_c
                            term1 = (x_diff * cos_t + y_diff * sin_t) ** 2 / (a ** 2)
                            term2 = (x_diff * sin_t - y_diff * cos_t) ** 2 / (b ** 2)
                            ellipse_eq = term1 + term2

                            if 0.9 <= ellipse_eq <= 1.1:  # Tighter tolerance
                                accumulator[(x_c, y_c, a, b, theta)] += 1

    # Extract ellipses with sufficient votes
    max_votes = max(accumulator.values(), default=0)
    if max_votes == 0:
        return []  # No ellipses detected
    threshold = threshold_ratio * max_votes  # Slightly higher threshold
    ellipses = [(x_c, y_c, a, b, theta) for (x_c, y_c, a, b, theta), votes in accumulator.items() 
                if votes >= threshold]

    return ellipses

# Example usage (assuming you have an edge image and original image):
# import cv2
# orig_img = cv2.imread("image.png", cv2.IMREAD_GRAYSCALE)
# edge_img = cv2.Canny(orig_img, 100, 200)
# ellipses = ellipse_hough_transform(edge_img, orig_img, step_a=5, step_b=5, step_theta=5, center_range=20)
