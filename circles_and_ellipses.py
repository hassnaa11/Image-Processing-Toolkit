from Image import Image
import numpy as np
from math import floor
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def circle_hough_transform(canny_filtered_img: np.ndarray, step_sz=5):
    height, width = canny_filtered_img.shape
    
    big_dim = height if height>width else width
    min_r, max_r = 1, floor(big_dim/2)
    
    radius_values = np.arange(min_r, max_r + 1)
    accumulator = np.zeros((height, width, len(radius_values)), dtype=np.uint64)

    edge_points = np.argwhere(canny_filtered_img == 255)

    for idx, r in enumerate(radius_values):
        for x, y in edge_points:
            for theta in range(0, 360, step_sz):  # 5-degree step for speed
                a = int(x - r * np.cos(np.deg2rad(theta)))
                b = int(y - r * np.sin(np.deg2rad(theta)))
                if 0 <= a < height and 0 <= b < width:
                    accumulator[a, b, idx] += 1

    return accumulator, radius_values

def detect_circles(canny_filtered_img_arr: np.ndarray, threshold_ratio=0.7):
    accumulator, radius_values = circle_hough_transform(canny_filtered_img_arr)
    threshold = threshold_ratio * np.max(accumulator)
    
    circles = []
    h, w, r_len = accumulator.shape
    for r_idx in range(r_len):
        acc_slice = accumulator[:, :, r_idx]
        centers = np.argwhere(acc_slice > threshold)
        for a, b in centers:
            circles.append((b, a, radius_values[r_idx]))  # (x_center, y_center, radius)
    return circles
    

def draw_circles_on_image(original_img, circles):
    # If grayscale, convert to RGB for color drawing
    if len(original_img.shape) == 2:
        img_to_show = np.stack([original_img]*3, axis=-1)
    else:
        img_to_show = original_img.copy()
    
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img_to_show, cmap='gray')
    
    for x_center, y_center, radius, votes in circles:
        circle = patches.Circle((x_center, y_center), radius, 
                                linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(circle)
        # Optionally: annotate votes
        ax.text(x_center, y_center, f"{int(votes)}", color='yellow', fontsize=8)
    
    ax.set_title("Detected Circles")
    plt.axis('off')
    plt.show()   
    
