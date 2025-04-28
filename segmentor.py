import numpy as np
from skimage.color import rgb2gray
import matplotlib.pyplot as plt

class Segmentor:
    def __init__(self):
        pass

    def segment_image(self, image, threshold=0.1):
        """
        Segments objects in an RGB or grayscale image using region growing from scratch.
        
        Args:
            image (ndarray): Input image (RGB or grayscale).
            threshold (float): Threshold for region growing similarity.
        
        Returns:
            segmented_image (ndarray): Image with segmented objects overlaid.
        """
        # Step 1: Convert RGB to grayscale if needed
        if len(image.shape) == 3:  # RGB image
            gray_image = rgb2gray(image)  # Convert to grayscale
        else:
            gray_image = image  # Already grayscale

        # Step 2: Automatic seed selection using histogram peaks
        seeds = self.select_seeds(gray_image)

        # Step 3: Perform region growing for each seed
        segmented_image = np.stack((gray_image,) * 3, axis=-1)  # Convert grayscale to RGB for overlay
        for seed in seeds:
            y, x = seed
            mask = self.region_grow(gray_image, (y, x), threshold)
            segmented_image[mask] = [255, 0, 0]  # Mark segmented regions in red

        return segmented_image

    def select_seeds(self, gray_image):
        """
        Automatically selects seed points based on histogram peaks.
        
        Args:
            gray_image (ndarray): Grayscale image.
        
        Returns:
            seeds (list): List of seed points (y, x).
        """
        # Compute histogram
        hist, bins = np.histogram(gray_image.flatten(), bins=256, range=(0, 1))

        # Find peaks in the histogram
        peaks = np.argpartition(hist, -3)[-3:]  # Select top 3 peaks
        seeds = []
        for peak in peaks:
            intensity = bins[peak]
            # Find pixels close to the peak intensity
            y, x = np.where((gray_image >= intensity - 0.01) & (gray_image <= intensity + 0.01))
            if len(y) > 0:
                seeds.append((y[0], x[0]))  # Select the first pixel as a seed

        return seeds

    def region_grow(self, image, seed, threshold):
        """
        Implements region growing from scratch.
        
        Args:
            image (ndarray): Grayscale image.
            seed (tuple): Starting point for region growing (y, x).
            threshold (float): Similarity threshold for region growing.
        
        Returns:
            mask (ndarray): Binary mask of the grown region.
        """
        height, width = image.shape
        mask = np.zeros_like(image, dtype=bool)  # Binary mask for the region
        visited = np.zeros_like(image, dtype=bool)  # Track visited pixels
        region_mean = image[seed]  # Initialize region mean with the seed intensity
        stack = [seed]  # Stack for region growing (DFS)

        while stack:
            y, x = stack.pop()
            if visited[y, x]:
                continue
            visited[y, x] = True

            # Check if the pixel is similar to the region mean
            if abs(image[y, x] - region_mean) <= threshold:
                mask[y, x] = True
                region_mean = (region_mean + image[y, x]) / 2  # Update region mean

                # Add neighbors to the stack
                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # 4-connectivity
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < height and 0 <= nx < width and not visited[ny, nx]:
                        stack.append((ny, nx))

        return mask


# Example Usage
if __name__ == "__main__":
    # Load an image (replace with your image path)
    image = plt.imread("path_to_image.jpg")
    if image.max() > 1:  # Normalize if needed
        image = image / 255.0

    # Create a Segmentor instance
    segmentor = Segmentor()

    # Segment the image
    segmented_image = segmentor.segment_image(image, threshold=0.1)

    # Display the original and segmented images
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(segmented_image)
    plt.title("Segmented Image")
    plt.axis("off")
    plt.show()  