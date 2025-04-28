import numpy as np
import matplotlib.pyplot as plt
from Image import Image

class Segmentor:
    def __init__(self):
        print("Segmentor class initialized")
        pass

    def segment(self, image:Image, method:str):
        if method == "Region Grow":
            self.segment_image_region_grow(image)
    
    def segment_image_region_grow(self, image: Image, threshold=0.1):      
        if image.is_RGB(): 
            rgb_copy_image = Image(image.image)
            image.rgb2gray()
            rgb = True
        
        #Automatic seed selection using histogram peaks
        seeds = self.select_seeds(image.image)

        segmented_image_arr =  rgb_copy_image.image if rgb else image.image # Convert grayscale to RGB for overlay
        for seed in seeds:
            y, x = seed
            mask = self.region_grow(image.image, (y, x), threshold)
            segmented_image_arr[mask] = [100, 100, 0]  # Yellow Marker

        return segmented_image_arr

    def select_seeds(self, gray_img_arr):
        """
        Automatically selects seed points based on histogram peaks.
        
        Args:
            gray_img_arr (ndarray): Grayscale image.
        
        Returns:
            seeds (list): List of seed points (y, x).
        """
        # Compute histogram
        hist, bins = np.histogram(gray_img_arr.flatten(), bins=256, range=(0, 1))

        # Find peaks in the histogram
        peaks = np.argpartition(hist, -3)[-3:]  # Select top 3 peaks
        seeds = []
        for peak in peaks:
            intensity = bins[peak]
            # Find pixels close to the peak intensity
            y, x = np.where((gray_img_arr >= intensity - 0.01) & (gray_img_arr <= intensity + 0.01))
            if len(y) > 0:
                seeds.append((y[0], x[0]))  # Select the first pixel as a seed

        return seeds

    def region_grow(self, image_arr, seed, threshold):
        """
        Implements region growing from scratch.
        
        Args:
            image (ndarray): Grayscale image.
            seed (tuple): Starting point for region growing (y, x).
            threshold (float): Similarity threshold for region growing.
        
        Returns:
            mask (ndarray): Binary mask of the grown region.
        """
        height, width = image_arr.shape
        mask = np.zeros_like(image_arr, dtype=bool)  # Binary mask for the region
        visited = np.zeros_like(image_arr, dtype=bool)  # Track visited pixels
        region_mean = image_arr[seed]  # Initialize region mean with the seed intensity
        stack = [seed]  # Stack for region growing (DFS)

        while stack:
            y, x = stack.pop()
            if visited[y, x]:
                continue
            visited[y, x] = True

            # Check if the pixel is similar to the region mean
            if abs(image_arr[y, x] - region_mean) <= threshold:
                mask[y, x] = True
                region_mean = (region_mean + image_arr[y, x]) / 2  # Update region mean

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
    segmented_image_arr = segmentor.segment_image(image, threshold=0.1)

    # Display the original and segmented images
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(segmented_image_arr)
    plt.title("Segmented Image")
    plt.axis("off")
    plt.show()  