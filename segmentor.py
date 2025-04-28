import numpy as np
import matplotlib.pyplot as plt
from Image import Image

class Segmentor:
    def __init__(self):
        self.__regions_num = None
        self.__seed_selection_tolerance = None
        self.__intensity_diff_threshold = None
        self.__image: Image = None 
        pass
    
    def segment(self, image:Image, method:str, regions_num=None, seed_selection_tolerance=None, intensity_diff_threshold=None):
        self.__image = image
        
        if method == "Region Growing":
            self.assing_region_growing_parameters(regions_num, seed_selection_tolerance, intensity_diff_threshold)
            segmented_image: Image = self.segment_image_region_grow()
            
        return segmented_image    
    
    def apply_region_growing_segmentation(self, regions_num, seed_selection_tolerance, intensity_diff_threshold):
        self.__intensity_diff_threshold = intensity_diff_threshold
        self.__regions_num = regions_num
        self.__seed_selection_tolerance = seed_selection_tolerance
    
    def segment_image_region_grow(self):      
        if self.__image.is_RGB(): 
            rgb_copy_image = Image(self.__image.image)
            self.__image.rgb2gray() # Normalized Gray Scale
            rgb = True
        
        #Automatic seed selection using histogram peaks
        seeds = self.select_seeds(self.__image.image)
        threshold = self.__intensity_diff_threshold

        segmented_image_arr =  rgb_copy_image.image if rgb else self.__image.image # Convert grayscale to RGB for overlay
        for seed in seeds:
            y, x = seed
            mask = self.region_grow(self.__image.image, (y, x), threshold)
            segmented_image_arr[mask] = [100, 100, 0]  # Yellow Marker

        segmented_image = Image(segmented_image_arr)
        return segmented_image

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
        
        regions_num = self.__regions_num
        tolerance = self.__seed_selection_tolerance

        # Indices of bins correspinding to top frequent intensities based on regions_num
        peaks = np.argpartition(hist, -regions_num)[-regions_num:]
        
        seeds = []
        for peak in peaks:
            intensity = bins[peak]
            # Find pixels close to the peak intensity
            y, x = np.where((gray_img_arr >= intensity - tolerance) & (gray_img_arr <= intensity + tolerance))
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


 