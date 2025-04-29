import numpy as np
import matplotlib.pyplot as plt
from Image import Image
from sklearn.cluster import MeanShift, estimate_bandwidth

class Segmentor:
    def __init__(self):
        self.__regions_num = None
        self.__seed_selection_tolerance = None
        self.__intensity_diff_threshold = None
        self.__K = None
        self.__image: Image = None 
        pass

    
    def segment(self, image:Image, method:str="Region Growing", regions_num=None, seed_selection_tolerance=None, intensity_diff_threshold=None, K=None, iterations=None):
        self.__image = image
        segmented_image: Image = None
        
        if method == "Region Growing":
            self.assing_region_growing_parameters(regions_num, seed_selection_tolerance, intensity_diff_threshold)
            segmented_image = self.segment_image_region_grow()
            
        elif method== "K Means":
            segmented_image = self.segment_image_k_means(self.__image, K)
        
        elif method == "Mean Shift":
            segmented_image = self.segment_image_mean_shift(self.__image, iterations)    
            
        return segmented_image    
 
    
    def assing_region_growing_parameters(self, regions_num, seed_selection_tolerance, intensity_diff_threshold):
        self.__intensity_diff_threshold = intensity_diff_threshold
        self.__regions_num = regions_num
        self.__seed_selection_tolerance = seed_selection_tolerance
    
    
    def segment_image_region_grow(self):      
        gray_cpy_image = Image(np.copy(self.__image.image))
        
        if self.__image.is_RGB():
            rgb = True
            gray_cpy_image.rgb2gray()
        else:  rgb = False
            
        #Normalize from 0 to 1
        gray_cpy_image.image = gray_cpy_image.image / gray_cpy_image.image.max() 
        
        #Automatic seed selection using histogram peaks
        seeds = self.select_seeds(gray_cpy_image.image)
        
        threshold = self.__intensity_diff_threshold
        overlay_image_arr = np.copy(self.__image.image)
        
        if rgb: colors = np.random.randint(0, 256, size=(len(seeds), 3))
        else: colors = np.linspace(50, 255, len(seeds), dtype=np.uint8)
        
        for i, seed in enumerate(seeds):
            y, x = seed
            mask = self.region_grow(gray_cpy_image.image, (y, x), threshold)
            overlay_image_arr[mask] = colors[i] 

        segmented_image = Image(overlay_image_arr)
        return segmented_image


    def select_seeds(self, gray_img_arr: np.ndarray):
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


    def segment_image_k_means(self, image,k):
        image_array = np.array(image.image)
        pixels = image_array.reshape(-1, 3).astype(np.float32)

        # take random centroid 
        # the number of them = the number of clusters "k"
        
        indices = np.random.choice(len(pixels), int(k), replace=False)
        centroids = pixels[indices]

        for _ in range(50):
            distances = np.linalg.norm(pixels[:, np.newaxis] - centroids, axis=2)
            labels = np.argmin(distances, axis=1)

            # calculate the mean to assign the new centroids
            new_centroids = np.array([
                pixels[labels == cluster].mean(axis=0) if np.any(labels == cluster) else centroids[cluster]
                for cluster in range(int(k))
            ])

            if np.allclose(centroids, new_centroids):
                break
            centroids = new_centroids

        segmented_pixels = centroids[labels].reshape(image_array.shape).astype(np.uint8)    
        segmented_pixels_rgb = segmented_pixels
      
        return  segmented_pixels_rgb
    
    
    def segment_image_mean_shift(self, image, Num_of_iteration=300):
        image_array = np.array(image.image)
        
        flat_image = image_array.reshape(-1, 3).astype(np.float32)

        
        bandwidth = estimate_bandwidth(flat_image, quantile=0.06, n_samples=3000)
        ms = MeanShift(bandwidth=bandwidth, max_iter=Num_of_iteration, bin_seeding=True)
        ms.fit(flat_image)
        labels = ms.labels_  
        cluster_centers = ms.cluster_centers_  

        segmented_image = cluster_centers[labels].reshape(image_array.shape).astype(np.uint8)

     
        print("Segmented Image Shape:", segmented_image.shape)

        return segmented_image