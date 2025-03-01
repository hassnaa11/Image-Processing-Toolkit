import cv2
import numpy as np
from PyQt5.QtGui import *
import cv2

class NoiseAdder:
    def __init__(self, image_array):
        self.image_array = image_array
        
        
    def apply_noise(self, noise_type, parameters):
        if noise_type == 'select noise':
            return self.image_array
        
        elif noise_type == 'Uniform':
            min_range, max_range = parameters[0], parameters[1]
            noise = np.random.randint(min_range, max_range, self.image_array.shape).astype(np.int16)
            self.image_array = np.clip(self.image_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        elif noise_type == 'Gaussian':
            mean, sigma = parameters[0], parameters[1]
            gauss = np.random.normal(mean, sigma, self.image_array.shape).astype(np.uint8)
            self.image_array = cv2.add(self.image_array, gauss)
        
        elif noise_type == 'Salt & Pepper':
            # print("parameters ", parameters)
            ratio, probability = parameters[0], parameters[1]
            num_salt = np.ceil(probability * ratio * self.image_array.size)
            num_pepper = np.ceil(probability * (1 - ratio) * self.image_array.size)
            # print("num_salt:  ", num_salt,"num_pepper: ", num_pepper)
            # print( "self.image_array.size: ", self.image_array.size)
            # print("self.image_array.shape[:2]:  ", self.image_array.shape[:2])
            if len(self.image_array.shape) == 2:
                coordinates = [np.random.randint(0, i , int(num_salt)) for i in self.image_array.shape]
                self.image_array[coordinates[0],coordinates[1]] = 255
                
                coordinates = [np.random.randint(0, i , int(num_pepper)) for i in self.image_array.shape]
                self.image_array[coordinates[0],coordinates[1]] = 0    
            else:
                coordinates = [np.random.randint(0, i , int(num_salt)) for i in self.image_array.shape[:2]]
                self.image_array[coordinates[0],coordinates[1], :] = 255
                
                coordinates = [np.random.randint(0, i , int(num_pepper)) for i in self.image_array.shape[:2]]
                self.image_array[coordinates[0],coordinates[1], :] = 0
        
        return self.image_array


class FilterProcessor:
    def __init__(self, image_array):
        self.image_array = image_array  
        
        
    def apply_filter(self, selected_filter, kernel_size):
        filtered_image = self.image_array

        if selected_filter == 'select filter':
            return self.image_array
               
        elif selected_filter == 'Average':
            average_kernel = np.ones(shape=(kernel_size, kernel_size))/ (kernel_size * kernel_size)
            filtered_image = cv2.filter2D(self.image_array,-1, average_kernel)
            # filtered_image  = cv2.blur(image_array, (3,3))

        elif selected_filter == 'Gaussian ':
            gaussian_kernel = self.get_gaussian_kernel(kernel_size)
            filtered_image = cv2.filter2D(self.image_array,-1, gaussian_kernel)
            # filtered_image = cv2.GaussianBlur(image_array, (5,5), 0)
                    
        elif selected_filter == 'Median':
            pad_size = kernel_size // 2
            if len(self.image_array.shape) == 2:
                padded_image = np.pad(self.image_array, ((pad_size, pad_size), (pad_size, pad_size)), mode='constant')
            else:
                padded_image = np.pad(self.image_array, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='constant')
            filtered_image = np.zeros_like(self.image_array)

            for i in range(self.image_array.shape[0]):
                for j in range(self.image_array.shape[1]):
                    region = padded_image[i:i+kernel_size, j:j+kernel_size]
                    filtered_image[i, j] = np.median(region, axis=(0, 1))   
            # filtered_image = cv2.medianBlur(image_array, 5)
        elif selected_filter == 'Low-Pass Filter':
            kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size ** 2)
            filtered_image = cv2.filter2D(self.image_array, -1, kernel)
        return filtered_image   
    
            
    def get_gaussian_kernel(self, kernel_size):
        if kernel_size == 3:
            sigma = 1
        elif kernel_size == 5:
            sigma = 1.4
        elif kernel_size == 7:
            sigma = 1.6        

        kernel = np.zeros(shape=(kernel_size, kernel_size))
        center = kernel_size // 2
        for x in range(kernel_size):
            for y in range(kernel_size):
                x_shifted = x - center
                y_shifted = y - center
                kernel[x][y] = (1 / (2 * np.pi * sigma**2)) * np.exp(-(x_shifted**2 + y_shifted**2) / (2 * sigma**2))
        
        kernel /= np.sum(kernel)
        return kernel   
    
    
    def histogram_equalization(self):
        # flatten the image to 1D array
        flat_image = self.image_array.flatten()
        
        # compute histogram
        hist, bins = np.histogram(flat_image, bins=256, range=[0,256])
        
        # compute CDF
        cdf = hist.cumsum()
        
        # normalize the image to map the values between 0, 255
        cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
        
        # Use the normalized CDF as a lookup table
        equalized_image = cdf_normalized[flat_image]
        
        # reshape back to original image shape
        equalized_image = equalized_image.reshape(self.image_array.shape).astype(np.uint8)
        
        return equalized_image
    
    
    def rgb_to_grayscale(self):
        # luminosity method
        grayscale_image = np.dot(self.image_array[..., :3], [0.2989, 0.5870, 0.1140])
        return grayscale_image.astype(np.uint8)
    
class edge_detection:
    def __init__(self, image_array):
        self.image_array = image_array  
        
    def apply_kernel(self, kernel, image=None):
        if image is None:
            self.preserved_image = self.image
        if len(image.shape) == 3:  # Color image
            gray_image = np.mean(image, axis=2).astype(np.uint8)
        else:
            gray_image = image

        kernel_size = kernel.shape[0]
        pad = kernel_size // 2
        padded_image = np.pad(gray_image, pad, mode='constant', constant_values=0)
        filtered_image = np.zeros_like(gray_image, dtype=np.int32) 
        for i in range(gray_image.shape[0]):
            for j in range(gray_image.shape[1]):
                region = padded_image[i:i+kernel_size, j:j+kernel_size]
                filtered_image[i, j] = np.sum(region * kernel)

        return filtered_image

    def normalize_and_adjust(self,image):
        # Normalize to 0-255
        norm_image = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-5) * 255
        threshold = 30
        norm_image = np.where(norm_image > threshold, norm_image, 0)
        return np.clip(norm_image, 0, 255).astype(np.uint8)

    def apply_edge_detection_filter(self, selected_edge_detection_filter):
      
        if len(self.image_array.shape) == 3 and self.image_array.shape[2] == 3:
            self.image_array = np.mean(self.image_array, axis=2).astype(np.uint8)
        
        if selected_edge_detection_filter == "select edge detection filter":
            return self.image_array
        
        elif selected_edge_detection_filter == "Sobel":
            sobel_x = np.array([[-1, 0, 1], 
                                [-2, 0, 2], 
                                [-1, 0, 1]])
            sobel_y = np.array([[-1, -2, -1], 
                                [0, 0, 0], 
                                [1, 2, 1]])
            gx = self.apply_kernel(sobel_x, self.image_array)
            gy = self.apply_kernel(sobel_y, self.image_array)
            sobel_magnitude = np.sqrt(gx**2 + gy**2)
            
            processed_array = self.normalize_and_adjust(sobel_magnitude)
            processed_array = np.stack((processed_array,) * 3, axis=-1)
            
        elif selected_edge_detection_filter == "Roberts":
            roberts_x = np.array([[1, 0], 
                                [0, -1]])
            roberts_y = np.array([[0, 1], 
                                [-1, 0]])
            gx = self.apply_kernel(roberts_x, self.image_array)
            gy = self.apply_kernel(roberts_y, self.image_array)
            roberts_magnitude = np.sqrt(gx**2 + gy**2)
            print("gx gy")
            processed_array = self.normalize_and_adjust(roberts_magnitude)
            processed_array = np.stack((processed_array,) * 3, axis=-1)


        elif selected_edge_detection_filter == "Prewitt":
            prewitt_x = np.array([[-1, 0, 1], 
                                [-1, 0, 1], 
                                [-1, 0, 1]])
            prewitt_y = np.array([[1, 1, 1], 
                                [0, 0, 0], 
                                [-1, -1, -1]])
            gx = self.apply_kernel(prewitt_x, self.image_array)
            gy = self.apply_kernel(prewitt_y, self.image_array)
            prewitt_magnitude = np.sqrt(gx**2 + gy**2)
            processed_array = self.normalize_and_adjust(prewitt_magnitude)
            processed_array = np.stack((processed_array,) * 3, axis=-1)
          
           
        elif selected_edge_detection_filter == "Canny":
            threshold1 = 100  # Lower threshold
            threshold2 = 200  # Upper threshold
            edges = cv2.Canny(self.image_array, threshold1, threshold2)
            
            # Convert edges to 3 channels (RGB) for display consistency
            processed_array = np.stack((edges,) * 3, axis=-1)
            
        return processed_array
    
class thresholding:
    def __init__(self, image_array):
        self.image_array = image_array 

    def apply_threshold(self,thresholding_type ,  window_size=15, C=10):
        # Convert to grayscale if it's a color image
        if len( self.image_array.shape) == 3:
            self.image_array = np.mean(self.image_array, axis=2).astype(np.uint8)
        
        if thresholding_type == "select thresholding type":
            return self.image_array
        
        elif thresholding_type == "Local":
            
            # Padding for border handling
            pad = window_size // 2
            padded_image = np.pad(self.image_array, pad, mode='constant', constant_values=0)
            
            binary_image = np.zeros_like(self.image_array)
            for i in range(self.image_array.shape[0]):
                for j in range(self.image_array.shape[1]):
                    # Extract local region
                    local_region = padded_image[i:i+window_size, j:j+window_size]
                    
                    # Calculate mean of the local window
                    local_mean = np.mean(local_region)
                    
                    # Apply threshold
                    if self.image_array[i, j] > (local_mean - C):
                        binary_image[i, j] = 255
                    else:
                        binary_image[i, j] = 0

        if thresholding_type == "Global":
            threshold=128
            # Convert to grayscale if it's a color image
            if len(self.image_array.shape) == 3:
                self.image_array = np.mean(self.imag_array, axis=2).astype(np.uint8)
            
            # Apply global thresholding
            binary_image = np.where(self.image_array > threshold, 255, 0)
            return binary_image.astype(np.uint8)
            
        return binary_image.astype(np.uint8)

