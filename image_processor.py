import cv2
import numpy as np
from PyQt5.QtGui import *
import pandas as pd
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
            ratio, probability = parameters[0], parameters[1]
            num_salt = np.ceil(probability * ratio * self.image_array.size)
            num_pepper = np.ceil(probability * (1 - ratio) * self.image_array.size)
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
        
        
    def apply_filter(self,sigma, selected_filter, kernel_size):
        filtered_image = self.image_array

        if selected_filter == 'select filter':
            return self.image_array
               
        elif selected_filter == 'Average':
            average_kernel = np.ones(shape=(kernel_size, kernel_size))/ (kernel_size * kernel_size)
            filtered_image = self.apply_kernel(average_kernel, kernel_size)
            # filtered_image = cv2.filter2D(self.image_array,-1, average_kernel) 
            # filtered_image  = cv2.blur(image_array, (3,3))  # opencv filter method
        
        elif selected_filter == 'Gaussian':
            gaussian_kernel = self.get_gaussian_kernel(sigma, kernel_size)
            filtered_image = self.apply_kernel(gaussian_kernel, kernel_size) 
            # filtered_image = cv2.filter2D(self.image_array,-1, average_kernel)
                    
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
            # filtered_image = cv2.medianBlur(image_array, 5)   # opencv filter method
        
        return filtered_image   
    
    def apply_kernel(self, kernel, kernel_size):
        pad_size = kernel_size // 2

        # Handle padding for grayscale and RGB images
        if len(self.image_array.shape) == 2:  # Grayscale
            padded_image = np.pad(self.image_array, ((pad_size, pad_size), (pad_size, pad_size)), mode='reflect')
            filtered_image = np.zeros_like(self.image_array, dtype=np.float32)

            for i in range(self.image_array.shape[0]):
                for j in range(self.image_array.shape[1]):
                    region = padded_image[i:i+kernel_size, j:j+kernel_size]
                    filtered_image[i, j] = np.sum(region * kernel)  # Summation added

        else:  # RGB Image
            padded_image = np.pad(self.image_array, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='reflect')
            filtered_image = np.zeros_like(self.image_array, dtype=np.float32)

            for i in range(self.image_array.shape[0]):
                for j in range(self.image_array.shape[1]):
                    for c in range(self.image_array.shape[2]):  # Loop over channels
                        region = padded_image[i:i+kernel_size, j:j+kernel_size, c]
                        filtered_image[i, j, c] = np.sum(region * kernel)  # Summation added

        # Convert to uint8 
        filtered_image = np.clip(filtered_image, 0, 255).astype(np.uint8)
        return filtered_image
    
            
    def get_gaussian_kernel(self,sigma, kernel_size):
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



class FrequencyFilterProcessor:
    def __init__(self, image_array):
        self.image_array = image_array

    def frequency_filter(self, cutoff, filter_type='Low-Pass Frequency Domain'):
        rows, cols = self.image_array.shape
        crow, ccol = rows // 2, cols // 2  # Center of the frequency domain
        
        # Create meshgrid of distances from the center
        u = np.arange(rows) - crow
        v = np.arange(cols) - ccol
        U, V = np.meshgrid(v, u)
        D = np.sqrt(U**2 + V**2)  # Distance matrix
        
        # Normalize the cutoff frequency to pixel units
        D0 = cutoff * min(rows, cols) / 2  
        
        if filter_type == 'Low-Pass Frequency Domain':
            filter_mask = (D <= D0).astype(np.float32)
        elif filter_type == 'High-Pass Frequency Domain':
            filter_mask = (D > D0).astype(np.float32)
        
        return filter_mask


    def apply_frequency_filter(self, cutoff, filter_type='Low-Pass Frequency Domain'):
        is_rgb = len(self.image_array.shape) == 3 and self.image_array.shape[2] == 3
        
        if is_rgb:
            # Split channels
            channels = cv2.split(self.image_array)
            filtered_channels = [FrequencyFilterProcessor(ch).apply_frequency_filter(cutoff, filter_type) for ch in channels]
            return cv2.merge(filtered_channels)
        
        # Convert image to float32
        image = np.float32(self.image_array)
        
        # Compute FFT and shift zero frequency component to center
        dft = np.fft.fft2(image)
        dft_shift = np.fft.fftshift(dft)
        
        # Create the frequency filter
        filter_mask = self.frequency_filter(cutoff, filter_type)
        
        # Apply filter in frequency domain
        dft_filtered = dft_shift * filter_mask
        
        # Inverse FFT to get the filtered image
        dft_ishift = np.fft.ifftshift(dft_filtered)
        filtered_image = np.fft.ifft2(dft_ishift)
        filtered_image = np.abs(filtered_image)
        
        return np.uint8(cv2.normalize(filtered_image, None, 0, 255, cv2.NORM_MINMAX))
    
    
    
class edge_detection:
    def __init__(self, image_array):
        self.image_array = image_array  
        self.gradient = 0
        
    def apply_kernel(self, kernel, image=None):
        
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
       
        return np.clip(norm_image, 0, 255).astype(np.uint8)

    def apply_edge_detection_filter(self, selected_edge_detection_filter, low_threshold,high_threshold, sigma_gaussian):
        print(low_threshold)
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
            self.gradient = sobel_magnitude
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
            filter =FilterProcessor(self.image_array)
            modified_image=filter.apply_filter(sigma=sigma_gaussian, selected_filter="Gaussian",kernel_size= 3) #apply gaussian filter
            sobel_x = np.array([[-1, 0, 1], 
                                [-2, 0, 2], 
                                [-1, 0, 1]])
            sobel_y = np.array([[-1, -2, -1], 
                                [0, 0, 0], 
                                [1, 2, 1]])
            gx = self.apply_kernel(sobel_x, modified_image)
            gy = self.apply_kernel(sobel_y, modified_image)
            magnitude = np.sqrt(gx**2 + gy**2)
            self.gradient =magnitude
            direction = np.arctan2(gy, gx)
            h,w=magnitude.shape
            angle = np.rad2deg(direction) % 180 
            suppressed = np.zeros((h, w), dtype=np.float32)
            angle = np.rad2deg(direction) % 180  # Convert to degrees
            for i in range(1, h - 1):
                for j in range(1, w - 1):
                    try:
                        q, r = 255, 255
                        if (0 <= angle[i, j] < 45):
                            q = magnitude[i, j + 1]
                            r = magnitude[i, j - 1]
                        elif 45 <= angle[i, j] <90:
                            q = magnitude[i + 1, j - 1]
                            r = magnitude[i - 1, j + 1]
                        elif 90 == angle[i, j]:
                            q = magnitude[i + 1, j]
                            r = magnitude[i - 1, j]
                        elif 90 <= angle[i, j] < 135:
                            q = magnitude[i - 1, j - 1]
                            r = magnitude[i + 1, j + 1]
                        elif 135<= angle[i, j] < 180:
                            q = magnitude[i - 1, j - 1]
                            r = magnitude[i + 1, j + 1]
                        elif angle[i,j]>135:
                            q = magnitude[i, j + 1]
                            r = magnitude[i, j - 1]


                        if (magnitude[i, j] >= q) and (magnitude[i, j] >= r):
                            suppressed[i, j] = magnitude[i, j]
                        else:
                            suppressed[i, j] = 0
                    except IndexError:
                        pass
            strong = 255
            weak = 0
            strong_edges = (suppressed >= high_threshold)
            weak_edges = (suppressed <= low_threshold) 
            intermediate= (suppressed>= low_threshold) & (suppressed < high_threshold)
            result = np.zeros(suppressed.shape, dtype=np.uint8)
            result[strong_edges] = strong
            result[weak_edges] = weak
            result[intermediate]=low_threshold
            for i in range(1, h - 1):
                for j in range(1, w - 1):
                 if  result[i, j] == low_threshold:
                    if np.any( result[i-1:i+2, j-1:j+2] == strong):
                                result[i, j] = strong
                    else:
                                result[i, j] = 0

            processed_array=result
            print("Canny output type:", type(processed_array))
            print("Canny output shape:", processed_array.shape)

                  
        return processed_array
    
    
class thresholding:
    def __init__(self, image_array):
        self.image_array = image_array 

    def apply_threshold(self,thresholding_type ,  window_size=15):
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
                    if self.image_array[i, j] > (local_mean ):
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

