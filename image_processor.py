import cv2
import numpy as np
from PyQt5.QtGui import *
import cv2

class NoiseAdder:
    def __init__(self, image_array):
        self.image_array = image_array
        
        
    def apply_noise(self, noise_type, parameters):
        if noise_type == 'Uniform':
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
       
        if selected_filter == 'Average':
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
