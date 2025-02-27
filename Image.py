import cv2
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QGraphicsScene
import numpy as np
from scipy import ndimage
class Image:
    def __init__(self, path):
        self.image_path = path
        self.read_image()
    
    
    def read_image(self):
        self.image = cv2.imread(self.image_path)    
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
       
        
    def display_image(self):
        height, width, channel = self.image.shape
        bytes_per_line = channel * width
        
        # convert image to QImage
        q_image = QImage(self.image, width, height, bytes_per_line, QImage.Format_RGB888) 
        
        # convert QImage to QPixmap
        pixmap = QPixmap.fromImage(q_image) 
        
        # create a QGraphicsScene and add to the pixmap
        scene = QGraphicsScene()
        scene.addPixmap(pixmap)
        return scene
        
        
    def add_noise(self, noise_type, parameters):
        if noise_type == 'Uniform':
            min_range, max_range = parameters[0], parameters[1]
            noise = np.random.randint(min_range, max_range, self.image.shape).astype(np.int16)
            self.image = np.clip(self.image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        elif noise_type == 'Gaussian':
            mean, sigma = parameters[0], parameters[1]
            gauss = np.random.normal(mean, sigma, self.image.shape).astype(np.uint8)
            self.image = cv2.add(self.image, gauss)
        
        elif noise_type == 'Salt & Pepper':
            # print("parameters ", parameters)
            ratio, probability = parameters[0], parameters[1]
            num_salt = np.ceil(probability * ratio * self.image.size)
            num_pepper = np.ceil(probability * (1 - ratio) * self.image.size)
            # print("num_salt:  ", num_salt,"num_pepper: ", num_pepper)
            # print( "self.image.size: ", self.image.size)
            # print("self.image.shape[:2]:  ", self.image.shape[:2])

            coordinates = [np.random.randint(0, i , int(num_salt)) for i in self.image.shape[:2]]
            self.image[coordinates[0],coordinates[1], :] = 255
            
            coordinates = [np.random.randint(0, i , int(num_pepper)) for i in self.image.shape[:2]]
            self.image[coordinates[0],coordinates[1], :] = 0
    def apply_kernel(self, kernel, image=None):
        # Use the preserved image if no specific image is passed
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
        threshold = 50
        norm_image = np.where(norm_image > threshold, norm_image, 0)
        return np.clip(norm_image, 0, 255).astype(np.uint8)

    def apply_edge_detection_filter(self, selected_edge_detection_filter):
        image_to_process = np.copy(self.image)
        if len(image_to_process.shape) == 3 and image_to_process.shape[2] == 3:
            image_to_process = np.mean(image_to_process, axis=2).astype(np.uint8)
        
        if selected_edge_detection_filter == "Sobel":
            sobel_x = np.array([[-1, 0, 1], 
                                [-2, 0, 2], 
                                [-1, 0, 1]])
            sobel_y = np.array([[-1, -2, -1], 
                                [0, 0, 0], 
                                [1, 2, 1]])
            gx = self.apply_kernel(sobel_x, image_to_process)
            gy = self.apply_kernel(sobel_y, image_to_process)
            sobel_magnitude = np.sqrt(gx**2 + gy**2)
            
            processed_array = self.normalize_and_adjust(sobel_magnitude)
            processed_array = np.stack((processed_array,) * 3, axis=-1)
            self.processed_array = processed_array
            self.image =self. processed_array
           
            
        elif selected_edge_detection_filter == "Roberts":
            roberts_x = np.array([[1, 0], 
                                [0, -1]])
            roberts_y = np.array([[0, 1], 
                                [-1, 0]])
            gx = self.apply_kernel(roberts_x, image_to_process)
            gy = self.apply_kernel(roberts_y, image_to_process)
            roberts_magnitude = np.sqrt(gx**2 + gy**2)
            
            processed_array = self.normalize_and_adjust(roberts_magnitude)
            processed_array = np.stack((processed_array,) * 3, axis=-1)
            self.processed_array = processed_array
            self.image = self.processed_array
           
            
        elif selected_edge_detection_filter == "Prewitt":
            prewitt_x = np.array([[-1, 0, 1], 
                                [-1, 0, 1], 
                                [-1, 0, 1]])
            prewitt_y = np.array([[1, 1, 1], 
                                [0, 0, 0], 
                                [-1, -1, -1]])
            gx = self.apply_kernel(prewitt_x, image_to_process)
            gy = self.apply_kernel(prewitt_y, image_to_process)
            prewitt_magnitude = np.sqrt(gx**2 + gy**2)
            
            processed_array = self.normalize_and_adjust(prewitt_magnitude)
            processed_array = np.stack((processed_array,) * 3, axis=-1)
            self.processed_array = processed_array
            self.image = self.processed_array
        elif selected_edge_detection_filter == "Canny":
            threshold1 = 100  # Lower threshold
            threshold2 = 200  # Upper threshold
            edges = cv2.Canny(image_to_process, threshold1, threshold2)
            
            # Convert edges to 3 channels (RGB) for display consistency
            processed_array = np.stack((edges,) * 3, axis=-1)
            self.image = processed_array
        return self.display_image()
        


                        