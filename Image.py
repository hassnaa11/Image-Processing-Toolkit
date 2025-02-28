import cv2
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QGraphicsScene
from PyQt5.QtGui import *
import cv2
import numpy as np

class Image:
    def __init__(self):
        self.image = None    
    
    def read_image(self, path):
        self.image_path = path
        self.image = cv2.imread(self.image_path)    
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        

    def display_image(self):
        height, width = self.image.shape[:2]
        if len(self.image.shape) == 2: # gray-scale image
            bytes_per_line = width
            q_image = QImage(self.image, width, height, bytes_per_line, QImage.Format_Grayscale8)
        else:  # RGB image 
            bytes_per_line = 3 * width 
            q_image = QImage(self.image, width, height, bytes_per_line, QImage.Format_RGB888)

        # convert QImage to QPixmap
        pixmap = QPixmap.fromImage(q_image) 
        
        # create a QGraphicsScene and add to the pixmap
        scene = QGraphicsScene()
        scene.addPixmap(pixmap)
        return scene
        
    
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
        

    def get_image(self):
        return np.copy(self.image)
 
            
                            