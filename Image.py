import cv2
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QGraphicsScene
import numpy as np
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
            
            