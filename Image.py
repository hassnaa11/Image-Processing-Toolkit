import cv2
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QGraphicsScene
from PyQt5.QtGui import *
import cv2
import numpy as np
import matplotlib.pyplot as plt

class Image:
    def __init__(self):
        self.image = None
        self.image_path = None    
    
    def read_image(self, path):
        self.image_path = path
        self.image = cv2.imread(self.image_path)
        
        if self.isRGB(): self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            
        else: self.image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
                
        
    def isRGB(self):
        if len(self.image.shape == 2):
            print("Greyscale image read")
            return False
        
        print("RGB image read")
        return True    

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
    
    def plot_histogram(self):
        plt.figure()
        plt.title("Image Histogram")
        plt.xlabel("Pixel Intensity")
        plt.ylabel("Frequency")
        
        if self.isRGB():
            channels = ("red", "green", "blue")
         
            for i, channel in enumerate(channels):
                histogram = cv2.calcHist([self.image], [i], None, 256, [0,256])
                plt.plot(histogram, color=channels[i])
         
        else:
            histogram, bins = np.histogram(self.image.flatten(), bins=256, range=[0,256])
            plt.plot(histogram, color='black')
             
                
        plt.xlim([0, 256])
        plt.show()            
    
    
    def plot_distribution_curve(self):
        pass
 
    def RGB2GRAY(self):
        """return a grayscale copy of a colored img
        """
        if self.image.shape == 2:
            print("Image is already loaded in grayscale")
            return
        
        gray_img = cv2.cvtcolor(self.image, cv2.COLOR_BGR2GRAY)
        return gray_img
            
    def get_image(self):
        return np.copy(self.image)
    
    