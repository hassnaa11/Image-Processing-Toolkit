import cv2
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QGraphicsScene
from PyQt5.QtGui import *
import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List

RGB_Channels = ("red", "green", "blue")

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
        if len(self.image.shape) == 2:
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
    
    def compute_histogram(self):
        """Compute histogram for RGB or Grayscale image
        """
        if self.isRGB():
            histogram = []
            
            for i, channel in enumerate(RGB_Channels):
                hist = cv2.calcHist([self.image], [i], None, 256, [0,256])
                histogram.append(hist)
                
        else: histogram, bins = np.histogram(self.image.flatten(), bins=256, range=[0,256]) 
                
        self.__hg = histogram
        return histogram
        
    def compute_CDF(self, histogram):
        """compute the comulative distribution function for RGB or Grayscale image.\n
        requires histogram to be computed first
        """
        if not self.__hg:
            print("Histogram must be computed first")
            return
        
        if self.isRGB():
            CDF = []
            
            for i, hist in enumerate(histogram):
                cdf = hist.cumsum()
                cdf = cdf / cdf.max()
                CDF.append(cdf)   
        else:
            CDF = histogram.cumsum()
            CDF = CDF / CDF.max()
            
        self.__cdf = CDF
        return CDF         
        
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
    
    