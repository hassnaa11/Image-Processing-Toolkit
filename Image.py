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
        self.rgb_histogram: List = None
        self.gray_histogram = None  
    
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
    
    def plot_histogram(self):
        plt.figure()
        plt.title("Image Histogram")
        plt.xlabel("Pixel Intensity")
        plt.ylabel("Frequency")
        
        if self.isRGB():
            for i, channel in enumerate(RGB_Channels):
                histogram = cv2.calcHist([self.image], [i], None, 256, [0,256])
                self.rgb_histogram.append(histogram)
                
                plt.plot(histogram, color=channel)
                
        else:
            self.gray_histogram, bins = np.histogram(self.image.flatten(), bins=256, range=[0,256])
            plt.plot(self.gray_histogram, color='black') 
                
        plt.xlim([0, 256])
        plt.show()            
    
    
    def plot_CDF(self):
        """plot the comulative distribution function.\n
        requires histogram to be computed first
        """
        if (self.gray_histogram==None) and len(self.rgb_histogram)==0:
            print("Histogram must be computed first")
            return
        
        plt.figure(figsize=(15, 5))
        
        if self.isRGB():
            for i, hist in enumerate(self.rgb_histogram):
                cdf = hist.cumsum()
                cdf_normalized = cdf / cdf.max()
                
                channel = RGB_Channels[i]
                plt.plot(cdf_normalized, color=channel)
                plt.title(f"CDF of {channel} Channel")
                plt.xlabel("Pixel Intensity")
                plt.ylabel("Cumulative Probability")    
        
        else:
            CDF = self.gray_histogram.cumsum()
            normalized_CDF = CDF / CDF.max()     

            plt.plot(cdf_normalized, color='black')
            plt.title("Grayscale Image CDF")
            plt.xlabel("Pixel Intensity")
            plt.ylabel("Cumulative Probability")

        plt.show()
        
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
    
    