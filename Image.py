import cv2
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QGraphicsScene
from PyQt5.QtGui import *
import cv2
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from typing import List
rgb_channels = ["red", "green", "blue"]

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
    
    
    def get_histogram(self):
        return self.__hg
    
    def get_CDF(self):
        return self.__cdf
    
    def compute_histogram(self):
        """Compute histogram for RGB or Grayscale image
        """
        if self.isRGB():
            histogram = []
            
            
            for i, channel in enumerate(rgb_channels):
                hist = cv2.calcHist([self.image], [i], None, [256], [0,256])
                histogram.append(hist)
                
        else: histogram = cv2.calcHist([self.image], [0], None, [256], [0, 256]) 
                
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

            
    def get_image(self):
        return np.copy(self.image)
    

    def plot_histogram(self):
        """return a canvas for the histogram
        """
        
        if self.__hg is None:
            self.__hg = self.compute_histogram()
        
        # Create figure and canvas
        fig = Figure(figsize=(5, 4), dpi=100)
        canvas = FigureCanvasQTAgg(fig)
        ax = fig.add_subplot(111)
        bin_edges = np.arange(257)
        
        # Plot histogram
        if self.isRGB():
            colors = ['r', 'g', 'b']
            
            for i, hist in enumerate(self.__hg):
                ax.bar(bin_edges[:-1], hist.flatten(), width=1, color=colors[i], 
                  alpha=0.5, label=rgb_channels[i])
            ax.legend()
            ax.set_title('RGB Histogram')
        else:
            ax.bar(bin_edges[:-1], self.__hg.flatten(), width=1, color='k')
            ax.set_title('Grayscale Histogram')
            
        ax.set_xlabel('Pixel Value')
        ax.set_ylabel('Frequency')
        ax.grid(True)
        
        self.__hg_canvas = canvas
        return canvas
    
    def get_histogram_canvas(self):
        return self.__hg_canvas
    
    def plot_cdf(self):
        """return a canvas for cdf
        """
        if self.__cdf is None:
            if self.__hg is None:
                self.__hg = self.compute_histogram()
            
            self.__cdf = self.compute_CDF(self.__hg)    
            
        # Create figure and canvas
        fig = Figure(figsize=(5, 4), dpi=100)
        canvas = FigureCanvasQTAgg(fig)
        ax = fig.add_subplot(111)
        
        # Plot CDF
        if self.isRGB():
            colors = ['r', 'g', 'b']
            for i, cdf in enumerate(self.__cdf):
                ax.plot(cdf, color=colors[i], label=rgb_channels[i])
            ax.legend()
            ax.set_title('RGB CDF')
        else:
            ax.plot(self.__cdf, color='k')
            ax.set_title('Grayscale CDF')
            
        ax.set_xlabel('Pixel Value')
        ax.set_ylabel('Cumulative Probability')
        ax.grid(True)
        
        self.__cdf_canvas = canvas
        return canvas 
    
    def get_cdf_canvas(self):
        return self.__cdf_canvas   