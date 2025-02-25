

import cv2
import numpy as np
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QGraphicsView, QGraphicsScene
from PyQt5 import QtWidgets, QtGui, QtCore, uic   # Added uic import

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
        
        