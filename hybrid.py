import numpy as np
import cv2
from PIL import Image
from PyQt5.uic.properties import QtGui, QtWidgets
from qtpy import uic

class HybridImageProcessor:
 def __init__(self, uic=None):
   
    uic.loadUi('ui.ui', self)

    self.image1 = None
    self.image2 = None
    self.filtered_image1 = None
    self.filtered_image2 = None

    # Connect buttons
    self.input1_button.clicked.connect(lambda: self.upload_image(1))
    self.input2_button.clicked.connect(lambda: self.upload_image(2))
    self.filters_combobox.currentIndexChanged.connect(self.apply_selected_filter)


def upload_image(self, key):
    file_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.jpg *.jpeg)")
    if file_path:
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)  # Read as grayscale
        if key == 1:
            self.image1 = image
            self.input1_image.setScene(self.display_image(image))
        elif key == 2:
            self.image2 = image
            self.input2_image.setScene(self.display_image(image))


def apply_selected_filter(self):
    if self.image1 is None or self.image2 is None:
        return  # Ensure images are loaded before applying filters

    selected_filter = self.filters_combobox.currentText()
    processor = HybridImageProcessor(self.image1, self.image2)

    if selected_filter == "Low Pass":
        self.filtered_image1 = processor.low_pass_filter(self.image1, 5)
        self.filtered_image2 = processor.low_pass_filter(self.image2, 5)
    elif selected_filter == "High Pass":
        self.filtered_image1 = processor.high_pass_filter(self.image1, 5)
        self.filtered_image2 = processor.high_pass_filter(self.image2, 5)
    else:
        return  # No valid filter selected

    self.display_filtered_images()


def display_filtered_images(self):
    """Update UI with filtered images and automatically generate hybrid image"""
    if self.filtered_image1 is not None and self.filtered_image2 is not None:
        self.output1_image.setScene(self.display_image(self.filtered_image1))
        self.output2_image.setScene(self.display_image(self.filtered_image2))

        # Generate hybrid image by combining filtered images
        hybrid_image = cv2.addWeighted(self.filtered_image1, 0.5, self.filtered_image2, 0.5, 0)
        self.hybrid_output_image.setScene(self.display_image(hybrid_image))


def display_image(self, image):
    """Convert image to QPixmap and display it in QGraphicsView"""
    height, width = image.shape
    bytes_per_line = width
    q_img = QtGui.QImage(image.data, width, height, bytes_per_line, QtGui.QImage.Format_Grayscale8)
    pixmap = QtGui.QPixmap.fromImage(q_img)
    scene = QtWidgets.QGraphicsScene()
    scene.addPixmap(pixmap)
    return scene
