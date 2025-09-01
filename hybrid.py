import numpy as np
import cv2
from PIL import Image

class HybridImageProcessor:
    def __init__(self, image1_array, image2_array):
        self.image1 = image1_array
        self.image2 = image2_array

    def low_pass_filter(self, image, kernel_size=5):
        kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size ** 2)
        return cv2.filter2D(image, -1, kernel)

    def high_pass_filter(self, image, kernel_size=5):
        blurred = self.low_pass_filter(image, kernel_size)
        return cv2.subtract(image, blurred)

    def apply_filters(self, low_kernel_size=5, high_kernel_size=5):
        low_passed_image1 = self.low_pass_filter(self.image1, low_kernel_size)
        high_passed_image2 = self.high_pass_filter(self.image2, high_kernel_size)
        return low_passed_image1, high_passed_image2

    def get_filtered_images(self, low_kernel_size=5, high_kernel_size=5):
        return self.apply_filters(low_kernel_size, high_kernel_size)
