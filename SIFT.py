import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QFileDialog
from PyQt5.QtGui import QPixmap, QImage
from image_processor import FilterProcessor
from scipy.ndimage import  zoom
import random
class SIFTApp(QWidget):
    def __init__(self,image_1,image_2):
        self.image_1=image_1
        self.image_2=image_2
    def apply_SIFT(self):
            self.image_1 = self.load_grayscale_image(self.image_1)
            self.image_2 = self.load_grayscale_image(self.image_2)
            keypoints1, descriptors1 = self.apply(self.image_1)
            keypoints2, descriptors2 = self.apply(self.image_2)
            matches = self.match_keypoints(descriptors1, descriptors2)
            output_image = self.draw_matches(self.image_1, self.image_2, keypoints1, keypoints2, matches)
            return output_image

    def load_grayscale_image(self, img):
        img.rgb2gray()

        return img.get_image()


    def apply(self, image):
        octaves = self.create_octaves(image)
        DoG = self.compute_DoG(octaves)
        keypoints = self.detect_keypoints(DoG)
        descriptors = self.extract_descriptors(image, keypoints)
        return keypoints, descriptors

    def create_octaves(self, image, num_octaves=4, num_scales=4):
        octaves = []
        gaussian=FilterProcessor(self.image_1)
        for _ in range(num_octaves):
            scales = [gaussian.apply_filter( sigma=1.6 * (2 ** (i / num_scales)),selected_filter='Gaussian',kernel_size=3) for i in range(num_scales)]
            octaves.append(scales)
            image = zoom(image, 0.5)

        return octaves

    def compute_DoG(self, octaves):
        return [[octaves[o][i + 1] - octaves[o][i] for i in range(len(octaves[o]) - 1)] for o in range(len(octaves))]

    def detect_keypoints(self, DoG, threshold=2):
        keypoints = []
        for o in range(len(DoG)):
            for i in range(1, len(DoG[o]) - 1):
                for y in range(1, DoG[o][i].shape[0] - 1):
                    for x in range(1, DoG[o][i].shape[1] - 1):
                        pixel = DoG[o][i][y, x]
                        if abs(pixel) > threshold:
                            keypoints.append((x, y, o, i))
        return keypoints



    def extract_descriptors(self, image, keypoints, patch_size=16):
        descriptors = []
        
        for x, y, _, _ in keypoints:
            # Define patch limits
            x1, x2 = max(0, x - patch_size // 2), min(image.shape[1], x + patch_size // 2)
            y1, y2 = max(0, y - patch_size // 2), min(image.shape[0], y + patch_size // 2)
            
            # Extract patch
            patch = image[y1:y2, x1:x2]
            
            # Pad the patch to ensure it's always (patch_size, patch_size)
            pad_x = patch_size - (x2 - x1)
            pad_y = patch_size - (y2 - y1)
            patch = np.pad(patch, ((0, pad_y), (0, pad_x)), mode='constant', constant_values=0)
            
            # Flatten and append
            descriptors.append(patch.flatten())
        
        return np.array(descriptors)


   

    def match_keypoints(self, descriptors1, descriptors2):
        matches = []
        
        for i, desc1 in enumerate(descriptors1):
            distances = np.linalg.norm(descriptors2 - desc1, axis=1)
            best_match = np.argmin(distances)
            matches.append((i, best_match, distances[best_match]))  

        # Randomize matches before sorting
        random.shuffle(matches)
        
        # Sort matches by distance
        matches = sorted(matches, key=lambda x: x[2])

        return matches



    def draw_matches(self, image1, image2, keypoints1, keypoints2, matches, num_matches_to_display=50):
        from PIL import ImageDraw, Image

        img1 = Image.fromarray(image1).convert("RGB")
        img2 = Image.fromarray(image2).convert("RGB")

        new_image = Image.new("RGB", (img1.width + img2.width, max(img1.height, img2.height)))
        new_image.paste(img1, (0, 0))
        new_image.paste(img2, (img1.width, 0))

        draw = ImageDraw.Draw(new_image)

        print(f"Total matches found: {len(matches)}")

        # Shuffle matches to avoid top-region bias
        import random
        random.shuffle(matches)

        # Select top matches
        best_matches = matches[:num_matches_to_display]

        # Draw match lines
        for i1, i2, _ in best_matches:
            x1, y1, _, _ = keypoints1[i1]
            x2, y2, _, _ = keypoints2[i2]

            # Offset x2 correctly to place it in the second image
            x2 += img1.width  

            draw.line([(x1, y1), (x2, y2)], fill=(255, 0, 0), width=2)  

        return np.array(new_image)
