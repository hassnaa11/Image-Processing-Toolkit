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
            self.keypoints1, self.descriptors1 = self.apply(self.image_1)
            self.keypoints2, self.descriptors2 = self.apply(self.image_2)
            matches = self.match_keypoints(self.descriptors1, self.descriptors2)
            output_image = self.draw_matches(self.image_1, self.image_2, self.keypoints1, self.keypoints2, matches)
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



    def match_keypoints_ssd(self, descriptors1, descriptors2):
        matches = []
        for i, desc1 in enumerate(descriptors1):
            ssd = np.sum((descriptors2 - desc1) ** 2, axis=1)
            best_match = np.argmin(ssd) # smallest SSD is the closest match
            matches.append((i, best_match, ssd[best_match]))
        
        # Sort matches by distance (SSD)
        matches = sorted(matches, key=lambda x: x[2]) # sort by ssd score (third element in matches) 
        
        matched_image = self.draw_matched_rectangle(self.image_1, self.image_2, self.keypoints1, self.keypoints2, matches)
        return matched_image

    def match_keypoints_ncc(self, descriptors1, descriptors2):
        matches = []
        for i, desc1 in enumerate(descriptors1):
            desc1_std = np.sqrt(np.sum((desc1 - np.mean(desc1)) ** 2))
            desc1 = (desc1 - np.mean(desc1)) / desc1_std  
            desc2_std = np.sqrt( np.sum( (descriptors2 - np.mean(descriptors2, axis=1, keepdims=True))**2, axis=1, keepdims=True) ) 
            descs2_norm = (descriptors2 - np.mean(descriptors2, axis=1, keepdims=True)) / desc2_std
            ncc = np.dot(descs2_norm, desc1) / (len(desc1))
            best_match = np.argmax(ncc)  # Higher NCC better match
            matches.append((i, best_match, ncc[best_match]))
        
        # Sort matches by similarity (NCC, descending order)
        matches = sorted(matches, key=lambda x: x[2], reverse=True)
        matched_image = self.draw_matched_rectangle(self.image_1, self.image_2, self.keypoints1, self.keypoints2, matches)
        return matched_image
    
    def draw_matched_rectangle(self, image1, image2, keypoints1, keypoints2, matches):
        image1_copy = image1.copy()
        image2_copy = image2.copy()

        # Extract the keypoints that have been matched
        matched_points1 = np.array([keypoints1[i1][:2] for i1, i2, _ in matches])
        matched_points2 = np.array([keypoints2[i2][:2] for i1, i2, _ in matches])

        if len(matched_points2) == 0:
            print("No matches found to draw a rectangle.")
            return image1_copy

        # Find the bounding box of matched points in the full image (image2)
        x_min, y_min = np.min(matched_points2, axis=0).astype(int)
        x_max, y_max = np.max(matched_points2, axis=0).astype(int)

        # Draw a rectangle around the detected region in image1 (full image)
        cv2.rectangle(image1_copy, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        return image1_copy
    


    
    # def draw_matched_rectangles(self, image1, image2, keypoints1, keypoints2, matches):
    #     image1 = image1.copy()
    #     image2 = image2.copy()
        
    #     for i1, i2, _ in matches:
    #         x1, y1, _, _ = keypoints1[i1]
    #         x2, y2, _, _ = keypoints2[i2]
            
    #         # Draw rectangles around matched keypoints
    #         cv2.rectangle(image1, (x1 - 5, y1 - 5), (x1 + 5, y1 + 5), (0, 0, 255), 2)
    #         cv2.rectangle(image2, (x2 - 5, y2 - 5), (x2 + 5, y2 + 5), (0, 0, 255), 2)

    #     return image1   
    
    # def draw_matched_rectangles(self, image1, image2, keypoints1, keypoints2, matches):
    #     from PIL import ImageDraw, Image
    #     import cv2
    #     import numpy as np

    #     # Convert images to PIL format (for drawing)
    #     img1_pil = Image.fromarray(image1).convert("RGB")
    #     img2_pil = Image.fromarray(image2).convert("RGB")

    #     # Create a new image with both images side by side
    #     new_image = Image.new("RGB", (img1_pil.width, img1_pil.height))
    #     new_image.paste(img1_pil, (0, 0))

    #     draw = ImageDraw.Draw(new_image)

    #     # Extract (x, y) coordinates from keypoints (ignore octave and scale)
    #     src_pts = np.float32([ [keypoints1[m[0]][0], keypoints1[m[0]][1]] for m in matches[:50] ]).reshape(-1, 1, 2)
    #     dst_pts = np.float32([ [keypoints2[m[1]][0], keypoints2[m[1]][1]] for m in matches[:50] ]).reshape(-1, 1, 2)

    #     # Find homography (maps template points to target points)
    #     M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    #     if M is not None:
    #         # Get the corners of the template image
    #         h, w = image1.shape
    #         template_corners = np.float32([
    #             [0, 0],       # Top-left
    #             [0, h-1],     # Bottom-left
    #             [w-1, h-1],  # Bottom-right
    #             [w-1, 0]      # Top-right
    #         ]).reshape(-1, 1, 2)

    #         # Transform the corners to the target image space
    #         matched_corners = cv2.perspectiveTransform(template_corners, M)

    #         # Offset the corners to account for the side-by-side display
    #         # matched_corners += (img1_pil.width, 0)

    #         # Draw the bounding box (convert to integers)
    #         draw.polygon(
    #             [
    #                 (int(matched_corners[0][0][0]), int(matched_corners[0][0][1])),  # Top-left
    #                 (int(matched_corners[1][0][0]), int(matched_corners[1][0][1])),  # Bottom-left
    #                 (int(matched_corners[2][0][0]), int(matched_corners[2][0][1])),  # Bottom-right
    #                 (int(matched_corners[3][0][0]), int(matched_corners[3][0][1]))   # Top-right
    #             ],
    #             outline="red",
    #             width=3
    #         )

    #     return np.array(new_image)  
    
    

# def draw_matched_rectangles(self, image1, image2, keypoints1, keypoints2, matches):
    #     from PIL import ImageDraw, Image

    #     img1 = Image.fromarray(image1).convert("RGB")
    #     img2 = Image.fromarray(image2).convert("RGB")

    #     new_image = Image.new("RGB", (img1.width + img2.width, max(img1.height, img2.height)))
    #     new_image.paste(img1, (0, 0))
    #     new_image.paste(img2, (img1.width, 0))

    #     draw = ImageDraw.Draw(new_image)

    #     print(f"Total matches found: {len(matches)}")

    #     # Shuffle matches to avoid top-region bias
    #     import random
    #     random.shuffle(matches)

    #     # Select top matches
    #     best_matches = matches[:50]

    #     # Draw match lines
    #     for i1, i2, _ in best_matches:
    #         x1, y1, _, _ = keypoints1[i1]
    #         x2, y2, _, _ = keypoints2[i2]

    #         # Offset x2 correctly to place it in the second image
    #         x2 += img1.width  
    #         draw.rectangle([x1 - 5, y1 - 5, x1 + 5, y1 + 5], outline="red", width=2)
    #         # draw.rectangle([x2 - 5, y2 - 5, x2 + 5, y2 + 5], outline="red", width=2)

    #         # draw.line([(x1, y1), (x2, y2)], fill=(255, 0, 0), width=2)  

    #     return np.array(new_image)

    
    # def draw_matched_rectangles(self, image1, image2, keypoints1, keypoints2, matches):
    #     """ Draw rectangles around matched regions in image2 only. """
    #     from PIL import ImageDraw, Image

    #     img1 = Image.fromarray(image1).convert("RGB")
    #     image1 = image1.copy()
        
    #     for _, i2, _ in matches:
    #         x2, y2, _, _ = keypoints2[i2]
            
    #         # Draw rectangles around matched keypoints in image2
    #         cv2.rectangle(image1, (x2 - 5, y2 - 5), (x2 + 5, y2 + 5), (0, 255, 0), 2)
        
    #     return image1 
 