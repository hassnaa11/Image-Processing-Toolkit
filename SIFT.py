
import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QFileDialog
from PyQt5.QtGui import QPixmap, QImage
from image_processor import FilterProcessor
from scipy.ndimage import zoom, gaussian_filter, sobel
import random
from scipy.ndimage import rotate
import math
from scipy.ndimage import affine_transform
from scipy import ndimage as ndi


class SIFTApp(QWidget):
    def __init__(self, image_1, image_2):
        super().__init__()
        self.image_1 = image_1
        self.image_2 = image_2

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
        DoG = self.DoG_pyramid(octaves)
        keypoints = self.detect_keypoints(DoG)
        localized = self.localize_keypoints(DoG, keypoints)  
        oriented = self.assign_orientation(octaves, localized) 
        descriptors = self.extract_descriptors(octaves,oriented)
        return oriented, descriptors
    
    def create_octaves(self,image):
        num_of_octaves=5
        num_of_scales=5
        octaves=[]
        for o in range(num_of_octaves):
            scale_images = []
            base_image = zoom(image, 1 / (2 ** o))  
            for s in range(num_of_scales):
                sigma = 1.6 * (2 ** (s / num_of_scales))  
                smoothed = gaussian_filter(base_image, sigma=sigma)
                scale_images.append(smoothed)
            octaves.append(scale_images)
        return octaves
    

    def DoG_pyramid(self, octaves):
        dog_pyramid = []
        for scale_images in octaves:
            dog_octave = []
            for i in range(1, len(scale_images)):
                dog = scale_images[i] - scale_images[i - 1]
                dog_octave.append(dog)
            dog_pyramid.append(dog_octave)

        return dog_pyramid
    
    def detect_keypoints(self,dog_pyramid):
        keypoints = []
        contrast_threshold=0.03
        for o, dog_octave in enumerate(dog_pyramid):  
            for s in range(1, len(dog_octave) - 1):    
                current = dog_octave[s]
                prev = dog_octave[s - 1]
                next_ = dog_octave[s + 1]

                for y in range(1, current.shape[0] - 1):
                    for x in range(1, current.shape[1] - 1):
                        value = current[y, x]
                        if abs(value) < contrast_threshold:
                            continue
                        patch = np.array([
                            prev[y-1:y+2, x-1:x+2], #(3 rows and 3 columns).
                            current[y-1:y+2, x-1:x+2],
                            next_[y-1:y+2, x-1:x+2]
                        ])
                        if value == patch.max() or value == patch.min():
                            keypoints.append((x, y, o, s))
        return keypoints 
    
    def localize_keypoints(self,dog_pyramid, keypoints, contrast_thresh=0.03):
        refined = []
        for x, y, o, s in keypoints:
            dog = dog_pyramid[o][s]
            if 1 <= x < dog.shape[1] - 1 and 1 <= y < dog.shape[0] - 1:
                val = dog[int(y), int(x)]
                if abs(val) > contrast_thresh:
                    refined.append((x, y, o, s))
        return refined


    def assign_orientation(self,octaves, keypoints, num_bins=36):
        assigned = []

        for x, y, o, s in keypoints:
            image = octaves[o][s]
            dx = sobel(image, axis=1)
            dy = sobel(image, axis=0)

            hist = np.zeros(num_bins)
            radius = 3
            for i in range(-radius, radius + 1):
                for j in range(-radius, radius + 1):
                    yy = int(y + i)
                    xx = int(x + j)
                    if 1 <= yy < image.shape[0]-1 and 1 <= xx < image.shape[1]-1:
                        gx = dx[yy, xx]
                        gy = dy[yy, xx]
                        mag = np.sqrt(gx**2 + gy**2)
                        angle = np.rad2deg(np.arctan2(gy, gx)) % 360
                        bin_idx = int(angle / 360 * num_bins)
                        hist[bin_idx] += mag

            max_val = np.max(hist)
            for i in range(num_bins):
                if hist[i] >= 0.8 * max_val:
                    assigned.append((x, y, o, s, i * 360 / num_bins))

        return assigned

    def extract_descriptors(self,octaves, keypoints):
        
        descriptors = []

        for kp in keypoints:
            if len(kp) == 5:
                x, y, o, s, orientation = kp  
            else:
                x, y, o, s = kp  
                orientation = 0

            image = octaves[o][int(s)]  

            dx = sobel(image, axis=1)  #  the horizontal gradient
            dy = sobel(image, axis=0)  #  the vertical gradient
            magnitude = np.hypot(dx, dy)  #  the magnitude of the gradient
            orientation_map = np.arctan2(dy, dx)  # the gradient direction 

            x, y = int(x), int(y)  #

       
            if x - 8 < 0 or x + 8 >= image.shape[1] or y - 8 < 0 or y + 8 >= image.shape[0]:
                continue  

            patch_mag = magnitude[y-8:y+8, x-8:x+8]  
            patch_ori = orientation_map[y-8:y+8, x-8:x+8]  

            if patch_mag.size == 0 or patch_ori.size == 0:
                continue  

            descriptor = np.zeros((4, 4, 8))  # Create a 4x4 grid of 8 orientation bins for the descriptor

            for i in range(16):
                for j in range(16):
                    cell_x = j // 4 
                    cell_y = i // 4  
                    angle = patch_ori[i, j] % (2 * np.pi)  # Normalize the angle to the range [0, 2π]
                    bin_idx = int(angle / (2 * np.pi / 8))  # Determine the bin index based on the angle
                    descriptor[cell_y, cell_x, bin_idx] += patch_mag[i, j]  # Accumulate gradient magnitude in the bin

            descriptors.append(descriptor.flatten())  

        return np.array(descriptors)  

    def match_keypoints(self, descriptors1, descriptors2):
        matches = []
        for i, desc1 in enumerate(descriptors1):
            distances = np.linalg.norm(descriptors2 - desc1, axis=1)
            best_match = np.argmin(distances)
            matches.append((i, best_match, distances[best_match]))
        random.shuffle(matches)
        matches = sorted(matches, key=lambda x: x[2])
        return matches
    def draw_matches(self, image1, image2, keypoints1, keypoints2, matches, num_matches_to_display=25,circle_radius=5):
        from PIL import ImageDraw, Image
        img1 = Image.fromarray(image1).convert("RGB")
        img2 = Image.fromarray(image2).convert("RGB")

        new_image = Image.new("RGB", (img1.width + img2.width, max(img1.height, img2.height)))
        new_image.paste(img1, (0, 0))
        new_image.paste(img2, (img1.width, 0))

        draw = ImageDraw.Draw(new_image)

        print(f"Total matches found: {len(matches)}")

        best_matches = matches[:num_matches_to_display]
        for i1, i2, _ in best_matches:
            x1, y1 = keypoints1[i1][0], keypoints1[i1][1]
            x2, y2 = keypoints2[i2][0], keypoints2[i2][1]
            draw.ellipse([(x1 - circle_radius, y1 - circle_radius), (x1 + circle_radius, y1 + circle_radius)], outline=(0, 255, 0), width=2)
            x2 += img1.width  # Shift x2 to the second image space
            draw.ellipse([(x2 - circle_radius, y2 - circle_radius), (x2 + circle_radius, y2 + circle_radius)], outline=(0, 255, 0), width=2)

            draw.line([(x1, y1), (x2, y2)], fill=(255, 0, 0), width=2)
        return np.array(new_image)

# from numpy import all, any, array, arctan2, cos, sin, exp, dot, log, logical_and, roll, sqrt, stack, trace, unravel_index, pi, deg2rad, rad2deg, where, zeros, floor, full, nan, isnan, round, float32
# from numpy.linalg import det, lstsq, norm
# from cv2 import resize, GaussianBlur, subtract, KeyPoint, INTER_LINEAR, INTER_NEAREST
# from functools import cmp_to_key
# import logging
# class SIFTApp(QWidget):
#     def __init__(self, image_1, image_2):
#         super().__init__()
#         self.image_1 = image_1
#         self.image_2 = image_2
#         self.float_tolerance = 1e-7

#     def apply_SIFT(self):
#         self.image_1=self. load_grayscale_image(self.image_1)
#         self.image_2=self. load_grayscale_image(self.image_2)
#         Key_points1,descriptors1=self.computeKeypointsAndDescriptors(self.image_1)
#         Key_points2,descriptors2=self.computeKeypointsAndDescriptors(self.image_2)
#         matches = self.match_keypoints(descriptors1, descriptors2)
#         output_image = self.draw_matches(self.image_1, self.image_2, Key_points1, Key_points2, matches)
#         return output_image
#     def load_grayscale_image(self, img):
#         img.rgb2gray()
#         return img.get_image()

#     def computeKeypointsAndDescriptors(self ,image, sigma=1.6, num_intervals=3, assumed_blur=0.5, image_border_width=5):
      
#         base_image = self.generateBaseImage(image, sigma, assumed_blur)
#         num_octaves = self.computeNumberOfOctaves(base_image.shape)
#         gaussian_kernels = self.generateGaussianKernels(sigma, num_intervals)
#         gaussian_images = self.generateGaussianImages(base_image, num_octaves, gaussian_kernels)
#         dog_images = self.generateDoGImages(gaussian_images)
#         keypoints = self.findScaleSpaceExtrema(gaussian_images, dog_images, num_intervals, sigma, image_border_width)
#         keypoints = self.removeDuplicateKeypoints(keypoints)
#         keypoints = self.convertKeypointsToInputImageSize(keypoints)
#         descriptors = self.generateDescriptors(keypoints, gaussian_images)
#         return keypoints, descriptors

#     def generateBaseImage(self,image, sigma, assumed_blur):

#         image = resize(image, (0, 0), fx=2, fy=2, interpolation=INTER_LINEAR)
#         sigma_diff = sqrt(max((sigma ** 2) - ((2 * assumed_blur) ** 2), 0.01))
#         return GaussianBlur(image, (0, 0), sigmaX=sigma_diff, sigmaY=sigma_diff)  # the image blur is now sigma instead of assumed_blur

#     def computeNumberOfOctaves(self,image_shape):
       
#         return int(round(log(min(image_shape)) / log(2) - 1))

#     def generateGaussianKernels(self,sigma, num_intervals):
   
#         num_images_per_octave = num_intervals + 3
#         k = 2 ** (1. / num_intervals)
#         gaussian_kernels = zeros(num_images_per_octave)  # scale of gaussian blur necessary to go from one blur scale to the next within an octave
#         gaussian_kernels[0] = sigma

#         for image_index in range(1, num_images_per_octave):
#             sigma_previous = (k ** (image_index - 1)) * sigma
#             sigma_total = k * sigma_previous
#             gaussian_kernels[image_index] = sqrt(sigma_total ** 2 - sigma_previous ** 2)
#         return gaussian_kernels

#     def generateGaussianImages(self,image, num_octaves, gaussian_kernels):


#         gaussian_images = []

#         for octave_index in range(num_octaves):
#             gaussian_images_in_octave = []
#             gaussian_images_in_octave.append(image)  # first image in octave already has the correct blur
#             for gaussian_kernel in gaussian_kernels[1:]:
#                 image = GaussianBlur(image, (0, 0), sigmaX=gaussian_kernel, sigmaY=gaussian_kernel)
#                 gaussian_images_in_octave.append(image)
#             gaussian_images.append(gaussian_images_in_octave)
#             octave_base = gaussian_images_in_octave[-3]
#             image = resize(octave_base, (int(octave_base.shape[1] / 2), int(octave_base.shape[0] / 2)), interpolation=INTER_NEAREST)
#         return array(gaussian_images, dtype=object)

#     def generateDoGImages(self,gaussian_images):
#         dog_images = []

#         for gaussian_images_in_octave in gaussian_images:
#             dog_images_in_octave = []
#             for first_image, second_image in zip(gaussian_images_in_octave, gaussian_images_in_octave[1:]):
#                 dog_images_in_octave.append(subtract(second_image, first_image))  # ordinary subtraction will not work because the images are unsigned integers
#             dog_images.append(dog_images_in_octave)
#         return array(dog_images, dtype=object)




#     def findScaleSpaceExtrema(self,gaussian_images, dog_images, num_intervals, sigma, image_border_width, contrast_threshold=0.04):

#         threshold = floor(0.5 * contrast_threshold / num_intervals * 255)  # from OpenCV implementation
#         keypoints = []

#         for octave_index, dog_images_in_octave in enumerate(dog_images):
#             for image_index, (first_image, second_image, third_image) in enumerate(zip(dog_images_in_octave, dog_images_in_octave[1:], dog_images_in_octave[2:])):
#                 # (i, j) is the center of the 3x3 array
#                 for i in range(image_border_width, first_image.shape[0] - image_border_width):
#                     for j in range(image_border_width, first_image.shape[1] - image_border_width):
#                         if self.isPixelAnExtremum(first_image[i-1:i+2, j-1:j+2], second_image[i-1:i+2, j-1:j+2], third_image[i-1:i+2, j-1:j+2], threshold):
#                             localization_result = self.localizeExtremumViaQuadraticFit(i, j, image_index + 1, octave_index, num_intervals, dog_images_in_octave, sigma, contrast_threshold, image_border_width)
#                             if localization_result is not None:
#                                 keypoint, localized_image_index = localization_result
#                                 keypoints_with_orientations = self.computeKeypointsWithOrientations(keypoint, octave_index, gaussian_images[octave_index][localized_image_index])
#                                 for keypoint_with_orientation in keypoints_with_orientations:
#                                     keypoints.append(keypoint_with_orientation)
#         return keypoints

#     def isPixelAnExtremum(self,first_subimage, second_subimage, third_subimage, threshold):
#         """Return True if the center element of the 3x3x3 input array is strictly greater than or less than all its neighbors, False otherwise
#         """
#         center_pixel_value = second_subimage[1, 1]
#         if abs(center_pixel_value) > threshold:
#             if center_pixel_value > 0:
#                 return all(center_pixel_value >= first_subimage) and \
#                     all(center_pixel_value >= third_subimage) and \
#                     all(center_pixel_value >= second_subimage[0, :]) and \
#                     all(center_pixel_value >= second_subimage[2, :]) and \
#                     center_pixel_value >= second_subimage[1, 0] and \
#                     center_pixel_value >= second_subimage[1, 2]
#             elif center_pixel_value < 0:
#                 return all(center_pixel_value <= first_subimage) and \
#                     all(center_pixel_value <= third_subimage) and \
#                     all(center_pixel_value <= second_subimage[0, :]) and \
#                     all(center_pixel_value <= second_subimage[2, :]) and \
#                     center_pixel_value <= second_subimage[1, 0] and \
#                     center_pixel_value <= second_subimage[1, 2]
#         return False

#     def localizeExtremumViaQuadraticFit(self,i, j, image_index, octave_index, num_intervals, dog_images_in_octave, sigma, contrast_threshold, image_border_width, eigenvalue_ratio=10, num_attempts_until_convergence=5):
      
#         extremum_is_outside_image = False
#         image_shape = dog_images_in_octave[0].shape
#         for attempt_index in range(num_attempts_until_convergence):
#             # need to convert from uint8 to float32 to compute derivatives and need to rescale pixel values to [0, 1] to apply Lowe's thresholds
#             first_image, second_image, third_image = dog_images_in_octave[image_index-1:image_index+2]
#             pixel_cube = stack([first_image[i-1:i+2, j-1:j+2],
#                                 second_image[i-1:i+2, j-1:j+2],
#                                 third_image[i-1:i+2, j-1:j+2]]).astype('float32') / 255.
#             gradient = self.computeGradientAtCenterPixel(pixel_cube)
#             hessian = self.computeHessianAtCenterPixel(pixel_cube)
#             extremum_update = -lstsq(hessian, gradient, rcond=None)[0]
#             if abs(extremum_update[0]) < 0.5 and abs(extremum_update[1]) < 0.5 and abs(extremum_update[2]) < 0.5:
#                 break
#             j += int(round(extremum_update[0]))
#             i += int(round(extremum_update[1]))
#             image_index += int(round(extremum_update[2]))
#             # make sure the new pixel_cube will lie entirely within the image
#             if i < image_border_width or i >= image_shape[0] - image_border_width or j < image_border_width or j >= image_shape[1] - image_border_width or image_index < 1 or image_index > num_intervals:
#                 extremum_is_outside_image = True
#                 break
#         if extremum_is_outside_image:
            
#             return None
#         if attempt_index >= num_attempts_until_convergence - 1:
#             return None
#         functionValueAtUpdatedExtremum = pixel_cube[1, 1, 1] + 0.5 * dot(gradient, extremum_update)
#         if abs(functionValueAtUpdatedExtremum) * num_intervals >= contrast_threshold:
#             xy_hessian = hessian[:2, :2]
#             xy_hessian_trace = trace(xy_hessian)
#             xy_hessian_det = det(xy_hessian)
#             if xy_hessian_det > 0 and eigenvalue_ratio * (xy_hessian_trace ** 2) < ((eigenvalue_ratio + 1) ** 2) * xy_hessian_det:
#                 # Contrast check passed -- construct and return OpenCV KeyPoint object
#                 keypoint = KeyPoint()
#                 keypoint.pt = ((j + extremum_update[0]) * (2 ** octave_index), (i + extremum_update[1]) * (2 ** octave_index))
#                 keypoint.octave = octave_index + image_index * (2 ** 8) + int(round((extremum_update[2] + 0.5) * 255)) * (2 ** 16)
#                 keypoint.size = sigma * (2 ** ((image_index + extremum_update[2]) / float32(num_intervals))) * (2 ** (octave_index + 1))  # octave_index + 1 because the input image was doubled
#                 keypoint.response = abs(functionValueAtUpdatedExtremum)
#                 return keypoint, image_index
#         return None

#     def computeGradientAtCenterPixel(self,pixel_array):
#         """Approximate gradient at center pixel [1, 1, 1] of 3x3x3 array using central difference formula of order O(h^2), where h is the step size
#         """
#         # With step size h, the central difference formula of order O(h^2) for f'(x) is (f(x + h) - f(x - h)) / (2 * h)
#         # Here h = 1, so the formula simplifies to f'(x) = (f(x + 1) - f(x - 1)) / 2
#         # NOTE: x corresponds to second array axis, y corresponds to first array axis, and s (scale) corresponds to third array axis
#         dx = 0.5 * (pixel_array[1, 1, 2] - pixel_array[1, 1, 0])
#         dy = 0.5 * (pixel_array[1, 2, 1] - pixel_array[1, 0, 1])
#         ds = 0.5 * (pixel_array[2, 1, 1] - pixel_array[0, 1, 1])
#         return array([dx, dy, ds])

#     def computeHessianAtCenterPixel(self,pixel_array):
#         """Approximate Hessian at center pixel [1, 1, 1] of 3x3x3 array using central difference formula of order O(h^2), where h is the step size
#         """
#         # With step size h, the central difference formula of order O(h^2) for f''(x) is (f(x + h) - 2 * f(x) + f(x - h)) / (h ^ 2)
#         # Here h = 1, so the formula simplifies to f''(x) = f(x + 1) - 2 * f(x) + f(x - 1)
#         # With step size h, the central difference formula of order O(h^2) for (d^2) f(x, y) / (dx dy) = (f(x + h, y + h) - f(x + h, y - h) - f(x - h, y + h) + f(x - h, y - h)) / (4 * h ^ 2)
#         # Here h = 1, so the formula simplifies to (d^2) f(x, y) / (dx dy) = (f(x + 1, y + 1) - f(x + 1, y - 1) - f(x - 1, y + 1) + f(x - 1, y - 1)) / 4
#         # NOTE: x corresponds to second array axis, y corresponds to first array axis, and s (scale) corresponds to third array axis
#         center_pixel_value = pixel_array[1, 1, 1]
#         dxx = pixel_array[1, 1, 2] - 2 * center_pixel_value + pixel_array[1, 1, 0]
#         dyy = pixel_array[1, 2, 1] - 2 * center_pixel_value + pixel_array[1, 0, 1]
#         dss = pixel_array[2, 1, 1] - 2 * center_pixel_value + pixel_array[0, 1, 1]
#         dxy = 0.25 * (pixel_array[1, 2, 2] - pixel_array[1, 2, 0] - pixel_array[1, 0, 2] + pixel_array[1, 0, 0])
#         dxs = 0.25 * (pixel_array[2, 1, 2] - pixel_array[2, 1, 0] - pixel_array[0, 1, 2] + pixel_array[0, 1, 0])
#         dys = 0.25 * (pixel_array[2, 2, 1] - pixel_array[2, 0, 1] - pixel_array[0, 2, 1] + pixel_array[0, 0, 1])
#         return array([[dxx, dxy, dxs], 
#                     [dxy, dyy, dys],
#                     [dxs, dys, dss]])

#     #########################
#     # Keypoint orientations #
#     #########################

#     def computeKeypointsWithOrientations(self,keypoint, octave_index, gaussian_image, radius_factor=3, num_bins=36, peak_ratio=0.8, scale_factor=1.5):

#         keypoints_with_orientations = []
#         image_shape = gaussian_image.shape

#         scale = scale_factor * keypoint.size / float32(2 ** (octave_index + 1))  # compare with keypoint.size computation in localizeExtremumViaQuadraticFit()
#         radius = int(round(radius_factor * scale))
#         weight_factor = -0.5 / (scale ** 2)
#         raw_histogram = zeros(num_bins)
#         smooth_histogram = zeros(num_bins)

#         for i in range(-radius, radius + 1):
#             region_y = int(round(keypoint.pt[1] / float32(2 ** octave_index))) + i
#             if region_y > 0 and region_y < image_shape[0] - 1:
#                 for j in range(-radius, radius + 1):
#                     region_x = int(round(keypoint.pt[0] / float32(2 ** octave_index))) + j
#                     if region_x > 0 and region_x < image_shape[1] - 1:
#                         dx = gaussian_image[region_y, region_x + 1] - gaussian_image[region_y, region_x - 1]
#                         dy = gaussian_image[region_y - 1, region_x] - gaussian_image[region_y + 1, region_x]
#                         gradient_magnitude = sqrt(dx * dx + dy * dy)
#                         gradient_orientation = rad2deg(arctan2(dy, dx))
#                         weight = exp(weight_factor * (i ** 2 + j ** 2))  # constant in front of exponential can be dropped because we will find peaks later
#                         histogram_index = int(round(gradient_orientation * num_bins / 360.))
#                         raw_histogram[histogram_index % num_bins] += weight * gradient_magnitude

#         for n in range(num_bins):
#             smooth_histogram[n] = (6 * raw_histogram[n] + 4 * (raw_histogram[n - 1] + raw_histogram[(n + 1) % num_bins]) + raw_histogram[n - 2] + raw_histogram[(n + 2) % num_bins]) / 16.
#         orientation_max = max(smooth_histogram)
#         orientation_peaks = where(logical_and(smooth_histogram > roll(smooth_histogram, 1), smooth_histogram > roll(smooth_histogram, -1)))[0]
#         for peak_index in orientation_peaks:
#             peak_value = smooth_histogram[peak_index]
#             if peak_value >= peak_ratio * orientation_max:
#                 # Quadratic peak interpolation
#                 # The interpolation update is given by equation (6.30) in https://ccrma.stanford.edu/~jos/sasp/Quadratic_Interpolation_Spectral_Peaks.html
#                 left_value = smooth_histogram[(peak_index - 1) % num_bins]
#                 right_value = smooth_histogram[(peak_index + 1) % num_bins]
#                 interpolated_peak_index = (peak_index + 0.5 * (left_value - right_value) / (left_value - 2 * peak_value + right_value)) % num_bins
#                 orientation = 360. - interpolated_peak_index * 360. / num_bins
#                 if abs(orientation - 360.) <  self.float_tolerance:
#                     orientation = 0
#                 new_keypoint = KeyPoint(*keypoint.pt, keypoint.size, orientation, keypoint.response, keypoint.octave)
#                 keypoints_with_orientations.append(new_keypoint)
#         return keypoints_with_orientations

#     ##############################
#     # Duplicate keypoint removal #
#     ##############################

#     def compareKeypoints(self,keypoint1, keypoint2):
#         """Return True if keypoint1 is less than keypoint2
#         """
#         if keypoint1.pt[0] != keypoint2.pt[0]:
#             return keypoint1.pt[0] - keypoint2.pt[0]
#         if keypoint1.pt[1] != keypoint2.pt[1]:
#             return keypoint1.pt[1] - keypoint2.pt[1]
#         if keypoint1.size != keypoint2.size:
#             return keypoint2.size - keypoint1.size
#         if keypoint1.angle != keypoint2.angle:
#             return keypoint1.angle - keypoint2.angle
#         if keypoint1.response != keypoint2.response:
#             return keypoint2.response - keypoint1.response
#         if keypoint1.octave != keypoint2.octave:
#             return keypoint2.octave - keypoint1.octave
#         return keypoint2.class_id - keypoint1.class_id

#     def removeDuplicateKeypoints(self,keypoints):
#         """Sort keypoints and remove duplicate keypoints
#         """
#         if len(keypoints) < 2:
#             return keypoints

#         keypoints.sort(key=cmp_to_key(self.compareKeypoints))
#         unique_keypoints = [keypoints[0]]

#         for next_keypoint in keypoints[1:]:
#             last_unique_keypoint = unique_keypoints[-1]
#             if last_unique_keypoint.pt[0] != next_keypoint.pt[0] or \
#             last_unique_keypoint.pt[1] != next_keypoint.pt[1] or \
#             last_unique_keypoint.size != next_keypoint.size or \
#             last_unique_keypoint.angle != next_keypoint.angle:
#                 unique_keypoints.append(next_keypoint)
#         return unique_keypoints

#     #############################
#     # Keypoint scale conversion #
#     #############################

#     def convertKeypointsToInputImageSize(self,keypoints):
#         """Convert keypoint point, size, and octave to input image size
#         """
#         converted_keypoints = []
#         for keypoint in keypoints:
#             keypoint.pt = tuple(0.5 * array(keypoint.pt))
#             keypoint.size *= 0.5
#             keypoint.octave = (keypoint.octave & ~255) | ((keypoint.octave - 1) & 255)
#             converted_keypoints.append(keypoint)
#         return converted_keypoints

#     #########################
#     # Descriptor generation #
#     #########################

#     def unpackOctave(self,keypoint):
#         """Compute octave, layer, and scale from a keypoint
#         """
#         octave = keypoint.octave & 255
#         layer = (keypoint.octave >> 8) & 255
#         if octave >= 128:
#             octave = octave | -128
#         scale = 1 / float32(1 << octave) if octave >= 0 else float32(1 << -octave)
#         return octave, layer, scale

#     def generateDescriptors(self,keypoints, gaussian_images, window_width=4, num_bins=8, scale_multiplier=3, descriptor_max_value=0.2):
       
#         descriptors = []

#         for keypoint in keypoints:
#             octave, layer, scale = self.unpackOctave(keypoint)
#             gaussian_image = gaussian_images[octave + 1, layer]
#             num_rows, num_cols = gaussian_image.shape
#             point = round(scale * array(keypoint.pt)).astype('int')
#             bins_per_degree = num_bins / 360.
#             angle = 360. - keypoint.angle
#             cos_angle = cos(deg2rad(angle))
#             sin_angle = sin(deg2rad(angle))
#             weight_multiplier = -0.5 / ((0.5 * window_width) ** 2)
#             row_bin_list = []
#             col_bin_list = []
#             magnitude_list = []
#             orientation_bin_list = []
#             histogram_tensor = zeros((window_width + 2, window_width + 2, num_bins))   # first two dimensions are increased by 2 to account for border effects

#             # Descriptor window size (described by half_width) follows OpenCV convention
#             hist_width = scale_multiplier * 0.5 * scale * keypoint.size
#             half_width = int(round(hist_width * sqrt(2) * (window_width + 1) * 0.5))   # sqrt(2) corresponds to diagonal length of a pixel
#             half_width = int(min(half_width, sqrt(num_rows ** 2 + num_cols ** 2)))     # ensure half_width lies within image

#             for row in range(-half_width, half_width + 1):
#                 for col in range(-half_width, half_width + 1):
#                     row_rot = col * sin_angle + row * cos_angle
#                     col_rot = col * cos_angle - row * sin_angle
#                     row_bin = (row_rot / hist_width) + 0.5 * window_width - 0.5
#                     col_bin = (col_rot / hist_width) + 0.5 * window_width - 0.5
#                     if row_bin > -1 and row_bin < window_width and col_bin > -1 and col_bin < window_width:
#                         window_row = int(round(point[1] + row))
#                         window_col = int(round(point[0] + col))
#                         if window_row > 0 and window_row < num_rows - 1 and window_col > 0 and window_col < num_cols - 1:
#                             dx = gaussian_image[window_row, window_col + 1] - gaussian_image[window_row, window_col - 1]
#                             dy = gaussian_image[window_row - 1, window_col] - gaussian_image[window_row + 1, window_col]
#                             gradient_magnitude = sqrt(dx * dx + dy * dy)
#                             gradient_orientation = rad2deg(arctan2(dy, dx)) % 360
#                             weight = exp(weight_multiplier * ((row_rot / hist_width) ** 2 + (col_rot / hist_width) ** 2))
#                             row_bin_list.append(row_bin)
#                             col_bin_list.append(col_bin)
#                             magnitude_list.append(weight * gradient_magnitude)
#                             orientation_bin_list.append((gradient_orientation - angle) * bins_per_degree)

#             for row_bin, col_bin, magnitude, orientation_bin in zip(row_bin_list, col_bin_list, magnitude_list, orientation_bin_list):
#                 # Smoothing via trilinear interpolation
#                 # Notations follows https://en.wikipedia.org/wiki/Trilinear_interpolation
#                 # Note that we are really doing the inverse of trilinear interpolation here (we take the center value of the cube and distribute it among its eight neighbors)
#                 row_bin_floor, col_bin_floor, orientation_bin_floor = floor([row_bin, col_bin, orientation_bin]).astype(int)
#                 row_fraction, col_fraction, orientation_fraction = row_bin - row_bin_floor, col_bin - col_bin_floor, orientation_bin - orientation_bin_floor
#                 if orientation_bin_floor < 0:
#                     orientation_bin_floor += num_bins
#                 if orientation_bin_floor >= num_bins:
#                     orientation_bin_floor -= num_bins

#                 c1 = magnitude * row_fraction
#                 c0 = magnitude * (1 - row_fraction)
#                 c11 = c1 * col_fraction
#                 c10 = c1 * (1 - col_fraction)
#                 c01 = c0 * col_fraction
#                 c00 = c0 * (1 - col_fraction)
#                 c111 = c11 * orientation_fraction
#                 c110 = c11 * (1 - orientation_fraction)
#                 c101 = c10 * orientation_fraction
#                 c100 = c10 * (1 - orientation_fraction)
#                 c011 = c01 * orientation_fraction
#                 c010 = c01 * (1 - orientation_fraction)
#                 c001 = c00 * orientation_fraction
#                 c000 = c00 * (1 - orientation_fraction)

#                 histogram_tensor[row_bin_floor + 1, col_bin_floor + 1, orientation_bin_floor] += c000
#                 histogram_tensor[row_bin_floor + 1, col_bin_floor + 1, (orientation_bin_floor + 1) % num_bins] += c001
#                 histogram_tensor[row_bin_floor + 1, col_bin_floor + 2, orientation_bin_floor] += c010
#                 histogram_tensor[row_bin_floor + 1, col_bin_floor + 2, (orientation_bin_floor + 1) % num_bins] += c011
#                 histogram_tensor[row_bin_floor + 2, col_bin_floor + 1, orientation_bin_floor] += c100
#                 histogram_tensor[row_bin_floor + 2, col_bin_floor + 1, (orientation_bin_floor + 1) % num_bins] += c101
#                 histogram_tensor[row_bin_floor + 2, col_bin_floor + 2, orientation_bin_floor] += c110
#                 histogram_tensor[row_bin_floor + 2, col_bin_floor + 2, (orientation_bin_floor + 1) % num_bins] += c111

#             descriptor_vector = histogram_tensor[1:-1, 1:-1, :].flatten()  # Remove histogram borders
#             # Threshold and normalize descriptor_vector
#             threshold = norm(descriptor_vector) * descriptor_max_value
#             descriptor_vector[descriptor_vector > threshold] = threshold
#             descriptor_vector /= max(norm(descriptor_vector),  self.float_tolerance)
#             # Multiply by 512, round, and saturate between 0 and 255 to convert from float32 to unsigned char (OpenCV convention)
#             descriptor_vector = round(512 * descriptor_vector)
#             descriptor_vector[descriptor_vector < 0] = 0
#             descriptor_vector[descriptor_vector > 255] = 255
#             descriptors.append(descriptor_vector)
#         return array(descriptors, dtype='float32')
#     def match_keypoints(self, descriptors1, descriptors2):
#         matches = []
#         for i, desc1 in enumerate(descriptors1):
#             distances = np.linalg.norm(descriptors2 - desc1, axis=1)
#             best_match = np.argmin(distances)
#             matches.append((i, best_match, distances[best_match]))
#         random.shuffle(matches)
#         matches = sorted(matches, key=lambda x: x[2])
#         return matches

#     def draw_matches(self, image1, image2, keypoints1, keypoints2, matches, num_matches_to_display=25,circle_radius=5):
#         from PIL import ImageDraw, Image
#         img1 = Image.fromarray(image1).convert("RGB")
#         img2 = Image.fromarray(image2).convert("RGB")

#         new_image = Image.new("RGB", (img1.width + img2.width, max(img1.height, img2.height)))
#         new_image.paste(img1, (0, 0))
#         new_image.paste(img2, (img1.width, 0))

#         draw = ImageDraw.Draw(new_image)

#         print(f"Total matches found: {len(matches)}")

#         best_matches = matches[:num_matches_to_display]
#         for i1, i2, _ in best_matches:
#             x1, y1 = keypoints1[i1].pt
#             x2, y2 = keypoints2[i2].pt

#             draw.ellipse([(x1 - circle_radius, y1 - circle_radius), (x1 + circle_radius, y1 + circle_radius)], outline=(0, 255, 0), width=2)
#             x2 += img1.width  # Shift x2 to the second image space
#             draw.ellipse([(x2 - circle_radius, y2 - circle_radius), (x2 + circle_radius, y2 + circle_radius)], outline=(0, 255, 0), width=2)

#             draw.line([(x1, y1), (x2, y2)], fill=(255, 0, 0), width=2)
#         return np.array(new_image)