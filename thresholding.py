import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imsave
import cv2

class ThresholdingProcessor:
    def __init__(self, image = None):
        self.image = image
        
        
    def otsu_threshold(self, image):
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            # colored_full_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        pixel_number = gray.shape[0] * gray.shape[1] # number of pixels
        mean_weight = 1.0/pixel_number # sum of all weights
        his, bins = np.histogram(gray, np.arange(0,257)) # calculating the histogram of the image
        final_thresh = -1 # defining the best threshold calculated
        final_variance = -1 # defining the highest between class variance
        intensity_arr = np.arange(256) # creating array of all the possible pixel values (0-255)
        # Iterating through all the possible pixel values from the histogram as thresholds
        for t in bins[0:-1]:
            pcb = np.sum(his[:t]) # summing the frequency of the values before the threshold (background)
            pcf = np.sum(his[t:]) # summing the frequency of the values after the threshold (foreground)
            Wb = pcb * mean_weight # calculating the weight of the background (divide the frequencies by the sum of all weights)
            Wf = pcf * mean_weight # calculating the weight of the foreground

            mub = np.sum(intensity_arr[:t]*his[:t]) / float(pcb) # calculating the mean of the background (multiply the background 
            # pixel value with its weight, then divide it with the sum of frequencies of the background)
            muf = np.sum(intensity_arr[t:]*his[t:]) / float(pcf) # calculating the mean of the foreground
            
            variance = Wb * Wf * (mub - muf) ** 2 # calculate the between class variance

            if variance > final_variance: # compare the variance in each step with the previous
                final_thresh = t
                final_variance = variance

        return final_thresh


    def local_thresholding(self, image, t1, t2, t3, t4, mode):
        # If the image is colored, change it to grayscale, otherwise take the image as it is
        if (image.ndim == 3):
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif (image.ndim == 2):
            gray = image

        height, width = gray.shape # get the height and width of the image
        # In this case we will divide the image into a 2x2 grid image
        half_height = height//2 
        half_width = width//2

        # Getting the four section of the 2x2 image
        section_1 = gray[:half_height, :half_width]
        section_2 = gray[:half_height, half_width:]
        section_3 = gray[half_height:, :half_width]
        section_4 = gray[half_height:, half_width:]

        # Check if the threshold is calculated through Otsu's method or given by the user
        if (mode == 1): # calculating the threshold using Otsu's methond for each section
            t1 = self.otsu_threshold(section_1)
            t2 = self.otsu_threshold(section_2)
            t3 = self.otsu_threshold(section_3)
            t4 = self.otsu_threshold(section_4)

        # Applying the threshold of each section on its corresponding section
        section_1[section_1 > t1] = 255
        section_1[section_1 < t1] = 0

        section_2[section_2 > t2] = 255
        section_2[section_2 < t2] = 0

        section_3[section_3 > t3] = 255
        section_3[section_3 < t3] = 0

        section_4[section_4 > t4] = 255
        section_4[section_4 < t4] = 0

        # Regroup the sections to form the final image
        top_section = np.concatenate((section_1, section_2), axis = 1)
        bottom_section = np.concatenate((section_3, section_4), axis = 1)
        final_img = np.concatenate((top_section, bottom_section), axis=0)

            # final_img = gray.copy()
            # final_img[gray > t] = 255
            # final_img[gray < t] = 0

        
        plt.imshow(final_img, cmap = 'gray')
        path = "images/output/local.png"
        plt.axis("off")
        plt.savefig(path)
        return path
        # return final_img


    def global_thresholding(self, image, t, mode):
        # If the image is colored, change it to grayscale, otherwise take the image as it is
        if (image.ndim == 3):
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif (image.ndim == 2):
            gray = image

        # Check if the threshold is calculated through Otsu's method or the threshold is given by the user
        if (mode == 1): # calculating the threshold using Otsu's methond for the whole image
            t = self.otsu_threshold(gray)

        # Applying the threshold on the image whether it is calculated or given by the user according to the previous condition
        final_img = gray.copy()
        final_img[gray > t] = 255
        final_img[gray < t] = 0

        plt.imshow(final_img, cmap = 'gray')
        path = "images/output/global.png"
        plt.axis("off")
        plt.savefig(path)
        return path

# # image_path = 'data\dog.jpg'
# # image = cv2.imread(image_path)            

# # image = cv2.cvtColor(image_path, cv2.COLOR_BGR2RGB)

# # image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
# # otsu_threshold(image)

# import os
# import cv2
# import matplotlib.pyplot as plt

# # Import the functions you wrote
# # from your_script import local_thresholding, global_thresholding

# # Load an example image
# # Make sure you have an 'images/input/' folder and a grayscale or RGB image there
# input_image_path = "data\dog.jpg"  # Adjust path and name if different
# import os
# import cv2
# import matplotlib.pyplot as plt
# import numpy as np

# # Import your functions
# # from your_script import local_thresholding, global_thresholding, otsu_threshold

# # Load an example image

# # Create output directory if it doesn't exist
# os.makedirs("images/output", exist_ok=True)

# # Read the image
# image = cv2.imread(input_image_path)

# if image is None:
#     raise ValueError(f"Failed to load image from {input_image_path}. Check the path!")

# # Show the original image
# plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
# plt.title("Original Image")
# plt.axis("off")
# plt.show()

# ### Test 1: Test Otsu Thresholding alone
# print("Testing Otsu Thresholding alone...")
# if image.ndim == 3:
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# else:
#     gray = image.copy()

# # Compute threshold manually
# otsu_thresh_value = otsu_threshold(gray)
# print(f"Otsu's calculated threshold: {otsu_thresh_value}")

# # Apply the calculated threshold manually
# otsu_img = gray.copy()
# otsu_img[gray > otsu_thresh_value] = 255
# otsu_img[gray < otsu_thresh_value] = 0

# # Save and display
# otsu_path = "images/output/otsu_manual.png"
# cv2.imwrite(otsu_path, otsu_img)

# plt.imshow(otsu_img, cmap='gray')
# plt.title(f"Otsu Thresholding (threshold={otsu_thresh_value})")
# plt.axis("off")
# plt.show()


# ### Test 2: Global Thresholding using your function
# global_path = global_thresholding(image, t=0, mode=1)
# print(f"Global thresholded image saved at: {global_path}")

# ### Test 3: Local Thresholding using your function
# local_path = local_thresholding(image, t1=0, t2=0, t3=0, t4=0, mode=1)
# print(f"Local thresholded image saved at: {local_path}")

# # Optional: Display Global and Local Thresholding Results
# global_result = cv2.imread(global_path, cv2.IMREAD_GRAYSCALE)
# local_result = cv2.imread(local_path, cv2.IMREAD_GRAYSCALE)

# plt.figure(figsize=(12,5))

# plt.subplot(1,2,1)
# plt.imshow(global_result, cmap='gray')
# plt.title("Global Thresholding")
# plt.axis("off")

# plt.subplot(1,2,2)
# plt.imshow(local_result, cmap='gray')
# plt.title("Local Thresholding")
# plt.axis("off")

# plt.show()
