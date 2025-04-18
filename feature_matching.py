import cv2
import numpy as np

class FeatureMatching:    
    def apply_ssd_matching(image, template):
        # convert to gray scale     
        if len(image.shape) == 3:
            gray_full_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            colored_full_image = image.copy()
        else:
            gray_full_image = image
            colored_full_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            
        if len(template.shape) == 3:
            gray_image_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        else:
            gray_image_template = template
        
        full_full_image_height, full_full_image_width = gray_full_image.shape
        template_height, template_width = gray_image_template.shape
        
        if template_height > full_full_image_height or template_width > full_full_image_width:
            raise ValueError("Template must be smaller than the image")
        
        best_ssd = float('inf')
        best_location = (0, 0)

        step = max(1, min(template_width, template_height) // 20)
        
        for y in range(0, full_full_image_height - template_height + 1, step):
            for x in range(0, full_full_image_width - template_width + 1, step):
                patch = gray_full_image[y:y+template_height, x:x+template_width]
                difference = gray_image_template.astype(np.float32) - patch.astype(np.float32)
                ssd = np.sum(np.square(difference))
                
                if ssd < best_ssd:
                    best_ssd = ssd
                    best_location = (x, y)
        
        x, y = best_location
        cv2.rectangle(colored_full_image, (x, y), (x + template_width, y + template_height), (0, 0, 255), 3)
        
        return colored_full_image
    
    
    def apply_ncc_matching(image, template):
        # convert to gray scale
        if len(image.shape) == 3:
            gray_full_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            colored_full_image = image.copy()
        else:
            gray_full_image = image
            colored_full_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        if len(template.shape) == 3:
            gray_image_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        else:
            gray_image_template = template

        gray_full_image = gray_full_image.astype(np.float32)
        gray_image_template = gray_image_template.astype(np.float32)

        full_image_height, full_image_width = gray_full_image.shape
        template_height, template_width = gray_image_template.shape

        best_ncc = -1  # NCC values range from -1 to 1
        best_location = (0, 0)

        template_mean = np.mean(gray_image_template)
        template_std = np.std(gray_image_template)
        template_norm = (gray_image_template - template_mean) / template_std 

        step = max(1, min(template_width, template_height) // 20)
        
        for y in range(0, full_image_height - template_height + 1, step):
            for x in range(0, full_image_width - template_width + 1, step):
                patch = gray_full_image[y:y+template_height, x:x+template_width]
                patch_mean = np.mean(patch)
                patch_std = np.std(patch)
                patch_norm = (patch - patch_mean) / patch_std

                ncc = np.mean(patch_norm * template_norm)

                if ncc > best_ncc:
                    best_ncc = ncc
                    best_location = (x, y)


        x, y = best_location
        cv2.rectangle(colored_full_image, (x, y), (x + template_width, y + template_height), (255, 0, 0), 3)

        return colored_full_image
