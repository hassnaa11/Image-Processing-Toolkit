import numpy as np

class ThresholdingProcessor:
    # Global methods
    def optimal_threshold_global(self, image, max_iter=100, tol=1):
        """
        Optimal thresholding using iterative method (global)
        """
        # Initialize threshold as the mean of min and max pixel values
        threshold = (np.min(image) + np.max(image)) / 2
        prev_threshold = 0
        
        for _ in range(max_iter):
            # Segment the image
            foreground = image > threshold
            background = image <= threshold
            
            # Calculate means
            mean_fore = np.mean(image[foreground]) if np.any(foreground) else threshold
            mean_back = np.mean(image[background]) if np.any(background) else threshold
            
            # Update threshold
            prev_threshold = threshold
            threshold = (mean_fore + mean_back) / 2
            
            # Check for convergence
            if abs(threshold - prev_threshold) < tol:
                break
        
        # Apply threshold
        binary = np.zeros_like(image)
        binary[image > threshold] = 255
        return binary
        
    def otsu_threshold_global(self, image):
        """
        Otsu's thresholding method (global)
        """
        # Calculate histogram
        hist, bins = np.histogram(image.flatten(), bins=256, range=[0,256])
        hist = hist.astype(float) / hist.sum()  # Normalize
        
        # Initialize variables
        best_thresh = 0
        best_var = 0
        
        # Iterate through all possible thresholds
        for threshold in range(1, 256):
            # Class probabilities
            w0 = np.sum(hist[:threshold])
            w1 = np.sum(hist[threshold:])
            
            if w0 == 0 or w1 == 0:
                continue
            
            # Class means
            mean0 = np.sum(np.arange(threshold) * hist[:threshold]) / w0
            mean1 = np.sum(np.arange(threshold, 256) * hist[threshold:]) / w1
            
            # Between-class variance
            var = w0 * w1 * (mean0 - mean1) ** 2
            
            if var > best_var:
                best_var = var
                best_thresh = threshold
        
        # Apply threshold
        binary = np.zeros_like(image)
        binary[image > best_thresh] = 255
        return binary
    
    def spectral_threshold_global(self, image, n_classes=3):
        """
        Spectral thresholding using multi-level Otsu (global)
        """
        # Calculate histogram
        hist, bins = np.histogram(image.flatten(), bins=256, range=[0,256])
        hist = hist.astype(float) / hist.sum()  # Normalize
        
        # Initialize best thresholds and variance
        best_thresholds = []
        best_var = 0
        
        # Try all possible threshold combinations (simplified version)
        # Note: For more classes, a more efficient algorithm would be needed
        if n_classes == 3:
            for t1 in range(1, 254):
                for t2 in range(t1+1, 255):
                    # Class probabilities
                    w0 = np.sum(hist[:t1])
                    w1 = np.sum(hist[t1:t2])
                    w2 = np.sum(hist[t2:])
                    
                    if w0 == 0 or w1 == 0 or w2 == 0:
                        continue
                    
                    # Class means
                    mean0 = np.sum(np.arange(t1) * hist[:t1]) / w0
                    mean1 = np.sum(np.arange(t1, t2) * hist[t1:t2]) / w1
                    mean2 = np.sum(np.arange(t2, 256) * hist[t2:]) / w2
                    
                    # Total mean
                    mean_total = mean0 * w0 + mean1 * w1 + mean2 * w2
                    
                    # Between-class variance
                    var = (w0 * (mean0 - mean_total)**2 + 
                        w1 * (mean1 - mean_total)**2 + 
                        w2 * (mean2 - mean_total)**2)
                    
                    if var > best_var:
                        best_var = var
                        best_thresholds = [t1, t2]
        
        # Apply thresholds
        segmented = np.zeros_like(image)
        if n_classes == 3:
            segmented[image <= best_thresholds[0]] = 0
            segmented[(image > best_thresholds[0]) & (image <= best_thresholds[1])] = 128
            segmented[image > best_thresholds[1]] = 255
        else:
            # For simplicity, we'll just do binary if n_classes != 3
            threshold = best_thresholds[0] if best_thresholds else 128
            segmented[image > threshold] = 255
        
        return segmented
    
    # Local methods
    def optimal_threshold_local(self, image, block_size=100, C=5):
        """
        Local optimal thresholding using sliding window
        """
        height, width = image.shape
        binary = np.zeros_like(image)
        
        for y in range(0, height, block_size//2):
            for x in range(0, width, block_size//2):
                # Get block
                y1 = min(y + block_size, height)
                x1 = min(x + block_size, width)
                block = image[y:y1, x:x1]
                
                if block.size == 0:
                    continue
                    
                # Apply optimal threshold to this block
                block_binary = self.optimal_threshold_global(block)
                binary[y:y1, x:x1] = block_binary
        
        return binary
    
    def otsu_threshold_local(self, image, block_size=100):
        """
        Local Otsu's thresholding using sliding window
        """
        height, width = image.shape
        binary = np.zeros_like(image)
        
        for y in range(0, height, block_size//2):
            for x in range(0, width, block_size//2):
                # Get block
                y1 = min(y + block_size, height)
                x1 = min(x + block_size, width)
                block = image[y:y1, x:x1]
                
                if block.size == 0:
                    continue
                    
                # Apply Otsu to this block
                block_binary = self.otsu_threshold_global(block)
                binary[y:y1, x:x1] = block_binary
        
        return binary
        
    def spectral_threshold_local(self, image, block_size=200, n_classes=3):
        """
        Local spectral thresholding using sliding window
        """
        height, width = image.shape
        segmented = np.zeros_like(image)
        
        for y in range(0, height, block_size//2):
            for x in range(0, width, block_size//2):
                # Get block
                y1 = min(y + block_size, height)
                x1 = min(x + block_size, width)
                block = image[y:y1, x:x1]
                
                if block.size == 0:
                    continue
                    
                # Apply spectral threshold to this block
                block_segmented = self.spectral_threshold_global(block, n_classes)
                segmented[y:y1, x:x1] = block_segmented
        
        return segmented
    
    # Unified method to call all types
    def apply_threshold(self, image, method='otsu', scope='Global', **kwargs):
        """
        Apply thresholding with specified method and scope
        
        Parameters:
        - image: Input grayscale image
        - method: 'optimal', 'otsu', or 'spectral'
        - scope: 'global' or 'local'
        - kwargs: Additional parameters for the methods
        
        Returns:
        - Thresholded image
        """
        if scope == 'Global':
            if method == 'optimal':
                return self.optimal_threshold_global(image, **kwargs)
            elif method == 'otsu':
                return self.otsu_threshold_global(image, **kwargs)
            elif method == 'spectral':
                return self.spectral_threshold_global(image, **kwargs)
        elif scope == 'Local':
            if method == 'optimal':
                return self.optimal_threshold_local(image, **kwargs)
            elif method == 'otsu':
                return self.otsu_threshold_local(image, **kwargs)
            elif method == 'spectral':
                return self.spectral_threshold_local(image, **kwargs)
        else:
            raise ValueError(f"Unknown method/scope: {method}/{scope}")