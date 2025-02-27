import numpy as np
def calc_gaussian_kernel( kernel_size):
        if kernel_size == 3:
            sigma = 1
        elif kernel_size == 5:
            sigma = 1.4
        elif kernel_size == 7:
            sigma = 1.6        

        kernel = np.zeros(shape=(kernel_size, kernel_size))
        center = kernel_size // 2
        for x in range(kernel_size):
            for y in range(kernel_size):
                x_shifted = x - center
                y_shifted = y - center
                kernel[x][y] = (1 / (2 * np.pi * sigma**2)) * np.exp(-(x_shifted**2 + y_shifted**2) / (2 * sigma**2))
        
        kernel /= np.sum(kernel)
        return kernel        
print(calc_gaussian_kernel(3))                

import numpy as np

# def calc_gaussian_kernel(kernel_size):
#     if kernel_size == 3:
#         sigma = 1
#     elif kernel_size == 5:
#         sigma = 1.4
#     elif kernel_size == 7:
#         sigma = 1.6  
#     else:
#         raise ValueError("Kernel size must be 3, 5, or 7")

#     kernel = np.zeros((kernel_size, kernel_size))  # Initialize a 2D array
#     center = kernel_size // 2  # Center of the kernel

#     for x in range(kernel_size):
#         for y in range(kernel_size):
#             x_shifted = x - center
#             y_shifted = y - center
#             kernel[x, y] = (1 / (2 * np.pi * sigma ** 2)) * np.exp(-(x_shifted ** 2 + y_shifted ** 2) / (2 * sigma ** 2))
    
#     kernel /= np.sum(kernel)  # Normalize so the sum of all elements is 1
#     return kernel

# Test the function
print(calc_gaussian_kernel(3))
# print(calc_gaussian_kernel(5))
# print(calc_gaussian_kernel(7))
