import numpy as np

def get_gaussian_kernel(kernel_size):
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
