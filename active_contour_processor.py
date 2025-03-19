import numpy as np 
from image_processor import edge_detection, FilterProcessor
import cv2

class ActiveContourProcessor:
    def __init__(self, image, alpha=0.1, beta=0.3, gamma=0.1, window_size=5, iterations=400, sigma=1):
        # smooth the image
        # image = cv2.GaussianBlur(image, (5, 5), sigmaX=3)
        filter_processor = FilterProcessor(image)
        image = filter_processor.apply_filter(3, 'Gaussian', sigma)
        # plt.imshow(image, cmap="gray")
        # plt.show() 
        
        # image = cv2.equalizeHist(image)
        # plt.imshow(image, cmap="gray")
        # plt.show() 
        self.image = image
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.window_size = window_size
        self.iterations = iterations
        
        self.inint_snake = self.initialize_snake()
        self.snake = self.inint_snake.copy()
        
        edge_detection_processor = edge_detection(image)
        edge_detection_processor.apply_edge_detection_filter("Canny")
        edges = edge_detection_processor.gradient
        # Normalize edge values to [0, 255]
        edges_normalized = cv2.normalize(edges.astype(np.float64), None, 0, 255, cv2.NORM_MINMAX)
        # Increase contrast for edges only
        edge_enhanced = np.clip(edges_normalized * 1.5, 0, 255)
        # Convert to float64 for safe computations
        self.gradient = edge_enhanced.astype(np.float64)
        # Display result
        # plt.imshow(self.gradient, cmap="gray")
        # plt.show()
        
        
        
        # # Sobel X (detects vertical edges)
        # sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)

        # # Sobel Y (detects horizontal edges)
        # sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)

        # # Compute the gradient magnitude
        # sobel_combined = cv2.magnitude(sobel_x, sobel_y)

        # # Normalize gradient to avoid overflow issues
        # sobel_combined = cv2.normalize(sobel_combined, None, 0, 255, cv2.NORM_MINMAX)
        
        # # Increase contrast of edges only
        # edge_enhanced = np.clip(sobel_combined * 2, 0, 255)

        # # Convert to float64 for safe computations
        # self.gradient = edge_enhanced.astype(np.float64)
        # self.gradient = cv2.normalize(self.gradient, None, 0, 1, cv2.NORM_MINMAX)
        # plt.imshow(edges)
        # plt.show() 
        
        # # mask_high = image > 130
        # # self.gradient[mask_high] = np.clip(self.gradient[mask_high] * 4, 0, 255)
        # # mask_low = image <= 130
        # # self.gradient[mask_low] = np.clip(self.gradient[mask_low] * 0.1, 0, 255)

        
        
        # edges = np.sqrt(sobel_x**2 + sobel_y**2)
        # edges = np.uint8(255 * edges / np.max(edges))  # Normalize edges to 0-255

        # # Create a contrast-enhanced image
        # contrast_image = np.clip(image * 1.5, 0, 255).astype(np.uint8)

        # # Apply contrast only to edge regions
        # mask = edges > 50  # Threshold to select strong edges
        # self.gradient = image.copy()
        # self.gradient[mask] = contrast_image[mask]  # Apply enhancement only at edges
        # enhanced_edges = np.clip(edges * 1.5, 0, 255).astype(np.uint8)  # Multiply by contrast factor
        # plt.imshow(self.gradient, cmap="gray")
        # plt.show() 
        
        
        # Apply Canny Edge Detection
        # edges = cv2.Canny(image, threshold1=50, threshold2=150)

        # # Normalize edge values to [0, 255]
        # edges_normalized = cv2.normalize(edges.astype(np.float64), None, 0, 255, cv2.NORM_MINMAX)

        # # Increase contrast for edges only
        # edge_enhanced = np.clip(edges_normalized * 1.5, 0, 255)

        # # Convert to float64 for safe computations
        # self.gradient = edge_enhanced.astype(np.float64)

        # # Display result
        # plt.imshow(self.gradient, cmap="gray")
        # plt.show()

        
            
    def initialize_snake(self):
        height, width = self.image.shape[:2]
        print(width, height)
        center_x, center_y = width // 2 , height // 2 
        
        radius_x, radius_y = width // 2, height // 2
        print(radius_x, radius_y)
        s = np.linspace(0, 2 * np.pi, 100)
        x = center_x + radius_x * np.cos(s)
        y = center_y + radius_y * np.sin(s)
        snake = np.array([x, y]).T
        # plt.imshow(self.image, cmap="gray")
        # plt.plot(snake[:, 0], snake[:, 1], color="b") 
        # plt.show() 
        return snake
        
        
    def get_energy(self, point, previous_point, next_point):
        x, y = point
        x_next, y_next = next_point
        x_prev, y_prev = previous_point
        # External Energy (image energy)
        image_energy = -self.gradient[int(y), int(x)]
        # internal_energy (elasticity)
        # elasticity = self.alpha * np.linalg.norm(previous_point - point)
        elasticity = self.alpha * np.sqrt(((x_next - x) ** 2 + (y_next - y) ** 2))
        # internal_energy (smoothness) 
        # smoothness = self.beta * np.linalg.norm(previous_point - 2 * point + next_point)   
        smoothness = self.beta * np.sqrt(((x_next - 2 * x + x_prev) ** 2 + (y_next - 2 * y + y_prev) ** 2))
        
        return image_energy + elasticity + smoothness
    
    
    def update_snake(self):
        for _ in range(self.iterations):
            new_snake = np.copy(self.snake)
            
            for i in range(len(self.snake)):
                previous_point = self.snake[i - 1]
                next_point = self.snake[(i + 1) % len(self.snake)]
                
                best_point = self.snake[i]
                min_energy = float('inf')
                
                for dx in range(-self.window_size, self.window_size + 1):
                    for dy in range(-self.window_size, self.window_size + 1):
                        candidate = self.snake[i] + np.array([dx, dy])
                        if 0 <= candidate[0] < self.image.shape[1] and 0 <= candidate[1] < self.image.shape[0]:
                            energy = self.get_energy(candidate, previous_point, next_point)
                            if energy < min_energy:
                                min_energy = energy
                                best_point = candidate
                                
                new_snake[i] = best_point
                
            self.snake = new_snake
            
            
    def get_snake(self):
        return self.snake, self.inint_snake
    
    
    
import sys
import numpy as np
import cv2
from PyQt5.QtWidgets import QApplication, QGraphicsScene, QGraphicsView, QGraphicsPixmapItem, QGraphicsPathItem
from PyQt5.QtGui import QImage, QPixmap, QPen, QPainterPath
from PyQt5.QtCore import Qt
import matplotlib.pyplot as plt
if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Load and process image
    image = cv2.imread(r"data\apple2.png", cv2.IMREAD_GRAYSCALE)
    print("Grayscale image read")

    snake_model = ActiveContourProcessor(image, alpha=0.05, beta=0.1, window_size=5)
    snake_model.update_snake()
    final_snake, init_snake, image_final = snake_model.get_snake()

    # Ensure image is contiguous in memory
    image_final = np.ascontiguousarray(image_final)

    # Convert OpenCV image to QImage
    height, width = image_final.shape
    bytes_per_line = width
    q_image = QImage(image_final.data, width, height, bytes_per_line, QImage.Format_Grayscale8)

    # Convert QImage to QPixmap
    pixmap = QPixmap.fromImage(q_image)

    # Create scene and add image
    scene = QGraphicsScene()
    pixmap_item = QGraphicsPixmapItem(pixmap)
    scene.addItem(pixmap_item)

    # Draw initial snake (Blue)
    path_init = QPainterPath()
    path_init.moveTo(init_snake[0, 0], init_snake[0, 1])
    for point in init_snake[1:]:
        path_init.lineTo(point[0], point[1])
    
    init_snake_item = QGraphicsPathItem(path_init)
    init_snake_item.setPen(QPen(Qt.blue, 2))
    scene.addItem(init_snake_item)

    # Draw final snake (Red)
    path_final = QPainterPath()
    path_final.moveTo(final_snake[0, 0], final_snake[0, 1])
    for point in final_snake[1:]:
        path_final.lineTo(point[0], point[1])
    
    final_snake_item = QGraphicsPathItem(path_final)
    final_snake_item.setPen(QPen(Qt.red, 2))
    scene.addItem(final_snake_item)

    # Create view
    view = QGraphicsView()
    view.setScene(scene)
    view.show()

    sys.exit(app.exec_())
    
# import matplotlib.pyplot as plt
# from PyQt5.QtWidgets import QApplication, QGraphicsScene, QGraphicsView, QGraphicsPixmapItem
# from PyQt5.QtGui import QImage, QPixmap
# import sys
# import cv2    
# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     image = cv2.imread(r"data\dog.jpg", cv2.IMREAD_GRAYSCALE)
#     # if len(image.shape) == 3 and image.shape[2] == 3:
#     #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     #     print("RGB image read")
#     # else:
#     # image = cv2.imread(r"data\apple3.png", cv2.IMREAD_GRAYSCALE)
#     print("Grayscale image read")    

#     snake_model = ActiveContourProcessor(image, alpha=0.05, beta=0.1, window_size=5)
    
#     snake_model.update_snake()
#     final_snake, init_snake,image_final = snake_model.get_snake()

#     # Convert OpenCV image to QImage
#     height, width = image_final.shape
#     bytes_per_line = width
#     q_image = QImage(image_final.data, width, height, bytes_per_line, QImage.Format_Grayscale8)

#     # Convert QImage to QPixmap
#     pixmap = QPixmap.fromImage(q_image)

#     # Create scene and add image
#     scene = QGraphicsScene()
#     pixmap_item = QGraphicsPixmapItem(pixmap)
#     scene.addItem(pixmap_item)

#     # Create view
#     view = QGraphicsView()
#     view.setScene(scene)
#     view.show()
#     sys.exit(app.exec_())

        
#     # plt.imshow(image_final, cmap="gray")
#     # plt.plot(init_snake[:, 0], init_snake[:, 1], color="b") 
#     # plt.plot(final_snake[:, 0], final_snake[:, 1], color="r")
#     # plt.legend()
#     # plt.show()                               