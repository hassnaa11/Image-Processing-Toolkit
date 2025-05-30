import numpy as np 
import cv2

from image_processor import edge_detection, FilterProcessor
class ActiveContourProcessor:
    def __init__(self, image, alpha=0.1, beta=0.3, gamma=0.1, window_size=5, iterations=400, sigma=1):
        # smooth the image
        filter_processor = FilterProcessor(image)
        image = filter_processor.apply_filter(sigma, 'Gaussian', 5)
        
        self.image = image
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.window_size = window_size
        self.iterations = iterations
        
        self.inint_snake = self.initialize_snake()
        self.snake = self.inint_snake.copy()
        
        # get gradient using canny
        edge_detection_processor = edge_detection(image)
        gradient=edge_detection_processor.apply_edge_detection_filter("Canny",  100, 150,1)
        
        # gradient = cv2.Canny(image, 100, 150)
        edges = np.array(gradient, dtype=np.float64)
        
        edges_normalized = cv2.normalize(edges.astype(np.float64), None, 0, 255, cv2.NORM_MINMAX)
        
        # Increase contrast for edges only
        edge_enhanced = np.clip(edges_normalized * 1.5, 0, 255)
        # Convert to float64 for safe computations
        self.gradient = edge_enhanced.astype(np.float64)

            
    def initialize_snake(self):
        height, width = self.image.shape[:2]
        center_x, center_y = width // 2 , height // 2 
        radius_x, radius_y = width // 2, height // 2
        s = np.linspace(0, 2 * np.pi, 100)
        x = center_x + radius_x * np.cos(s)
        y = center_y + radius_y * np.sin(s)
        snake = np.array([x, y]).T
        return snake
        
        
    def get_energy(self, point, previous_point, next_point):
        x, y = point
        x_next, y_next = next_point
        x_prev, y_prev = previous_point
        
        # External Energy (image energy)
        image_energy = -self.gamma * self.gradient[int(y), int(x)]
        
        # internal_energy (elasticity)
        elasticity = self.alpha * np.sqrt(((x_next - x) ** 2 + (y_next - y) ** 2))
        
        # internal_energy (smoothness) 
        smoothness = self.beta * np.sqrt(((x_next - 2 * x + x_prev) ** 2 + (y_next - 2 * y + y_prev) ** 2))
        
        return image_energy + elasticity + smoothness
    
    
    def update_snake(self):
        for _ in range(self.iterations): # num iterations
            new_snake = np.copy(self.snake)
            
            for i in range(len(self.snake)): # 100
                previous_point = self.snake[i - 1]
                next_point = self.snake[(i + 1) % len(self.snake)]
                
                best_point = self.snake[i]
                min_energy = float('inf')
                
                for dx in range(-self.window_size, self.window_size + 1): # 2 * window size 
                    for dy in range(-self.window_size, self.window_size + 1): # 2 * window size 
                        candidate = self.snake[i] + np.array([dx, dy])
                        if 0 <= candidate[0] < self.image.shape[1] and 0 <= candidate[1] < self.image.shape[0]:
                            energy = self.get_energy(candidate, previous_point, next_point)
                            if energy < min_energy:
                                min_energy = energy
                                best_point = candidate
                                
                new_snake[i] = best_point
                
            self.snake = new_snake
            self.resample_snake()
            
    def resample_snake(self, num_points=100):
        # Compute cumulative distances along the snake
        distances = np.sqrt(np.sum(np.diff(self.snake, axis=0) ** 2, axis=1))
        cumulative_dist = np.insert(np.cumsum(distances), 0, 0)

        # Generate new equally spaced distances
        total_length = cumulative_dist[-1]
        new_distances = np.linspace(0, total_length, num_points)

        # Interpolate new points
        x_new = np.interp(new_distances, cumulative_dist, self.snake[:, 0])
        y_new = np.interp(new_distances, cumulative_dist, self.snake[:, 1])

        self.snake = np.column_stack((x_new, y_new))
            
    def get_snake(self):
        return self.snake, self.inint_snake