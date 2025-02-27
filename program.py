from PyQt5 import QtWidgets, QtCore, uic 
import sys
from PyQt5.QtGui import *
import numpy as np
import cv2

from Image import Image
from image_processor import FilterProcessor, NoiseAdder
kernel_sizes = [3, 5, 7]

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        uic.loadUi('ui.ui', self)
        
        # upload buttons
        self.original_image = None
        self.upload_button.clicked.connect(lambda:self.upload_image(1))
        self.input1_button.clicked.connect(lambda:self.upload_image(2))
        self.input2_button.clicked.connect(lambda:self.upload_image(3))
        
        # noises checkbox
        self.noises_combobox.setDisabled(True)
        self.noises_combobox.currentIndexChanged.connect(self.apply_changes)
        
        # filters checkbox
        self.filters_combobox.setDisabled(True)
        self.filters_combobox.currentIndexChanged.connect(self.apply_changes)
        
        # sliders
        self.min_range_slider.valueChanged.connect(self.apply_changes)
        self.max_range_slider.valueChanged.connect(self.apply_changes)
        self.mean_slider.valueChanged.connect(self.apply_changes)
        self.sigma_slider.valueChanged.connect(self.apply_changes)
        self.probability_slider.valueChanged.connect(self.apply_changes)
        self.ratio_slider.valueChanged.connect(self.apply_changes)
        self.min_range_slider.setDisabled(True)
        self.max_range_slider.setDisabled(True)
        self.show_hide_parameters('Uniform')
        
        # kernel size
        self.kernel_index = 0
        self.kernel_size_slider.valueChanged.connect(self.change_kernel)

        #edge detection
        self.edge_filters_combobox.currentIndexChanged.connect(self.apply_edge_detection_filter)
        
        # equalize button
        self.equalization_button.clicked.connect(self.equalize_image)
        # convert to grayscale
        self.gray_scale_button.clicked.connect(self.convert_to_grayscale)
        self.is_gray_scale = False
        
        
    def upload_image(self, key):  
        self.file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select Image", "", "Images (*.png *.jpg *.jpeg *.bmp *.gif)"
        )
        if self.file_path:
            self.processor = Image()
            self.processor.read_image(self.file_path)
            self.noisy_image = Image()  # shelehom ya eman ma3lsh
            self.noisy_image.read_image(self.file_path)
            self.original_image = np.copy(self.processor.image)
            
            scene = self.processor.display_image()
            if key == 1:
                self.input_image.setScene(scene) 
                self.input_image.fitInView(scene.sceneRect(), QtCore.Qt.KeepAspectRatio)  
                self.noises_combobox.setDisabled(False)
                self.filters_combobox.setDisabled(False)
                self.min_range_slider.setDisabled(False)
                self.max_range_slider.setDisabled(False)
            elif key == 2:
                self.input1_image.setScene(scene)
                self.input1_image.fitInView(scene.sceneRect(), QtCore.Qt.KeepAspectRatio)
            elif key == 3:           
                self.input2_image.setScene(scene)
                self.input2_image.fitInView(scene.sceneRect(), QtCore.Qt.KeepAspectRatio)
    
    
    def apply_changes(self):
        kernel_size = kernel_sizes[self.kernel_index]
        if self.processor and self.original_image is not None:
            modified_image = np.copy(self.original_image)

            # Apply noise if selected
            noise_type = self.noises_combobox.currentText()
            if noise_type != "None":
                parameters = self.get_noise_parameters(noise_type)
                noise_adder = NoiseAdder(modified_image)
                modified_image = noise_adder.apply_noise(noise_type, parameters)

            # Apply filter if selected
            filter_type = self.filters_combobox.currentText()
            if filter_type != "None":
                filter_processor = FilterProcessor(modified_image)
                modified_image = filter_processor.apply_filter(filter_type, kernel_size)

            self.processor.image = modified_image
            scene = self.processor.display_image()
            self.output_image.setScene(scene) 
            self.output_image.fitInView(scene.sceneRect(), QtCore.Qt.KeepAspectRatio) 
        
    
    def get_noise_parameters(self, selected_noise):
        self.show_hide_parameters(selected_noise)
        parameters = []
        
        if selected_noise == 'Uniform':
            min_range, max_range = self.min_range_slider.value(), self.max_range_slider.value()
            parameters = [min_range, max_range]
            self.min_range_label.setText(f"Minimum Range: {min_range}")
            self.max_range_label.setText(f"Minimum Range: {max_range}")
            
        elif selected_noise == 'Gaussian': 
            mean, sigma = self.mean_slider.value(), self.sigma_slider.value()
            parameters = [mean, sigma] 
            self.mean_label.setText(f"Mean: {mean}")
            self.sigma_label.setText(f"Sigma: {sigma}")
            
        elif selected_noise == 'Salt & Pepper':
            ratio, probability = self.ratio_slider.value(),self.probability_slider.value()
            parameters = [ratio/10, probability/10]  
            self.ratio_label.setText(f"Ratio: {ratio/10}")
            self.probability_label.setText(f"Probability: {probability/10}")
        
        return parameters    
        
    
    def equalize_image(self):
        modified_image = np.copy(self.original_image)
        filter_processor = FilterProcessor(modified_image)
        modified_image = filter_processor.histogram_equalization()
        self.processor.image = modified_image
        scene = self.processor.display_image()
        self.output_image.setScene(scene) 
        self.output_image.fitInView(scene.sceneRect(), QtCore.Qt.KeepAspectRatio) 
    
    
    def convert_to_grayscale(self):
        modified_image = np.copy(self.original_image)
        filter_processor = FilterProcessor(modified_image)
        self.is_gray_scale = not self.is_gray_scale
        
        if self.is_gray_scale:
            self.rgb_image = np.copy(self.original_image)
            self.gray_scale_button.setText("Original") 
            modified_image = filter_processor.rgb_to_grayscale()
        else: 
            self.gray_scale_button.setText("GrayScale") 
            modified_image = np.copy(self.rgb_image)

        self.original_image = np.copy(modified_image)
        self.apply_changes()
        
        
    def change_kernel(self):
        self.kernel_index = self.kernel_size_slider.value()
        self.kernel_size_label.setText(f"Kernel Size: {kernel_sizes[self.kernel_index]}x{kernel_sizes[self.kernel_index]}")
        self.apply_changes()
    
    def show_hide_parameters(self, selected_noise):
        if selected_noise == 'Uniform':
            self.min_range_slider.show()
            self.max_range_slider.show()
            self.min_range_label.show()
            self.max_range_label.show()
            self.mean_slider.hide()
            self.sigma_slider.hide()
            self.mean_label.hide()
            self.sigma_label.hide()
            self.ratio_slider.hide()
            self.probability_slider.hide()
            self.ratio_label.hide()
            self.probability_label.hide()
            
        elif selected_noise == 'Gaussian':
            self.min_range_slider.hide()
            self.max_range_slider.hide()
            self.min_range_label.hide()
            self.max_range_label.hide() 
            self.ratio_slider.hide()
            self.probability_slider.hide()
            self.ratio_label.hide()
            self.probability_label.hide()
            self.mean_slider.show()
            self.sigma_slider.show()
            self.mean_label.show()
            self.sigma_label.show()
        
        elif selected_noise == 'Salt & Pepper':
            self.min_range_slider.hide()
            self.max_range_slider.hide()
            self.min_range_label.hide()
            self.max_range_label.hide() 
            self.mean_slider.hide()
            self.sigma_slider.hide()
            self.mean_label.hide()
            self.sigma_label.hide()    
            self.ratio_slider.show()
            self.probability_slider.show()
            self.ratio_label.show()
            self.probability_label.show()
            
    def apply_edge_detection_filter(self):
        selected_edge_detection_filter = self.edge_filters_combobox.currentText()
        scene =self.noisy_image.apply_edge_detection_filter(selected_edge_detection_filter)
        self.output_image.setScene(scene)
        self.output_image.fitInView(scene.sceneRect(), QtCore.Qt.KeepAspectRatio)

   
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())        