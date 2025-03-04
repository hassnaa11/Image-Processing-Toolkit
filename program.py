from PyQt5 import QtWidgets, QtCore, uic
from PyQt5.QtWidgets import QFrame
import sys
from PyQt5.QtGui import *
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from Image import Image
from image_processor import FilterProcessor,FrequencyFilterProcessor, NoiseAdder, edge_detection, thresholding

kernel_sizes = [3, 5, 7]
RGB_Channels = ("red", "green", "blue")
Color =('r', 'g', 'b')
filters = ['Average','Gaussian','Median','select filter']
edge_detection_filters = ['Sobel', 'Roberts', 'Prewitt', 'Canny']
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        uic.loadUi("ui.ui", self)

        # upload buttons
        self.original_image: Image = None

        self.upload_button.clicked.connect(lambda:self.upload_image(1))
        self.input1_button.clicked.connect(lambda:self.upload_image(2))
        self.input2_button.clicked.connect(lambda:self.upload_image(3))
        
        # noises checkbox
        self.noises_combobox.setDisabled(True)
        self.noises_combobox.currentIndexChanged.connect(
            lambda: self.apply_changes("noises")
        )

        # filters checkbox
        self.filters_combobox.setDisabled(True)
        self.filters_combobox.currentIndexChanged.connect(
            lambda: self.apply_changes("filters")
        )

        # edge detection
        self.edge_filters_combobox.setDisabled(True)
        self.edge_filters_combobox.currentIndexChanged.connect(
            lambda: self.apply_changes("edge")
        )

        # thresholding
        self.threshold_combobox.setDisabled(True)
        self.threshold_combobox.currentIndexChanged.connect(
            lambda: self.apply_changes("threshold")
        )

        # normalization button
        self.normalization_button.clicked.connect(self.normalize_image)

        # sliders
        self.min_range_slider.valueChanged.connect(lambda: self.apply_changes("noises"))
        self.max_range_slider.valueChanged.connect(lambda: self.apply_changes("noises"))
        self.mean_slider.valueChanged.connect(lambda: self.apply_changes("noises"))
        self.sigma_slider.valueChanged.connect(lambda: self.apply_changes("noises"))
        self.probability_slider.valueChanged.connect(
            lambda: self.apply_changes("noises")
        )
        self.ratio_slider.valueChanged.connect(lambda: self.apply_changes("noises"))
        self.show_hide_parameters("select noise")

        # kernel size
        self.kernel_index = 0
        self.kernel_size_slider.valueChanged.connect(self.change_kernel)

        # equalize button
        self.equalization_button.clicked.connect(self.equalize_image)

        # convert to grayscale
        self.gray_scale_button.clicked.connect(self.convert_to_grayscale)
        self.is_gray_scale = False

        # reset_button
        self.reset_button.clicked.connect(self.reset)
        
        # hybrid
        self.is_hybrid_mode = False
        self.original_input1 = None
        self.original_input2 = None
        

    def upload_image(self, key):
        self.file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select Image", "", "Images (*.png *.jpg *.jpeg *.bmp *.gif)"
        )
        if self.file_path:
            self.img = Image()
            self.img.read_image(self.file_path)
            self.original_image = np.copy(self.img.image)
            scene = self.img.display_image()
            
            if key == 1: # upload in filter tap
                self.input_image.setScene(scene)
                self.input_image.fitInView(scene.sceneRect(), QtCore.Qt.KeepAspectRatio)
                self.noises_combobox.setDisabled(False)
                self.filters_combobox.setDisabled(False)
                self.edge_filters_combobox.setDisabled(False)
                self.threshold_combobox.setDisabled(False)
                self.display_histogram(self.img)
                self.display_cdf(self.img)
                
            elif key == 2:  # upload in hybrid tap first input
                self.is_hybrid_mode = True
                self.image_input1 = Image()
                self.image_input1.read_image(self.file_path)
                self.original_input1 = np.copy(self.image_input1.image)
                self.input1_combobox.currentIndexChanged.connect(self.apply_hybrid_changes)
                self.input1_image.setScene(scene)
                self.input1_image.fitInView(scene.sceneRect(), QtCore.Qt.KeepAspectRatio)
                
            elif key == 3: # upload in hybrid tap second input
                self.hybrid_output_image = Image()
                self.is_hybrid_mode = True
                self.image_input2 = Image()
                self.image_input2.read_image(self.file_path)
                self.original_input2 = np.copy(self.image_input2.image)
                self.input2_combobox.currentIndexChanged.connect(self.apply_hybrid_changes)
                self.input2_image.setScene(scene)
                self.input2_image.fitInView(scene.sceneRect(), QtCore.Qt.KeepAspectRatio)


    def display_histogram(self, image:Image, viewport = "in"):
        """Display histogram in the UI"""
        if image is not None:
            h = image.compute_histogram()
            canvas = image.plot_histogram()
            
            if viewport == "in": target_frame: QFrame = self.input_histogram
            else: target_frame:QFrame = self.output_histogram
                        
            new_layout = QtWidgets.QVBoxLayout()
            new_layout.addWidget(canvas)
            
            # Clear previous content if any
            if target_frame.layout():
                while target_frame.layout().count():
                    item = target_frame.layout().takeAt(0)
                    if item.widget():
                        item.widget().deleteLater()
                
                QtWidgets.QWidget().setLayout(target_frame.layout())        
                    
        target_frame.setLayout(new_layout)        
    

    def display_cdf(self, image:Image, viewport = "in"):
        """Display CDF in the UI"""
        if image is not None:
            # First compute histogram and CDF if not already computed
            histogram = image.compute_histogram()
            __ = image.compute_CDF(histogram)
            canvas = image.plot_cdf()
            
            if viewport=="in" : target_frame: QFrame = self.input_distribution_curve
            else : target_frame: QFrame = self.output_distribution_curve
            
            new_layout = QtWidgets.QVBoxLayout()
            new_layout.addWidget(canvas)
            
            #delete old content if exists
            if target_frame.layout():
            # Remove all widgets from old layout
                while target_frame.layout().count():
                    item = target_frame.layout().takeAt(0)
                    if item.widget():
                        item.widget().deleteLater()
                
                # Delete old layout
                QtWidgets.QWidget().setLayout(target_frame.layout())     
    
        target_frame.setLayout(new_layout)
    
    
    def apply_changes(self,type):
        kernel_size = kernel_sizes[self.kernel_index]
        if self.img and self.original_image is not None:
            modified_image = np.copy(self.original_image)

            # Apply noise if selected
            noise_type = self.noises_combobox.currentText()
            if noise_type != "None" and (type == "noises" or type == "filters"):
                parameters = self.get_noise_parameters(noise_type)
                noise_adder = NoiseAdder(modified_image)
                modified_image = noise_adder.apply_noise(noise_type, parameters)

            # Apply filter if selected
            filter_type = self.filters_combobox.currentText()
            print(filter_type)
            if filter_type != "None" and (type == "filters" or type == "noises"):
                if filter_type == 'Low-Pass Frequency Domain':
                    filter_processor = FrequencyFilterProcessor(modified_image)
                    modified_image = filter_processor.apply_frequency_filter(0.5, filter_type)
                else:    
                    filter_processor = FilterProcessor(modified_image)
                    modified_image = filter_processor.apply_filter(filter_type, kernel_size)

            # apply edge detection
            edge_detection_type = self.edge_filters_combobox.currentText()
            if edge_detection_type != "None" and type == "edge":
                if edge_detection_type == 'High-Pass Frequency Domain':
                    filter_processor = FrequencyFilterProcessor(modified_image)
                    modified_image = filter_processor.apply_frequency_filter(0.5, edge_detection_type)
                else:  
                    edge_detection_processor = edge_detection(modified_image)
                    modified_image = edge_detection_processor.apply_edge_detection_filter(edge_detection_type)

            # apply thresholding
            thresholding_type = self.threshold_combobox.currentText()
            if thresholding_type != "None" and type == "threshold":
                print(thresholding_type)
                thresholding_processor = thresholding(modified_image)
                print(modified_image)
                modified_image = thresholding_processor.apply_threshold(thresholding_type)

            self.img.image = modified_image
            scene = self.img.display_image()
            self.output_image.setScene(scene)
            self.output_image.fitInView(scene.sceneRect(), QtCore.Qt.KeepAspectRatio)
            
            self.display_histogram(self.img, "out")
            self.display_cdf(self.img, "out")
    
    
    def apply_hybrid_changes(self):
        if self.original_input1 is not None:
            filter_type = self.input1_combobox.currentText()
            modified_image = np.copy(self.original_input1)
            
            if filter_type == 'Low-Pass Frequency Domain' or filter_type == 'High-Pass Frequency Domain':
                filter_processor = FrequencyFilterProcessor(modified_image)
                modified_image = filter_processor.apply_frequency_filter(0.5, filter_type)
            elif filter_type in filters:
                filter_processor = FilterProcessor(modified_image)
                modified_image = filter_processor.apply_filter(filter_type, 3)
            elif filter_type in edge_detection_filters:
                edge_detection_processor = edge_detection(modified_image)
                modified_image = edge_detection_processor.apply_edge_detection_filter(filter_type)        
                    
            self.image_input1.image = modified_image
            scene = self.image_input1.display_image()
            self.output1_image.setScene(scene)
            self.output1_image.fitInView(scene.sceneRect(), QtCore.Qt.KeepAspectRatio)
        
        if self.original_input2 is not None:
            filter_type = self.input2_combobox.currentText()
            modified_image = np.copy(self.original_input2)
            
            if filter_type == 'Low-Pass Frequency Domain' or filter_type == 'High-Pass Frequency Domain':
                filter_processor = FrequencyFilterProcessor(modified_image)
                modified_image = filter_processor.apply_frequency_filter(0.5, filter_type)
            elif filter_type in filters:
                filter_processor = FilterProcessor(modified_image)
                modified_image = filter_processor.apply_filter(filter_type, 3)
            elif filter_type in edge_detection_filters:
                edge_detection_processor = edge_detection(modified_image)
                modified_image = edge_detection_processor.apply_edge_detection_filter(filter_type)        
            
            self.image_input2.image = modified_image
            scene = self.image_input2.display_image()
            self.output2_image.setScene(scene)
            self.output2_image.fitInView(scene.sceneRect(), QtCore.Qt.KeepAspectRatio)    
        
        if (self.original_input1 is not None) and (self.original_input2 is not None):
            hybrid_image = (self.image_input1.image * 0.5 + self.image_input2.image * 0.5).astype(np.uint8)
            self.hybrid_output_image.image = hybrid_image
            scene = self.hybrid_output_image.display_image()
            self.hybrid_image.setScene(scene)
            self.hybrid_image.fitInView(scene.sceneRect(), QtCore.Qt.KeepAspectRatio)
            
        
    def normalize_image(self):

        modified_image = np.copy(self.original_image)
        # Convert to grayscale if it's a color image
        if len(modified_image.shape) == 3:
            grayscale_image = np.mean(modified_image, axis=2)
        else:
            grayscale_image = modified_image

        # Calculate min and max pixel values
        I_min = np.min(grayscale_image)
        I_max = np.max(grayscale_image)

        # Perform normalization
        normalized_image = (grayscale_image - I_min) / (I_max - I_min) * 255
        normalized_image = normalized_image.astype(np.uint8)

        # display the normalized image
        self.img.image = normalized_image
        
        scene = self.img.display_image()
        self.output_image.setScene(scene) 
        self.output_image.fitInView(scene.sceneRect(), QtCore.Qt.KeepAspectRatio)
        
        self.display_histogram(self.img, "out")
        self.display_cdf(self.img, "out") 
    
    def get_noise_parameters(self, selected_noise):
        self.show_hide_parameters(selected_noise)
        parameters = []

        if selected_noise == "Uniform":
            min_range, max_range = (
                self.min_range_slider.value(),
                self.max_range_slider.value(),
            )
            parameters = [min_range, max_range]
            self.min_range_label.setText(f"Minimum Range: {min_range}")
            self.max_range_label.setText(f"Maximum Range: {max_range}")

        elif selected_noise == "Gaussian":
            mean, sigma = self.mean_slider.value(), self.sigma_slider.value()
            parameters = [mean, sigma]
            self.mean_label.setText(f"Mean: {mean}")
            self.sigma_label.setText(f"Sigma: {sigma}")

        elif selected_noise == "Salt & Pepper":
            ratio, probability = (
                self.ratio_slider.value(),
                self.probability_slider.value(),
            )
            parameters = [ratio / 10, probability / 10]
            self.ratio_label.setText(f"Ratio: {ratio/10}")
            self.probability_label.setText(f"Probability: {probability/10}")

        return parameters

    def equalize_image(self):
        modified_image = np.copy(self.original_image)
        filter_processor = FilterProcessor(modified_image)
        modified_image = filter_processor.histogram_equalization()
        self.img.image = modified_image
        scene = self.img.display_image()
        self.output_image.setScene(scene) 
        self.output_image.fitInView(scene.sceneRect(), QtCore.Qt.KeepAspectRatio) 

        self.display_histogram(self.img, "out")
        self.display_cdf(self.img, "out")
    
    # def convert_to_grayscale(self):
    #     modified_image: Image = np.copy(self.original_image)
    #     filter_processor = FilterProcessor(modified_image)
    #     #self.is_gray_scale = not self.is_gray_scale
        
    #     if modified_image.is_RGB():
    #         self.rgb_image = np.copy(self.original_image)
    #         self.gray_scale_button.setText("Original") 
    #         #modified_image = filter_processor.rgb_to_grayscale()
    #         modified_image.rgb2gray()
    #     else: 
    #         self.gray_scale_button.setText("GrayScale") 
    #         modified_image = np.copy(self.rgb_image)

    #     self.original_image = np.copy(modified_image)
    #     self.apply_changes(type="filters")
        
    def convert_to_grayscale(self):
        modified_image: Image = np.copy(self.original_image)
        
        if modified_image.is_RGB():
            modified_image.rgb2gray()
            self.display_histogram(modified_image, "out")
            self.display_cdf(modified_image, "out")
            
        else:
            print("Image is already grayscale")
            return       
   
        
    def change_kernel(self):
        self.kernel_index = self.kernel_size_slider.value()
        self.kernel_size_label.setText(
            f"Kernel Size: {kernel_sizes[self.kernel_index]}x{kernel_sizes[self.kernel_index]}"
        )
        self.apply_changes(type="filters")

    def show_hide_parameters(self, selected_noise):
        if selected_noise == "select noise":
            self.min_range_slider.hide()
            self.max_range_slider.hide()
            self.min_range_label.hide()
            self.max_range_label.hide()
            self.mean_slider.hide()
            self.sigma_slider.hide()
            self.mean_label.hide()
            self.sigma_label.hide()
            self.ratio_slider.hide()
            self.probability_slider.hide()
            self.ratio_label.hide()
            self.probability_label.hide()

        if selected_noise == "Uniform":
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

        elif selected_noise == "Gaussian":
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

        elif selected_noise == "Salt & Pepper":
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


    def reset(self):
        if self.img and self.original_image is not None:
            self.original_image = None
            self.img = None
            self.show_hide_parameters("select noise")
            self.noises_combobox.setCurrentText("select noise")
            self.filters_combobox.setCurrentText("select filter")
            self.edge_filters_combobox.setCurrentText("select edge detection filter")
            self.threshold_combobox.setCurrentText("select thresholding type")
            self.noises_combobox.setDisabled(True)
            self.filters_combobox.setDisabled(True)
            self.edge_filters_combobox.setDisabled(True)
            self.threshold_combobox.setDisabled(True)
            self.input_image.scene().clear()
            if self.output_image.scene() is not None:
                self.output_image.scene().clear()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
