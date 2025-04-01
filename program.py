from PyQt5 import QtWidgets, QtCore, uic
from PyQt5.QtWidgets import QGraphicsPixmapItem,QGraphicsEllipseItem,QGraphicsPathItem, QFrame, QGraphicsScene, QGraphicsView
import sys
from PyQt5.QtCore import Qt
from PyQt5.QtGui import *
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from Image import Image
from typing import Dict, List
from image_processor import FilterProcessor,FrequencyFilterProcessor, NoiseAdder, edge_detection, thresholding
from active_contour_processor import ActiveContourProcessor
from reportlab.pdfgen import canvas
from shapes import detect_shapes, canny_filter
from SIFT import SIFTApp


kernel_sizes = [3, 5, 7]
RGB_Channels = ("red", "green", "blue")
Color =('r', 'g', 'b')
filters = ['Average','Gaussian','Median','select filter']
edge_detection_filters = ['Sobel', 'Roberts', 'Prewitt', 'Canny']


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        uic.loadUi('ui new.ui', self)
        
        self.database: List[Image] = []
        
        # upload buttons
        self.upload_button.clicked.connect(lambda:self.upload_image(1))
        self.input1_button.clicked.connect(lambda:self.upload_image(2))
        self.input2_button.clicked.connect(lambda:self.upload_image(3))
        self.upload_image_contour.clicked.connect(lambda:self.upload_image(4))
        self.hough_transform_upload_btn.clicked.connect(lambda: self.upload_image(5))
        self.pushButton_2.clicked.connect(lambda:self.upload_image(6))
        self.pushButton.clicked.connect(lambda:self.upload_image(7))
       
        
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
        self.sigma_filter_slider.valueChanged.connect(lambda: self.apply_changes("filters"))
        self.sigma_filter_slider.hide()
        self.sigma_filter_label.hide()
        self.probability_slider.valueChanged.connect(
            lambda: self.apply_changes("noises")
        )
        self.ratio_slider.valueChanged.connect(lambda: self.apply_changes("noises"))
        self.show_hide_parameters("select noise")
        self.low_threshold_label.setText(f"Low Threshold: 50")
        self.high_threshold_label.setText(f"High Threshold: 100")
        self.low_threshold_slider.setMinimum(0)   # Minimum value
        self.low_threshold_slider.setMaximum(100) # Maximum value
        self.low_threshold_slider.setValue(50) 
        self.high_threshold_slider.setMinimum(0)   # Minimum value
        self.high_threshold_slider.setMaximum(150) # Maximum value
        self.high_threshold_slider.setValue(100) 
        self.sigma_canny_slider.setRange(1,30)
        self.low_threshold_slider.setSingleStep(10)
        self.high_threshold_slider.setSingleStep(10)
        self.low_threshold_slider.hide()
        self.high_threshold_slider.hide()
        self.sigma_canny_slider.hide()
        self.low_threshold_label.hide()
        self.high_threshold_label.hide()
        self.sigma_canny_label.hide()
       

        # kernel size
        self.kernel_index = 0
        self.kernel_size_slider.valueChanged.connect(self.change_kernel)

        #low and high threshold sliders
        self.low_threshold_slider.valueChanged.connect(self.change_slider_value)
        self.high_threshold_slider.valueChanged.connect(self.change_slider_value)
        # equalize button
        self.equalization_button.clicked.connect(self.equalize_image)

        # convert to grayscale
        self.gray_scale_button.clicked.connect(self.convert_color_domain)
        self.is_gray_scale = False

        # reset_button
        self.reset_button.clicked.connect(self.reset)
        
        # hybrid
        self.is_hybrid_mode = False
        self.original_input1 = None
        self.original_input2 = None
        
        # contour 
        self.apply_contour_button.clicked.connect(self.apply_contour)

        #chain code
        self.chaincode_button.clicked.connect(self.apply_chain_code)
        
        #hough transform buttons
        self.apply_hough_button.clicked.connect(self.apply_hough_changes)
        self.hough_reset_btn.clicked.connect(self.reset_hough_tab)

        #apply sift radiobutton
        self.radioButton_3.toggled.connect(self.apply_sift)
        

    def upload_image(self, key):
        self.file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select Image", "", "Images (*.png *.jpg *.jpeg *.bmp *.gif)"
        )
        
        if self.file_path:
            uploaded_img = Image()
            uploaded_img.read_image(self.file_path)
            scene = uploaded_img.display_image()

            self.original_image_arr = np.copy(uploaded_img.image)  
            self.input_image = Image(np.copy(self.original_image_arr))          
            
            if key == 1: # upload in filter tap
                if self.output_image_frame.scene() is not None: 
                    self.output_image_frame.scene().clear()
                    self.output_image = None
                
                self.input_image = Image(np.copy(self.original_image_arr))
                
                self.input_image_frame.setScene(scene)
                self.input_image_frame.fitInView(scene.sceneRect(), QtCore.Qt.KeepAspectRatio)
                self.noises_combobox.setDisabled(False)
                self.filters_combobox.setDisabled(False)
                self.edge_filters_combobox.setDisabled(False)
                self.threshold_combobox.setDisabled(False)
                self.display_histogram(self.input_image)
                self.display_cdf(self.input_image) 
                
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
                
            elif key == 4: # upload in contour tap
                self.contour_output_frame.setScene(scene)
                self.contour_output_frame.fitInView(scene.sceneRect(), QtCore.Qt.KeepAspectRatio)
                
            elif key==5: # Upload in Hough Transform Tab
                #clear last image
                self.reset_hough_tab()
                h, w = self.hough_image.image.shape[0], self.hough_image.image.shape[1] 
                
                self.hough_dimensions_label.setText(str(h)+"x"+str(w))
                
                self.hough_transform_output_frame.setScene(scene)
                self.hough_transform_output_frame.fitInView(scene.sceneRect(), QtCore.Qt.KeepAspectRatio)
                self.hough_image = uploaded_img
            elif key==6:
                self.SIFT_image1=self.input_image
                self.graphicsView.setScene(scene)
                self.graphicsView.fitInView(scene.sceneRect(), QtCore.Qt.KeepAspectRatio)
            elif key==7:
                self.SIFT_image2=self.input_image
                self.graphicsView_2.setScene(scene)
                self.graphicsView_2.fitInView(scene.sceneRect(), QtCore.Qt.KeepAspectRatio)

                            
    # hough_transform_ratio_spinbox
    
    def apply_hough_changes(self):
        #apply canny filter first
        sigma = self.sigma_spinbox_hough.value()
        t_low = self.low_threshold_spinbox_hough.value()
        t_high = self.high_threshold_spinbox_hough.value()
        
        cpy_arr: np.ndarray = np.copy(self.hough_image.image)
        
        
        if self.kernel_3_radio_btn.isChecked(): kerenl_size = 3
        elif self.kernel_5_radio_btn.isChecked(): kerenl_size =5
        elif self.kernel_7_radio_btn.isChecked(): kerenl_size = 7
        
        canny_filtered_img_arr = canny_filter(cpy_arr, sigma, t_low, t_high, kerenl_size)
        
        detect_lines = True if self.lines_checkbox.isChecked() else False
        detect_ellipses = True if self.ellipses_checkbox.isChecked() else False
        detect_circles = True if self.circles_checkbox.isChecked() else False
        
        threhold_ratio = self.hough_transform_ratio_spinbox.value()
        theta_step_size = self.theta_step_size_spinbox.value()
        min_r = self.min_r_spinbox.value()
        max_r = self.max_r_spinbox.value()
        minor_axis_step_size = self.minor_axis_step_size_spinbox.value()
        major_axis_step_size = self.major_axis_step_size_spinbox.value()
        
        min_line_length = self.min_line_length_spinbox.value() 
        
        scene: QGraphicsScene = detect_shapes(self.hough_image.image, canny_filtered_img_arr, detect_lines, 
        detect_ellipses, detect_circles, min_line_length, threhold_ratio, theta_step_size, min_r, max_r, minor_axis_step_size, major_axis_step_size)
        
        self.hough_transform_output_frame.setScene(scene)
        self.hough_transform_output_frame.fitInView(scene.sceneRect(), QtCore.Qt.KeepAspectRatio)
    
    
    def reset_hough_tab(self):
        if self.hough_transform_output_frame.scene() is not None: 
            self.hough_transform_output_frame.scene().clear()
            self.hough_image = None
            
        self.lines_checkbox.setChecked(False)
        self.circles_checkbox.setChecked(False)    
        self.ellipses_checkbox.setChecked(False)
        
        self.kernel_3_radio_btn.setChecked(False)
        self.kernel_5_radio_btn.setChecked(False)
        self.kernel_7_radio_btn.setChecked(False)
        
        self.hough_transform_ratio_spinbox.setValue(0.80)
        self.theta_step_size_spinbox.setValue(20)
        self.minor_axis_step_size_spinbox.setValue(5)
        self.major_axis_step_size_spinbox.setValue(5)
        
        self.min_r_spinbox.setValue(1)
        self.max_r_spinbox.setValue(400)   
        
        self.min_line_length_spinbox.setValue(10)
        
        self.hough_dimensions_label.setText("Image Dimensions(hxw)")
        
            
    def change_slider_value(self):
        low_threshold=self.low_threshold_slider.value()
        high_threshold=self.high_threshold_slider.value()
        self.low_threshold_label.setText(f"Low Threshold: {low_threshold}")
        self.high_threshold_label.setText(f"High Threshold: {high_threshold}")
        self.apply_changes("edge")

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
        if self.input_image:
            input_img_arr_cpy = np.copy(self.input_image.image)
            
            modified_image = Image(input_img_arr_cpy)

            # Apply noise if selected
            noise_type = self.noises_combobox.currentText()
            if noise_type != "None" and (type == "noises" or type == "filters"):
                parameters = self.get_noise_parameters(noise_type)
                noise_adder = NoiseAdder(modified_image.image)
                modified_image.image = noise_adder.apply_noise(noise_type, parameters)

            # Apply filter if selected
            filter_type = self.filters_combobox.currentText()
            print(filter_type)
            sigma = 1
            if filter_type != "None" and (type == "filters" or type == "noises"):
                if filter_type == 'Low-Pass Frequency Domain':
                    filter_processor = FrequencyFilterProcessor(modified_image.image)
                    modified_image.image = filter_processor.apply_frequency_filter(0.5, filter_type)
                else:    
                    if filter_type == 'Gaussian':
                        self.sigma_filter_slider.show()
                        self.sigma_filter_label.show()
                        sigma = self.sigma_filter_slider.value()
                        self.sigma_filter_label.setText(f"Sigma: {sigma/2.0}")
                    else:
                        self.sigma_filter_slider.hide()
                        self.sigma_filter_label.hide()   
                    filter_processor = FilterProcessor(modified_image.image)
                    modified_image.image = filter_processor.apply_filter(sigma/2.0, filter_type, kernel_size)

            # apply edge detection
            edge_detection_type = self.edge_filters_combobox.currentText()
            if edge_detection_type == "Canny":
                self.low_threshold_slider.show()
                self.high_threshold_slider.show()
                self.sigma_canny_slider.show()
                self.low_threshold_label.show()
                self.high_threshold_label.show()
                self.sigma_canny_label.show()
            else:
                self.low_threshold_slider.hide()
                self.high_threshold_slider.hide()
                self.sigma_canny_slider.hide()  
                self.low_threshold_label.hide()
                self.high_threshold_label.hide()
                self.sigma_canny_label.hide()  
                
            if edge_detection_type != "None" and type == "edge":
                if edge_detection_type == 'High-Pass Frequency Domain':
                    filter_processor = FrequencyFilterProcessor(modified_image.image)
                    modified_image.image = filter_processor.apply_frequency_filter(0.5, edge_detection_type)
                else:
                    low_threshold,high_threshold, sigma_gaussian = self.low_threshold_slider.value(), self.high_threshold_slider.value(),self.sigma_canny_slider.value()
                    edge_detection_processor = edge_detection(modified_image.image)
                    modified_image.image = edge_detection_processor.apply_edge_detection_filter(edge_detection_type,low_threshold,high_threshold, sigma_gaussian)

            # apply thresholding
            thresholding_type = self.threshold_combobox.currentText()
            if thresholding_type != "None" and type == "threshold":
                print(thresholding_type)
                thresholding_processor = thresholding(modified_image.image)
                print(modified_image.image)
                modified_image.image = thresholding_processor.apply_threshold(thresholding_type)
            
            self.output_image = modified_image
            
            scene = self.output_image.display_image()        
            self.output_image_frame.setScene(scene) 
            self.output_image_frame.fitInView(scene.sceneRect(), QtCore.Qt.KeepAspectRatio)
            
            self.display_histogram(self.output_image, "out")
            self.display_cdf(self.output_image, "out")

        
    def apply_hybrid_changes(self):
        if self.original_input1 is not None:
            filter_type = self.input1_combobox.currentText()
            modified_image = np.copy(self.original_input1)
            
            if filter_type == 'Low-Pass Frequency Domain' or filter_type == 'High-Pass Frequency Domain':
                filter_processor = FrequencyFilterProcessor(modified_image)
                modified_image = filter_processor.apply_frequency_filter(0.5, filter_type)
            elif filter_type in filters:
                filter_processor = FilterProcessor(modified_image)
                modified_image = filter_processor.apply_filter(1,filter_type, 3)
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
                modified_image = filter_processor.apply_filter(1,filter_type, 3)
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
        
    def apply_sift(self):
        SIFT=SIFTApp(self.SIFT_image1,self.SIFT_image2)
        if self.radioButton_3.isChecked():
            image=SIFT.apply_SIFT()
            SIFT_IMAGE= Image(image)
            scene = SIFT_IMAGE.display_image()
            self.graphicsView_4.setScene(scene)
            self.graphicsView_4.fitInView(scene.sceneRect(), QtCore.Qt.KeepAspectRatio)



    def normalize_image(self):

        modified_image = Image(np.copy(self.input_image.image))
        # Convert to grayscale if it's a color image
        
        if modified_image.is_RGB(): modified_image.rgb2gray()
        grayscale_image = modified_image

        # Calculate min and max pixel values
        I_min = np.min(grayscale_image.image)
        I_max = np.max(grayscale_image.image)

        # Perform normalization
        normalized_image_arr = (grayscale_image.image - I_min) / (I_max - I_min) * 255
        normalized_image_arr = normalized_image_arr.astype(np.uint8)
        
        self.output_image = Image(normalized_image_arr)

        scene = self.output_image.display_image()
        
        if isinstance(self.output_image_frame, QGraphicsView) and scene:
            print("QGraphicsScene Was Set Successfully")
            self.output_image_frame.setScene(scene) 
            self.output_image_frame.fitInView(scene.sceneRect(), QtCore.Qt.KeepAspectRatio)
        
        self.display_histogram(self.output_image, "out")
        self.display_cdf(self.output_image, "out") 
    
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
        modified_image = Image(np.copy(self.input_image.image))
        filter_processor = FilterProcessor(modified_image.image)
        modified_image.image = filter_processor.histogram_equalization()
        
        self.output_image = modified_image
        
        scene = self.output_image.display_image()
        if isinstance(self.output_image_frame, QGraphicsView) and scene:
            print("QGraphicsScene Was Set Successfully")
            self.output_image_frame.setScene(scene) 
            self.output_image_frame.fitInView(scene.sceneRect(), QtCore.Qt.KeepAspectRatio) 

        self.display_histogram(self.output_image, "out")
        self.display_cdf(self.output_image, "out")
    
    def convert_color_domain(self):
        if self.gray_scale_button.text() == "GrayScale":
            self.convert_to_grayscale()
        else: self.convert_to_rgb()    
        
    def convert_to_grayscale(self):        
        if self.input_image.is_RGB() and self.input_image:
            self.prev_rgb_input_img = Image(np.copy(self.input_image.image))
            
            cpy_image: Image = Image(np.copy(self.input_image.image))
            cpy_image.rgb2gray()
            
            self.display_histogram(cpy_image)
            self.display_cdf(cpy_image)
            
            scene = cpy_image.display_image()
            self.input_image_frame.setScene(scene) 
            self.input_image_frame.fitInView(scene.sceneRect(), QtCore.Qt.KeepAspectRatio) 
            
            self.input_image = cpy_image
            
        if self.output_image.is_RGB() and self.output_image:
            self.prev_rgb_output_img = Image(np.copy(self.output_image.image))
            
            cpy_image: Image = Image(np.copy(self.output_image.image))
            cpy_image.rgb2gray()
            
            self.display_histogram(cpy_image)
            self.display_cdf(cpy_image)
            
            scene = cpy_image.display_image()
            self.output_image_frame.setScene(scene) 
            self.output_image_frame.fitInView(scene.sceneRect(), QtCore.Qt.KeepAspectRatio)
          
            self.output_image = cpy_image
        
        self.gray_scale_button.setText("Back to RGB")
     
    def convert_to_rgb(self):
        if self.input_image and self.input_image.is_RGB() == False:     
            scene = self.prev_rgb_input_img.display_image()
            self.input_image_frame.setScene(scene) 
            self.input_image_frame.fitInView(scene.sceneRect(), QtCore.Qt.KeepAspectRatio)
            
            self.display_histogram(self.prev_rgb_input_img)
            self.display_cdf(self.prev_rgb_input_img)
            
            self.input_image = self.prev_rgb_input_img
            self.prev_rgb_input_img = None
            
        if self.output_image and self.output_image.is_RGB() == False:     
            scene = self.prev_rgb_output_img.display_image()
            self.output_image_frame.setScene(scene) 
            self.output_image_frame.fitInView(scene.sceneRect(), QtCore.Qt.KeepAspectRatio)
            
            self.display_histogram(self.prev_rgb_output_img, "out")
            self.display_cdf(self.prev_rgb_output_img, "out")
            
            self.output_image = self.prev_rgb_output_img
            self.prev_rgb_output_img = None    
        
        self.gray_scale_button.setText("GrayScale")
        
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
        if self.input_image:
            self.show_hide_parameters("select noise")
            self.noises_combobox.setCurrentText("select noise")
            self.filters_combobox.setCurrentText("select filter")
            self.edge_filters_combobox.setCurrentText("select edge detection filter")
            self.threshold_combobox.setCurrentText("select thresholding type")
            self.noises_combobox.setDisabled(True)
            self.filters_combobox.setDisabled(True)
            self.edge_filters_combobox.setDisabled(True)
            self.threshold_combobox.setDisabled(True)
            self.input_image_frame.scene().clear()
            if self.output_image_frame.scene() is not None:
                self.output_image_frame.scene().clear()
    
    
    def apply_chain_code(self):
        direction_map_8 = {
            (1, 0): 0,   # Right →
            (1, -1): 1,  # Up-Right ↗
            (0, -1): 2,  # Up ↑
            (-1, -1): 3, # Up-Left ↖
            (-1, 0): 4,  # Left ←
            (-1, 1): 5,  # Down-Left ↙
            (0, 1): 6,   # Down ↓
            (1, 1): 7    # Down-Right ↘
        }

        chain_code = []
        final_snake, init_snake = self.snake_model.get_snake()
       
        final_snake = np.array(final_snake)
        print(final_snake)

        # Find the top-left point (minimum y, then minimum x)
        min_index = np.lexsort((final_snake[:, 1], final_snake[:, 0]))[0]
        print(min_index)

        # Reorder the snake to start from this point
        ordered_snake = np.roll(final_snake, -min_index, axis=0)

        # Ensure integer coordinates
        ordered_snake = np.round(ordered_snake ).astype(int)

        for i in range(len(ordered_snake ) - 1):  # Loop through snake points
            x1, y1 = ordered_snake [i]
            x2, y2 = ordered_snake [i + 1]

            move = (x2 - x1, y2 - y1)  # Compute movement vector

            if move in direction_map_8:  # Valid movement
                chain_code.append(direction_map_8[move])
            else:
                # If move is not in the direction map, find the nearest valid move
                closest_move = min(direction_map_8.keys(), key=lambda k: np.linalg.norm(np.array(move) - np.array(k)))
                chain_code.append(direction_map_8[closest_move])
        self.save_chain_code_to_pdf(chain_code)
        print("Chain Code:", chain_code)
        return chain_code
 

    def save_chain_code_to_pdf(self, chain_code, filename="chain_code.pdf"):
        # Create a PDF canvas
        c = canvas.Canvas(filename)

        # Add title
        c.setFont("Helvetica-Bold", 14)
        c.drawString(100, 800, "Snake Chain Code Representation")

        # Add chain code
        c.setFont("Helvetica", 12)
        y_position = 780  # Start position for text
        chain_text = "Chain Code: " + " ".join(map(str, chain_code))

        # Split text if too long
        max_width = 80  # Characters per line
        lines = [chain_text[i:i+max_width] for i in range(0, len(chain_text), max_width)]

        for line in lines:
            c.drawString(100, y_position, line)
            y_position -= 20  # Move down for next line

        # Save PDF
        c.save()
        print(f"Chain code saved to {filename}")

    
    def apply_contour(self):
        input_image_copy = np.copy(self.input_image.image)
        alpha, beta, gamma, window_size, iterations, sigma = self.get_contour_parameters()
        self.snake_model = ActiveContourProcessor(input_image_copy, alpha, beta, gamma, window_size, iterations, sigma)
        self.snake_model.update_snake()
        print("done update snake loop")
        final_snake, init_snake = self.snake_model.get_snake()
        
        scene = self.input_image.display_image()

        # Draw initial snake (Blue)
        path_init = QPainterPath()
        path_init.moveTo(init_snake[0, 0], init_snake[0, 1])
        for point in init_snake[1:]:
            path_init.lineTo(point[0], point[1])
        
        init_snake_item = QGraphicsPathItem(path_init)
        init_snake_item.setPen(QPen(Qt.blue, 1.7))
        scene.addItem(init_snake_item)

        # Draw final snake (Red)
        path_final = QPainterPath()
        path_final.moveTo(final_snake[0, 0], final_snake[0, 1])
        for point in final_snake[1:]:
            path_final.lineTo(point[0], point[1])
            
        for point in final_snake:
            marker = QGraphicsEllipseItem(point[0] - 2, point[1] - 2, 4, 4)  
            marker.setBrush(Qt.red)  
            scene.addItem(marker)    
        
        final_snake_item = QGraphicsPathItem(path_final)
        final_snake_item.setPen(QPen(Qt.red, 1.7))
        scene.addItem(final_snake_item)
        
        self.contour_output_frame.setScene(scene)
        
        # Compute perimeter
        final_snake_int = np.array(final_snake, dtype=np.int32)
        perimeter = cv2.arcLength(final_snake_int, closed=False)
        self.perimeter_label.setText(f"Perimeter: {perimeter:.2f} pixels")

        # Compute Area
        area = cv2.contourArea(final_snake_int)
        self.area_label.setText(f"Area: {area:.2f} square pixels")
    
    
    def get_contour_parameters(self):
        alpha = self.alpha_snake.value()
        beta = self.beta_snake.value()
        gamma = self.gamma_snake.value()
        window_size = self.window_size_snake.value()
        iterations = self.iterations_snake.value()
        sigma = self.sigma_snake.value()
        return alpha, beta, gamma, window_size, iterations, sigma

        

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())