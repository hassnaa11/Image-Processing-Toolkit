from PyQt5 import QtWidgets, QtCore, uic
import sys
from PyQt5.QtGui import *
from Image import Image
import numpy as np
import cv2

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        uic.loadUi('ui.ui', self)
        
        # Upload buttons
        self.original_image = None
        self.upload_button.clicked.connect(lambda: self.upload_image(1))
        self.input1_button.clicked.connect(lambda: self.upload_image(2))
        self.input2_button.clicked.connect(lambda: self.upload_image(3))
        
        # Noises combobox
        self.noises_combobox.setDisabled(True)
        self.noises_combobox.currentIndexChanged.connect(self.apply_noise)
        
        # Filters combobox
        self.filters_combobox.setDisabled(True)
        self.filters_combobox.currentIndexChanged.connect(self.apply_filter)
        
        # Sliders
        self.min_range_slider.valueChanged.connect(self.apply_noise)
        self.max_range_slider.valueChanged.connect(self.apply_noise)
        self.mean_slider.valueChanged.connect(self.apply_noise)
        self.sigma_slider.valueChanged.connect(self.apply_noise)
        self.probability_slider.valueChanged.connect(self.apply_noise)
        self.ratio_slider.valueChanged.connect(self.apply_noise)
        
        # Disable sliders initially
        self.min_range_slider.setDisabled(True)
        self.max_range_slider.setDisabled(True)
        
        self.show_hide_parameters('Uniform')

    def upload_image(self, key):
        self.file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select Image", "", "Images (*.png *.jpg *.jpeg *.bmp *.gif)"
        )
        if self.file_path:
            self.original_image = Image(self.file_path)  # Keep original image unchanged
            self.noisy_image = Image(self.file_path)  # Stores noise-modified image
            self.filtered_image = Image(self.file_path)  # Stores filter-modified image
            
            scene = self.original_image.display_image()
            if key == 1:
                self.input_image.setScene(scene)
                self.input_image.fitInView(scene.sceneRect(), QtCore.Qt.KeepAspectRatio)
                self.noises_combobox.setDisabled(False)
                self.filters_combobox.setDisabled(False)
                self.min_range_slider.setDisabled(False)
                self.max_range_slider.setDisabled(False)
                self.apply_noise()
                self.apply_filter()
            elif key == 2:
                self.input1_image.setScene(scene)
                self.input1_image.fitInView(scene.sceneRect(), QtCore.Qt.KeepAspectRatio)
            elif key == 3:
                self.input2_image.setScene(scene)
                self.input2_image.fitInView(scene.sceneRect(), QtCore.Qt.KeepAspectRatio)

    def apply_noise(self):
        """Resets to the original image, applies noise, and then reapplies the selected filter."""
        if not self.original_image:
            return

        # Reset noise-modified image to original
        self.noisy_image.image = np.copy(self.original_image.image)

        selected_noise = self.noises_combobox.currentText()
        self.show_hide_parameters(selected_noise)
        parameters = []

        if selected_noise == 'Uniform':
            min_range, max_range = self.min_range_slider.value(), self.max_range_slider.value()
            parameters = [min_range, max_range]
            self.min_range_label.setText(f"Minimum Range: {min_range}")
            self.max_range_label.setText(f"Maximum Range: {max_range}")

        elif selected_noise == 'Gaussian':
            mean, sigma = self.mean_slider.value(), self.sigma_slider.value()
            parameters = [mean, sigma]
            self.mean_label.setText(f"Mean: {mean}")
            self.sigma_label.setText(f"Sigma: {sigma}")

        elif selected_noise == 'Salt & Pepper':
            ratio, probability = self.ratio_slider.value(), self.probability_slider.value()
            parameters = [ratio / 10, probability / 10]
            self.ratio_label.setText(f"Ratio: {ratio / 10}")
            self.probability_label.setText(f"Probability: {probability / 10}")

        # Apply noise only if selected
        if selected_noise != "None":
            self.noisy_image.add_noise(selected_noise, parameters)

        # Apply the filter on top of the noisy image
        self.apply_filter()

    def apply_filter(self):
        """Resets the image to the noisy version, applies the selected filter, and updates display."""
        if not self.original_image:
            return

        # Reset filtered image to the noisy version
        self.filtered_image.image = np.copy(self.noisy_image.image)

        selected_filter = self.filters_combobox.currentText()
        img_array = self.filtered_image.image  # Extract NumPy array

        if selected_filter == 'Average':
            filtered_image = cv2.blur(img_array, (5, 5))
        elif selected_filter == 'Gaussian':
            filtered_image = cv2.GaussianBlur(img_array, (5, 5), 0)
        elif selected_filter == 'Median':
            filtered_image = cv2.medianBlur(img_array, 5)
        else:
            filtered_image = img_array  # No filter selected

        self.filtered_image.image = filtered_image  # Update the filtered image

        # Display the final image (Noise + Filter)
        scene = self.filtered_image.display_image()
        self.output_image.setScene(scene)
        self.output_image.fitInView(scene.sceneRect(), QtCore.Qt.KeepAspectRatio)

    def show_hide_parameters(self, selected_noise):
        """Show or hide sliders based on selected noise type."""
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

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
