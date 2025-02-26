from PyQt5 import QtWidgets, QtCore, uic 
import sys
from PyQt5.QtGui import *
from Image import Image
import numpy as np
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
        self.noises_combobox.currentIndexChanged.connect(self.add_noise)
        
        # sliders
        self.min_range_slider.valueChanged.connect(self.add_noise)
        self.max_range_slider.valueChanged.connect(self.add_noise)
        self.mean_slider.valueChanged.connect(self.add_noise)
        self.sigma_slider.valueChanged.connect(self.add_noise)
        self.probability_slider.valueChanged.connect(self.add_noise)
        self.ratio_slider.valueChanged.connect(self.add_noise)
        self.min_range_slider.setDisabled(True)
        self.max_range_slider.setDisabled(True)
        self.show_hide_parameters('Uniform')
        
        
    def upload_image(self, key):  
        self.file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select Image", "", "Images (*.png *.jpg *.jpeg *.bmp *.gif)"
        )
        if self.file_path:
            self.original_image = Image(self.file_path)
            self.processed_image = Image(self.file_path)
            scene = self.original_image.display_image()
            if key == 1:
                self.input_image.setScene(scene) 
                self.input_image.fitInView(scene.sceneRect(), QtCore.Qt.KeepAspectRatio)  
                self.noises_combobox.setDisabled(False)
                self.min_range_slider.setDisabled(False)
                self.max_range_slider.setDisabled(False)
                self.add_noise()
            elif key == 2:
                self.input1_image.setScene(scene)
                self.input1_image.fitInView(scene.sceneRect(), QtCore.Qt.KeepAspectRatio)
            elif key == 3:           
                self.input2_image.setScene(scene)
                self.input2_image.fitInView(scene.sceneRect(), QtCore.Qt.KeepAspectRatio)

    
    def add_noise(self):
        if not self.original_image:
            return
        
        selected_noise = self.noises_combobox.currentText()
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
            
        self.processed_image.image = np.copy(self.original_image.image)      
        self.processed_image.add_noise(selected_noise, parameters)
        scene = self.processed_image.display_image()
        self.output_image.setScene(scene) 
        self.output_image.fitInView(scene.sceneRect(), QtCore.Qt.KeepAspectRatio)  
    
    
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


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())        