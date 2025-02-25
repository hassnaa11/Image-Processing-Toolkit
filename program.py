from PyQt5 import QtWidgets, QtCore, uic   # Added uic import
from PyQt5 import QtWidgets, QtGui, QtCore, uic   # Added uic import

import sys
from PyQt5.QtGui import *
from Image import Image

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        uic.loadUi('ui.ui', self)
        self.upload_button.clicked.connect(lambda:self.upload_image(1))
        self.input1_button.clicked.connect(lambda:self.upload_image(2))
        self.input2_button.clicked.connect(lambda:self.upload_image(3))
        
        
    def upload_image(self, key):  
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select Image", "", "Images (*.png *.jpg *.jpeg *.bmp *.gif)"
        )
        if file_path:
            image = Image(file_path)
            scene = image.display_image()
            if key == 1:
                self.input_image.setScene(scene) 
                self.input_image.fitInView(scene.sceneRect(), QtCore.Qt.KeepAspectRatio)  
            elif key == 2:
                self.input1_image.setScene(scene)
                self.input1_image.fitInView(scene.sceneRect(), QtCore.Qt.KeepAspectRatio)
            elif key == 3:           
                self.input2_image.setScene(scene)
                self.input2_image.fitInView(scene.sceneRect(), QtCore.Qt.KeepAspectRatio)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())        