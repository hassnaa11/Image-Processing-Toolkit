def handleRadio(self, label):
        self.label = label
        self.worker.update_label(self.label)

    def handleHough(self):
        if self.label == "lines":
            if self.image is None:
                print("Error: Could not read the image.")
                return
            
            # To change the UI labels
            self.worker.update_label("lines")
            
            lowThreshold, highThreshold, votes,_ = self.worker.get_slider_values()
            result_image = Hough.detect_lines(self.image, lowThreshold, highThreshold, votes)

        elif self.label == "circles":
            self.worker.update_label("circles")
            minRadius, maxRadius, _, _ = self.worker.get_slider_values()
            result_image = Hough.hough_circles(self.image, min_radius=minRadius, max_radius=maxRadius)
            
        elif self.label == "ellipses":
            self.worker.update_label("ellipses")
            lowThreshold, highThreshold, minAxis, maxAxis = self.worker.get_slider_values()
            result_image = Hough.hough_ellipses(self.image, low_threshold=lowThreshold, high_threshold=highThreshold, min_axis=minAxis, max_axis=maxAxis)


        # Convert to QImage before setting as QPixmap
        qimage = self.convert_numpy_to_qimage(result_image)
        self.resultImage_hough.setPixmap(QPixmap.fromImage(qimage))

    def clear_hough(self):
        self.worker.clear()
        self.inputImage_hough.setPixmap(QPixmap.fromImage(self.q_image))
        self.resultImage_hough.setPixmap(QPixmap.fromImage(self.q_image))