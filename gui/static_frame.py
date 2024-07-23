import cv2
import numpy as np
import sys
from PyQt5.QtCore import Qt, QSize, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage, QColor, QPen, QPainter
from PyQt5.QtWidgets import (
    QWidget, QGridLayout, QLabel, QPushButton, QSizePolicy, QMessageBox,
    QFileDialog, QApplication, QHBoxLayout, QVBoxLayout, QProgressBar
)

from dehazing.dehazing import DehazingCPU

class ImageComparisonSlider(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.image1 = None
        self.image2 = None
        self.slider_position = 50
        self.is_mouse_pressed = False
        self.setMouseTracking(True)

    def setImages(self, image1, image2):
        self.image1 = image1
        self.image2 = image2
        self.updateImage()

    def updateImage(self):
        if self.image1 is None or self.image2 is None:
            return

        label_size = self.size()
        scale_factor = min(label_size.width() / self.image1.width(), label_size.height() / self.image1.height())
        
        scaled_size = QSize(int(self.image1.width() * scale_factor), int(self.image1.height() * scale_factor))
        
        offset_x = (label_size.width() - scaled_size.width()) // 2
        offset_y = (label_size.height() - scaled_size.height()) // 2

        scaled_image1 = self.image1.scaled(scaled_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        scaled_image2 = self.image2.scaled(scaled_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        result = QPixmap(label_size)
        result.fill(Qt.transparent)
        painter = QPainter(result)

        painter.drawPixmap(offset_x, offset_y, scaled_image1)

        overlay_width = self.slider_position * scaled_image1.width() // 100

        painter.drawPixmap(offset_x, offset_y, overlay_width, scaled_image1.height(), scaled_image2, 0, 0, overlay_width, scaled_image2.height())

        # Draw a more visible slider line
        pen = QPen(QColor(255, 255, 255))
        pen.setWidth(3)
        painter.setPen(pen)
        painter.drawLine(offset_x + overlay_width, offset_y, offset_x + overlay_width, offset_y + scaled_image1.height())

        # Add labels for original and processed images
        font = painter.font()
        font.setPointSize(10)
        painter.setFont(font)
        painter.drawText(offset_x + 10, offset_y + 20, "Original")
        painter.drawText(offset_x + scaled_image1.width() - 80, offset_y + 20, "Processed")

        painter.end()

        self.setPixmap(result)

    def resizeEvent(self, event):
        self.updateImage()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.is_mouse_pressed = True
            self.updateSliderPosition(event.pos().x())

    def mouseMoveEvent(self, event):
        if self.is_mouse_pressed:
            self.updateSliderPosition(event.pos().x())

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.is_mouse_pressed = False

    def updateSliderPosition(self, x):
        if self.image1 is None or self.image2 is None:
            return

        label_size = self.size()
        scale_factor = min(label_size.width() / self.image1.width(), label_size.height() / self.image1.height())
        scaled_width = int(self.image1.width() * scale_factor)
        offset_x = (label_size.width() - scaled_width) // 2
        
        self.slider_position = max(0, min(100, (x - offset_x) * 100 // scaled_width))
        self.updateImage()

    def enterEvent(self, event):
        self.setCursor(Qt.SplitHCursor)

    def leaveEvent(self, event):
        self.setCursor(Qt.ArrowCursor)

class ProcessingThread(QThread):
    update_progress = pyqtSignal(int)
    processing_complete = pyqtSignal(np.ndarray)

    def __init__(self, image_path):
        super().__init__()
        self.image_path = image_path

    def run(self):
        try:
            image = cv2.imread(self.image_path)
            dehazing_instance = DehazingCPU(progress_callback=self.update_progress.emit)
            processed_image = dehazing_instance.image_processing(image)
            self.processing_complete.emit(processed_image)
        except Exception as e:
            print(f"Error in processing thread: {e}")

class StaticFrame(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.processed_image = None
        self.image_path = None

        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout(self)

        # Image Comparison Slider
        self.comparison_slider = ImageComparisonSlider(self)
        self.comparison_slider.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.comparison_slider.setStyleSheet(
            "border: 2px solid gray; border-radius: 10px; background-color: #f0f0f0;"
        )

        # Buttons layout
        button_layout = QHBoxLayout()

        # Load Image button
        self.btn_load_image = self.create_styled_button("Load Image", self.load_image)
        button_layout.addWidget(self.btn_load_image)

        # Save Image button
        self.btn_save_image = self.create_styled_button("Save Image", self.save_image)
        self.btn_save_image.setEnabled(False)
        button_layout.addWidget(self.btn_save_image)

        # Start Processing button
        self.btn_start_processing = self.create_styled_button("Start Processing", self.start_processing)
        self.btn_start_processing.setEnabled(False)
        button_layout.addWidget(self.btn_start_processing)

        # Progress bar
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setVisible(False)

        # Add widgets to the main layout
        main_layout.addWidget(self.comparison_slider)
        main_layout.addLayout(button_layout)
        main_layout.addWidget(self.progress_bar)

        # Status label
        self.status_label = QLabel("No image loaded", self)
        self.status_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.status_label)

    def create_styled_button(self, text, connection):
        btn = QPushButton(text)
        btn.setToolTip(text)
        btn.clicked.connect(connection)
        btn.setStyleSheet(
            """
            QPushButton {
                background-color: #4CAF50;
                border: none;
                color: white;
                padding: 10px 20px;
                text-align: center;
                text-decoration: none;
                font-size: 16px;
                margin: 4px 2px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
            """
        )
        return btn

    def load_image(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        self.image_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Input Image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp);;All Files (*)",
            options=options,
        )
        if self.image_path:
            try:
                pixmap = QPixmap(self.image_path)
                if pixmap.isNull():
                    raise ValueError("Failed to load image")
                self.comparison_slider.setImages(pixmap, pixmap)
                self.btn_start_processing.setEnabled(True)
                self.btn_save_image.setEnabled(False)
                self.status_label.setText("Image loaded successfully")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load image: {str(e)}")
                self.status_label.setText("Failed to load image")

    def save_image(self):
        if self.processed_image is None:
            QMessageBox.information(self, "Error", "No processed image to save.")
            return

        output_image_path, _ = QFileDialog.getSaveFileName(
            self, "Save Processed Image", "", "Images (*.png *.jpg *.jpeg *.bmp)"
        )

        if output_image_path:
            try:
                cv2.imwrite(output_image_path, cv2.cvtColor(self.processed_image, cv2.COLOR_RGB2BGR))
                QMessageBox.information(self, "Success", "Image saved successfully.")
                self.status_label.setText("Image saved successfully")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save image: {str(e)}")
                self.status_label.setText("Failed to save image")

    def start_processing(self):
        if self.image_path is None:
            QMessageBox.information(self, "Error", "Please load an image first!")
            return

        self.btn_start_processing.setEnabled(False)
        self.btn_load_image.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.status_label.setText("Processing image...")

        self.processing_thread = ProcessingThread(self.image_path)
        self.processing_thread.update_progress.connect(self.update_progress)
        self.processing_thread.processing_complete.connect(self.processing_finished)
        self.processing_thread.start()

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def processing_finished(self, processed_image):
        self.processed_image = processed_image

        # Scale the processed image values to the range [0, 255] without data loss
        scaled_image = (processed_image * 255.0).clip(0, 255).astype(np.uint8)

        # Convert the NumPy array (BGR format) to an RGB image
        rgb_image = cv2.cvtColor(scaled_image, cv2.COLOR_BGR2RGB)

        # Create a QImage from the RGB image
        qimage = QImage(
            rgb_image.data, rgb_image.shape[1], rgb_image.shape[0], rgb_image.shape[1] * 3, QImage.Format_RGB888
        )
        
        # Convert the QImage to a QPixmap
        processed_pixmap = QPixmap.fromImage(qimage)
        
        # Update the comparison slider with the original and processed images
        original_pixmap = QPixmap(self.image_path)
        self.comparison_slider.setImages(processed_pixmap, original_pixmap)

        self.btn_start_processing.setEnabled(True)
        self.btn_load_image.setEnabled(True)
        self.btn_save_image.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_label.setText("Image processing complete") 