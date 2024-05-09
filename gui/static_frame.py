import cv2
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage, QIcon
from PyQt5.QtWidgets import (
    QWidget,
    QGridLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QMessageBox,
    QFileDialog,
)

from dehazing.dehazing import DehazingCPU


class StaticFrame(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Create layout
        layout = QGridLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        layout.setContentsMargins(0, 0, 0, 0)  # Remove any margin

        # Input File (Only Images and Videos)
        self.InputFile = QLabel()  # Use QLabel to display an image
        self.InputFile.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.InputFile.setContentsMargins(0, 0, 0, 0)  # Remove any margin
        self.InputFile.setStyleSheet(
            "border: 1px solid gray; border-radius: 10px; background-color: black;"
        )

        # Add the "Select Image" button
        btn_load_image = QPushButton("Load Image")
        btn_load_image.setToolTip("Load Image")
        btn_load_image.clicked.connect(self.load_image)

        # Apply button styling
        btn_load_image.setStyleSheet(
            """
            QPushButton {
                background-color: #fff;
                border: 1px solid gray;
                border-radius: 10px;
                padding: 15px; /* Adjust the padding as needed */
            }
            QPushButton:hover {
                background-color: #eeeeee;
            }
        """
        )

        # Add the "Load Image" button
        layout.addWidget(btn_load_image, 1, 0)
        # Add widgets to the layout
        layout.addWidget(self.InputFile, 0, 0)

        # Input File (Only Images and Videos)
        self.OutputFile = QLabel()
        self.OutputFile.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.OutputFile.setContentsMargins(0, 0, 0, 0)  # Remove any margin
        self.OutputFile.setStyleSheet(
            "border: 1px solid gray; border-radius: 10px; background-color: black;"
        )

        # Add widgets to the layout
        layout.addWidget(self.OutputFile, 0, 1)

        # Add the "Save Image" button
        btn_save_image = QPushButton("Save Image")
        btn_save_image.setToolTip("Save Image")
        btn_save_image.clicked.connect(self.save_image)

        # Apply button styling
        btn_save_image.setStyleSheet(
            """
            QPushButton {
                background-color: #fff;
                border: 1px solid gray;
                border-radius: 10px;
                padding: 15px; /* Adjust the padding as needed */
            }
            QPushButton:hover {
                background-color: #eeeeee;
            }
        """
        )

        layout.addWidget(btn_save_image, 1, 1)  # Add the "Save Image" button
        btn_start_processing = QPushButton("Start Processing")
        btn_start_processing.clicked.connect(self.start_processing)

        layout.addWidget(btn_start_processing, 2, 0, 1, 2)
        # Set equal stretch factors for the columns
        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(1, 1)

        self.processed_image = None
        self.image_path = None

    def load_image(self):
        # Define the action when the "Input Image" button is clicked
        # For example, open a file dialog to select an input image
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        self.image_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Input Image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp);;All Files (*)",
            options=options,
        )
        pixmap = QPixmap(self.image_path)
        pixmap = pixmap.scaled(self.InputFile.width(), self.InputFile.height(), Qt.KeepAspectRatio)
        self.InputFile.setPixmap(pixmap)
        self.InputFile.setAlignment(Qt.AlignCenter)

    def save_image(self):
        """Save the image to the specified path."""
        if self.processed_image is None:
            QMessageBox.information(self, "Error", "No image to save.")
            return

        output_image_path, _ = QFileDialog.getSaveFileName(
            self, "Save Processed Image", "", "Images (*.png *.jpg *.jpeg *.bmp)"
        )

        if output_image_path:
            cv2.imwrite(output_image_path, (self.processed_image * 255))
            QMessageBox.information(self, "Success", "Image saved successfully.")
            return

    def start_processing(self):
        try:
            if self.image_path is None:
                QMessageBox.information(self, "Error", "Please, Load an Image First!")
                return
            image = cv2.imread(self.image_path)
            dehazing_instance = DehazingCPU()
            self.processed_image = dehazing_instance.image_processing(image)
            pixmap = QPixmap(self.image_path)
            pixmap = pixmap.scaled(self.InputFile.width(), self.InputFile.height(), Qt.KeepAspectRatio)
            self.InputFile.setPixmap(pixmap)
            self.InputFile.setAlignment(Qt.AlignCenter)

            # Scale the processed image values to the range [0, 255] without data loss
            scaled_image = (self.processed_image * 255.0).clip(0, 255).astype(np.uint8)

            # Convert the NumPy array (BGR format) to an RGB image
            rgb_image = cv2.cvtColor(scaled_image, cv2.COLOR_BGR2RGB)

            # Create a QImage from the RGB image
            qimage = QImage(
                rgb_image.data, rgb_image.shape[1], rgb_image.shape[0], rgb_image.shape[1] * 3, QImage.Format_BGR888
            ).rgbSwapped()
            qimage = qimage.scaled(self.OutputFile.width(), self.OutputFile.height(), Qt.KeepAspectRatio)
            # Convert the QImage to a QPixmap
            pixmap = QPixmap(qimage)
            self.OutputFile.setPixmap(pixmap)
            self.OutputFile.setAlignment(Qt.AlignCenter)
        except Exception as e:
            QMessageBox.information(self, "Error", f"Error processing image: {e}")