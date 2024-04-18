import cv2
import numpy as np
from PyQt5.QtCore import pyqtSlot, Qt
from PyQt5.QtGui import QImage, QPixmap, QIcon
from PyQt5.QtWidgets import (
    QWidget,
    QGridLayout,
    QLabel,
    QPushButton,
    QHBoxLayout,
    QMessageBox,
    QSizePolicy
)
import configparser
import time
from dehazing.utils import CameraStream



class RealtimeFrame(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Create layout
        cctv_layout = QGridLayout(self)
        cctv_layout.setAlignment(Qt.AlignCenter)
        cctv_layout.setContentsMargins(0, 0, 0, 0)  # Remove any margin

        # CCTV Frames
        self.cctv_frame = QLabel()
        self.cctv_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.cctv_frame.setContentsMargins(0, 0, 0, 0)  # Remove any margin
        self.cctv_frame.setStyleSheet(
            "border: 1px solid gray; border-radius: 10px; background-color: black;"
        )

        # Add widgets to the layout
        cctv_layout.addWidget(self.cctv_frame, 1, 1)

        self.start_button = QPushButton("Start")
        self.start_button.setCheckable(True)  # Make it a toggle button
        # Connect the button's toggled signal to the start_camera_stream method
        self.start_button.toggled.connect(self.start_camera_stream)

        self.screenshot_button = QPushButton("Screenshot")
        self.screenshot_button.clicked.connect(self.take_screenshot)

        # Create the settings button
        manage_camera_button = QPushButton()
        manage_camera_button.setIcon(QIcon('gui/assets/icons/settings.svg'))

        manage_camera_button.setToolTip("Manage Cameras")
        manage_camera_button.setStyleSheet(
            """
            QPushButton {
                background-color: #fff;
                border: 1px solid gray;
                border-radius: 10px;
                padding: 5px;
            }
            QPushButton:hover {
                background-color: #eeeeee;
            }
        """
        )
        manage_camera_button.clicked.connect(self.parent.show_options_popup)

        # Create a horizontal layout and add the start button and the settings button to it
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(manage_camera_button)
        button_layout.addWidget(self.screenshot_button)

        # Add the button layout to the grid layout
        cctv_layout.addLayout(button_layout, 2, 0, 1, 2, Qt.AlignCenter)

        self.camera_stream = None

    @pyqtSlot()
    def start_camera_stream(self):
        config = configparser.ConfigParser()
        config.read("settings.cfg")
        if "DEFAULT" in config and "input" in config["DEFAULT"]:
            ip_address = config["DEFAULT"]["input"]
        else:
            ip_address = "0"

        if ip_address == "":
            QMessageBox.warning(
                self, "Error", "Please, set the camera IP address in the settings."
            )
            return

        if self.start_button.isChecked():
            # Create an instance of the CameraStreamThread class
            self.camera_stream = CameraStream(ip_address)
            self.camera_stream.frame_processed.connect(self.update_cctv_frame)
            self.camera_stream.start()
            self.start_button.setText("Stop")
        else:
            # Stop the camera stream if the button is unchecked
            self.start_button.setText("Start")
            if self.camera_stream:
                self.camera_stream.stop()
                self.camera_stream = None

    @pyqtSlot(np.ndarray)
    def update_cctv_frame(self, cv_img):
        scaled_image = (cv_img * 255.0).clip(0, 255).astype(np.uint8)
        rgb_image = cv2.cvtColor(scaled_image, cv2.COLOR_BGR2RGB)
        qimage = QImage(
            rgb_image.data, rgb_image.shape[1], rgb_image.shape[0], rgb_image.shape[1] * 3, QImage.Format_RGB888
        )
        # Convert the image to QPixmap
        pixmap = QPixmap.fromImage(qimage)

        # Scale the pixmap while keeping the aspect ratio
        pixmap = pixmap.scaled(
            self.cctv_frame.width(), self.cctv_frame.height(), Qt.KeepAspectRatio
        )

        # Update the camera feed label with the scaled pixmap
        self.cctv_frame.setPixmap(pixmap)
        self.cctv_frame.setAlignment(Qt.AlignCenter)

    def take_screenshot(self):
        # Capture the current frames
        if self.camera_stream:
            original_frame = self.camera_stream.img
            processed_frame = self.camera_stream.frame
            timestamp = time.time()
            # Save the original and processed frames as images
            cv2.imwrite(f"original_screenshot_{timestamp}.png", original_frame)
            cv2.imwrite(f"processed_screenshot_{timestamp}.png", processed_frame * 255)
            print("Screenshots saved successfully.")
        else:
            QMessageBox.warning(self, "Error", "No camera stream is running.")