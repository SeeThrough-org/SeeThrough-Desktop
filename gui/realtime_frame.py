import cv2
import numpy as np
from PyQt5.QtCore import pyqtSlot, Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QIcon
from PyQt5.QtWidgets import (
    QWidget,
    QGridLayout,
    QLabel,
    QPushButton,
    QHBoxLayout,
    QMessageBox,
    QSizePolicy,
    QFileDialog,
    QVBoxLayout,
    QDialog,
    QLineEdit
)
import configparser
import time
import os
from dehazing.utils import CameraStream

class RealtimeFrame(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
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
        
        self.success_label = QLabel("")
        self.success_label.setStyleSheet("color: green; font-size: 14px;")
        cctv_layout.addWidget(self.success_label, 3, 0, 1, 2, Qt.AlignCenter)

        # FPS 
        self.fps_value = 0
        self.prev_time = 0 
        self.fps_label = QLabel("FPS:")
        self.fps_label.setStyleSheet("color: black; font-size: 14px; background-color: transparent;")
        cctv_layout.addWidget(self.fps_label, 0, 1, Qt.AlignRight)

        # Add widgets to the layout
        cctv_layout.addWidget(self.cctv_frame, 1, 1)

        self.start_button = self.create_styled_button("Start", self.start_camera_stream)
        self.start_button.setCheckable(True)  # Make it a toggle button

        self.screenshot_button = self.create_styled_button("Screenshot", self.take_screenshot)

        # Create the settings button
        manage_camera_button = self.create_styled_button("", self.show_options_popup, QIcon('assets/settings.svg'))
        manage_camera_button.setToolTip("Manage Cameras")

        # Create a horizontal layout and add the start button and the settings button to it
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(manage_camera_button)
        button_layout.addWidget(self.screenshot_button)
        cctv_layout.addLayout(button_layout, 2, 0, 1, 2, Qt.AlignCenter)

        self.camera_stream = None

        # Read screenshot directory from settings
        self.screenshot_dir = self.get_screenshot_directory()
        
    def create_styled_button(self, text, connection, icon=None):
        btn = QPushButton(text)
        btn.setToolTip(text)
        btn.clicked.connect(connection)
        if icon:
            btn.setIcon(icon)
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
            self.camera_stream.frame_processed.connect(self.update_cctv_frame, self.fps_value)
            self.camera_stream.start()
            self.start_button.setText("Stop")
        else:
            # Stop the camera stream if the button is unchecked
            self.start_button.setText("Start")
            if self.camera_stream:
                self.camera_stream.stop()
                self.camera_stream = None

    @pyqtSlot(np.ndarray, float)
    def update_cctv_frame(self, cv_img, fps_value):
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

        self.fps_label.setText(f"FPS: {fps_value:.2f}")
        # Update the camera feed label with the scaled pixmap
        self.cctv_frame.setPixmap(pixmap)
        self.cctv_frame.setAlignment(Qt.AlignCenter)

    def take_screenshot(self):
        # Capture the current frames
        if self.camera_stream:
            original_frame = self.camera_stream.img
            processed_frame = self.camera_stream.processed_frame
            timestamp = time.time()

            # Ensure the directory exists
            if not os.path.exists(self.screenshot_dir):
                os.makedirs(self.screenshot_dir)

            # Save the original and processed frames as images
            original_path = os.path.join(self.screenshot_dir, f"original_screenshot_{timestamp}.png")
            processed_path = os.path.join(self.screenshot_dir, f"processed_screenshot_{timestamp}.png")
            cv2.imwrite(original_path, original_frame)
            cv2.imwrite(processed_path, processed_frame * 255)
            self.success_label.setText("Screenshots saved successfully.")
            QTimer.singleShot(3000, lambda: self.success_label.setText(""))  # Clear after 3 seconds
        else:
            QMessageBox.warning(self, "Error", "No camera stream is running.")

    def show_options_popup(self):
        options_dialog = QDialog(self)
        options_dialog.setWindowTitle("Manage Cameras")

        layout = QVBoxLayout(options_dialog)

        # Add IP address input
        config = configparser.ConfigParser()
        config.read("settings.cfg")
        ip_address = config["DEFAULT"].get("input", "")

        self.ip_input = QLineEdit()
        self.ip_input.setPlaceholderText("Enter IP address")
        self.ip_input.setText(ip_address)

        layout.addWidget(QLabel("Camera IP Address:"))
        layout.addWidget(self.ip_input)

        # Add the directory selection button
        self.dir_button = QPushButton("Select Screenshot Directory")
        self.dir_button.clicked.connect(self.select_directory)

        self.dir_label = QLabel(f"Current Directory: {self.screenshot_dir}")

        layout.addWidget(self.dir_button)
        layout.addWidget(self.dir_label)

        save_button = QPushButton("Save")
        save_button.clicked.connect(self.save_options)

        layout.addWidget(save_button)

        options_dialog.setLayout(layout)
        options_dialog.exec_()

    def save_options(self):
        ip_address = self.ip_input.text()
        self.save_ip_address(ip_address)
        QMessageBox.information(self, "Settings Saved", "Settings have been saved successfully.")

    def select_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Directory", os.path.expanduser("~/Pictures/SeeThrough"))
        if directory:
            self.screenshot_dir = directory
            self.save_screenshot_directory(directory)
            self.dir_label.setText(f"Current Directory: {directory}")

    def get_screenshot_directory(self):
        config = configparser.ConfigParser()
        config.read("settings.cfg")
        if "DEFAULT" in config and "screenshot_directory" in config["DEFAULT"]:
            return config["DEFAULT"]["screenshot_directory"]
        return os.path.expanduser("~/Pictures/SeeThrough")

    def save_screenshot_directory(self, directory):
        config = configparser.ConfigParser()
        config.read("settings.cfg")
        if "DEFAULT" not in config:
            config["DEFAULT"] = {}
        config["DEFAULT"]["screenshot_directory"] = directory
        with open("settings.cfg", "w") as config_file:
            config.write(config_file)

    def save_ip_address(self, ip_address):
        config = configparser.ConfigParser()
        config.read("settings.cfg")
        if "DEFAULT" not in config:
            config["DEFAULT"] = {}
        config["DEFAULT"]["input"] = ip_address
        with open("settings.cfg", "w") as config_file:
            config.write(config_file)
