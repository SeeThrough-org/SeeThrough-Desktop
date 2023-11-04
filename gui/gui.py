import time
import cv2
import numpy as np
from PyQt5.QtCore import (Qt, QSize)
from PyQt5.QtGui import (QIcon, QPixmap, QImage)
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QPushButton, QVBoxLayout, QStackedWidget, QSizePolicy,
                             QGridLayout, QHBoxLayout, QMessageBox, QLabel, QMenu, QDialog, QFileDialog, QVBoxLayout, QLineEdit, QDialogButtonBox)
from dehazing.dehazing import dehazing
from dehazing.utils import CameraStream
import configparser
import os
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = r'path/to/qt/plugins/platforms'


class GUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SUVIDE")
        self.setGeometry(100, 100, 1440, 900)
        self.setMinimumSize(1280, 720)  # Minimum width and height
        self.setWindowIcon(QIcon('logo.svg'))
        self.setStyleSheet("QMainWindow {background-color: #fff;}")

        # Create a stacked widget to manage frames
        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)

        # Create and add frames to the stacked widget
        self.stacked_widget.addWidget(self.realtime_frames())
        self.stacked_widget.addWidget(self.static_dehazing_frames())

        # Create navbar and connect buttons to frame switching
        navbar = self.navbar()
        navbar_buttons = navbar.findChildren(QPushButton)
        for button in navbar_buttons:
            button.clicked.connect(self.switch_frame)

        # Create switch_framea layout for the central widget and add the stacked widget and navbar
        central_layout = QVBoxLayout()
        central_layout.addWidget(self.stacked_widget)
        central_layout.addWidget(navbar)

        central_widget = QWidget()
        central_widget.setLayout(central_layout)
        self.setCentralWidget(central_widget)

        # call dehazing class
        # self.processed_image = None

    def load_image(self):
        # Define the action when the "Input Image" button is clicked
        # For example, open a file dialog to select an input image
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        input_image_path, _ = QFileDialog.getOpenFileName(
            self, "Select Input Image", "", "Images (*.png *.jpg *.jpeg *.bmp);;All Files (*)", options=options)

        if input_image_path:
            # Read the image using OpenCV
            open_image = cv2.imread(input_image_path)

            if open_image is not None:
                dehazing_instance = dehazing()
                self.processed_image = dehazing_instance.image_processing(
                    open_image)

                pixmap = QPixmap(input_image_path)
                self.InputFile.setPixmap(pixmap)

                # Scale the processed image values to the range [0, 255] without data loss
                scaled_image = (self.processed_image *
                                255.0).clip(0, 255).astype(np.uint8)

                # Convert the NumPy array (BGR format) to an RGB image
                rgb_image = cv2.cvtColor(
                    scaled_image, cv2.COLOR_BGR2RGB)

                # Create a QImage from the RGB image
                qimage = QImage(rgb_image.data, rgb_image.shape[1],
                                rgb_image.shape[0], rgb_image.shape[1] * 3, QImage.Format_BGR888).rgbSwapped()

                # Convert the QImage to a QPixmap
                pixmap = QPixmap(qimage)
                self.OutputFile.setPixmap(pixmap)
            else:
                print("Error: Unable to open the selected image.")
        else:
            print("No image selected.")

    def save_image(self, image):
        """Save the image to the specified path."""
        if image is None:
            self.show_info_dialog("Make sure to load an image first.")
            return
        output_image_path, _ = QFileDialog.getSaveFileName(
            self, "Save Processed Image", "", "Images (*.png *.jpg *.jpeg *.bmp)")

        if output_image_path:
            cv2.imwrite(output_image_path, (image * 255))
            print("Image saved to:", output_image_path)

    def navbar(self):
        # Create a widget for the navigation bar
        navbar = QWidget()
        navbar.setFixedHeight(64)

        # Label for the logo
        logo = QLabel('seeThrough')
        logo.setStyleSheet('''
            QLabel {
                font-family: "Montserrat";
                font-size: 20px;
                font-weight: bold;
                color: #191919;
            }
        ''')

        # Create buttons for frame switching
        btn_realtime_dehazing = QPushButton('Realtime Dehazing')
        btn_realtime_dehazing.setStyleSheet('''
            QPushButton {
                background-color: #fff;
                border: 1px solid gray; /* Add a border */
                border-radius: 10px;
                padding: 10px 60px; /* Adjust padding */
                font-size: 13px; /* Increase font size */
            }

            QPushButton:hover {
                background-color: #373030; /* Change background color on hover */
                color: #fff; /* Change text color on hover */
            }
        ''')
        btn_realtime_dehazing.clicked.connect(
            lambda: self.stacked_widget.setCurrentIndex(0)
        )

        btn_static_dehazing = QPushButton('Static Dehazing')
        btn_static_dehazing.setStyleSheet('''
            QPushButton {
                background-color: #fff;
                border: 1px solid gray; /* Add a border */
                border-radius: 10px;
                padding: 10px 60px; /* Adjust padding */
                font-size: 13px; /* Increase font size */
            }

            QPushButton:hover {
                background-color: #373030; /* Change background color on hover */
                color: #fff; /* Change text color on hover */
            }
        ''')
        btn_static_dehazing.clicked.connect(
            lambda: self.stacked_widget.setCurrentIndex(1))
        btn_video_dehazing = QPushButton('Video Dehazing')
        btn_video_dehazing.setStyleSheet('''
            QPushButton {
                background-color: #fff;
                border: 1px solid gray; /* Add a border */
                border-radius: 10px;
                padding: 10px 60px; /* Adjust padding */
                font-size: 13px; /* Increase font size */
            }

            QPushButton:hover {
                background-color: #373030; /* Change background color on hover */
                color: #fff; /* Change text color on hover */
            }
        ''')
        btn_exit = QPushButton()
        # btn_exit icon
        btn_exit.setIcon(QIcon('./images/exit.svg'))
        btn_exit.setIconSize(QSize(32, 32))
        btn_exit.clicked.connect(self.confirm_exit)

        # Add buttons to the navbar
        layout = QHBoxLayout(navbar)
        layout.addWidget(logo, alignment=Qt.AlignLeft)
        layout.addWidget(btn_realtime_dehazing, )
        layout.addWidget(btn_static_dehazing, )
        layout.addWidget(btn_video_dehazing, )
        layout.addWidget(btn_exit, alignment=Qt.AlignRight)

        return navbar

    def switch_frame(self):
        frame_text = self.sender().text()
        if frame_text == 'Realtime Dehazing':
            self.stacked_widget.setCurrentIndex(0)
        elif frame_text == 'Static Dehazing':
            self.stacked_widget.setCurrentIndex(1)

    def show_options_popup(self):
        options_popup = QDialog()
        options_popup.setWindowTitle("Camera Options")
        options_popup.setWindowFlags(
            options_popup.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        options_popup.setFixedWidth(320)
        options_popup.setFixedHeight(240)

        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)  # Add padding to the dialog

        # Add a title label
        title_label = QLabel("<h2>Camera Options</h2>")
        layout.addWidget(title_label)

        # Create labels and input fields for Camera Name and IP Address
        camera_name_label = QLabel("Camera Name:")
        layout.addWidget(camera_name_label)

        camera_name = QLineEdit()
        camera_name.setPlaceholderText("Enter camera name")
        layout.addWidget(camera_name)

        input_label = QLabel("IP Address:")
        layout.addWidget(input_label)

        input_field = QLineEdit()
        input_field.setPlaceholderText("Enter IP address")
        layout.addWidget(input_field)

        # Add a button box with custom styling
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            options_popup)
        buttons.accepted.connect(options_popup.accept)
        buttons.rejected.connect(options_popup.reject)
        layout.addWidget(buttons)

        # Apply custom styling using CSS
        options_popup.setStyleSheet("""
            QDialog {
                background-color: #F5F5F5;
            }
            QLabel {
                font-size: 18px;
            }
            QLineEdit {
                padding: 8px;
                font-size: 16px;
                border: 2px solid #000;
                border-radius: 4px;
            }
            QPushButton {
                padding: 8px 16px;
                font-size: 16px;
                background-color: #007ACC;
                color: white;
                border: none;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #005FAA;
            }
        """)

        options_popup.setLayout(layout)

        # Load settings with error handling
        try:
            config = configparser.ConfigParser()
            config.read('settings.cfg')
            if 'DEFAULT' in config and 'input' in config['DEFAULT']:
                input_field.setText(config['DEFAULT']['input'])
            if 'DEFAULT' in config and 'camera_name' in config['DEFAULT']:
                camera_name.setText(config['DEFAULT']['camera_name'])
        except (FileNotFoundError, configparser.Error) as e:
            print(f"Error loading settings: {e}")

        result = options_popup.exec_()

        if result == QDialog.Accepted:
            # Save settings with error handling
            try:
                config = configparser.ConfigParser()
                config.read('settings.cfg')
                if 'DEFAULT' not in config:
                    config['DEFAULT'] = {}
                config['DEFAULT']['input'] = input_field.text()
                config['DEFAULT']['camera_name'] = camera_name.text()
                with open('settings.cfg', 'w') as configfile:
                    config.write(configfile)
            except (FileNotFoundError, configparser.Error) as e:
                print(f"Error saving settings: {e}")

    def static_dehazing_frames(self):
        # Create the widget
        widget_static = QWidget()
        widget_static.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Create layout
        layout = QGridLayout(widget_static)
        layout.setAlignment(Qt.AlignCenter)
        layout.setContentsMargins(0, 0, 0, 0)  # Remove any margin

        # Input File (Only Images and Videos)
        self.InputFile = QLabel()  # Use QLabel to display an image
        self.InputFile.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.InputFile.setContentsMargins(0, 0, 0, 0)  # Remove any margin
        self.InputFile.setStyleSheet(
            "border: 1px solid gray; border-radius: 10px; background-color: green;")

        # Add the "Select Image" button
        btn_load_image = QPushButton("Load Image")
        btn_load_image.setIcon(QIcon('./images/camerasettings.svg'))
        btn_load_image.setToolTip('Load Image')
        btn_load_image.clicked.connect(self.load_image)

        # Apply button styling
        btn_load_image.setStyleSheet('''
            QPushButton {
                background-color: #fff;
                border: 1px solid gray;
                border-radius: 10px;
                padding: 15px; /* Adjust the padding as needed */
            }
            QPushButton:hover {
                background-color: #eeeeee;
            }
        ''')

        # Add the "Load Image" button
        layout.addWidget(btn_load_image, 1, 0)
        # Add widgets to the layout
        layout.addWidget(self.InputFile, 0, 0)

        # Input File (Only Images and Videos)
        self.OutputFile = QLabel()
        self.OutputFile.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.OutputFile.setContentsMargins(0, 0, 0, 0)  # Remove any margin
        self.OutputFile.setStyleSheet(
            "border: 1px solid gray; border-radius: 10px; background-color: black;")

        # Add widgets to the layout
        layout.addWidget(self.OutputFile, 0, 1)

        # Add the "Save Image" button
        btn_save_image = QPushButton("Save Image")
        btn_save_image.setIcon(QIcon('./images/camerasettings.svg'))
        btn_save_image.setToolTip('Save Image')
        btn_save_image.clicked.connect(
            lambda: self.save_image(self.processed_image))

        # Apply button styling
        btn_save_image.setStyleSheet('''
            QPushButton {
                background-color: #fff;
                border: 1px solid gray;
                border-radius: 10px;
                padding: 15px; /* Adjust the padding as needed */
            }
            QPushButton:hover {
                background-color: #eeeeee;
            }
        ''')

        layout.addWidget(btn_save_image, 1, 1)  # Add the "Save Image" button

        # Set equal stretch factors for the columns
        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(1, 1)

        return widget_static

    def realtime_frames(self):
        # Create widget
        widget_rt = QWidget()
        widget_rt.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Create layout
        cctv_layout = QGridLayout(widget_rt)
        cctv_layout.setAlignment(Qt.AlignCenter)
        cctv_layout.setContentsMargins(0, 0, 0, 0)  # Remove any margin

        # CCTV Frames

        cctv_frame = QWidget()
        cctv_frame.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding)

        cctv_frame.setContentsMargins(0, 0, 0, 0)  # Remove any margin
        cctv_frame.setStyleSheet(
            "border: 1px solid gray; border-radius: 10px; background-color: #fff;")

        # I want to add a label here that will display the camera name
        label = QLabel("Camera Name")
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("font-size: 20px; font-weight: bold;")
        label.setContentsMargins(0, 0, 0, 0)  # Remove any margin

        # You can create an image label to display the camera feed
        image_label = QLabel()
        image_label.setAlignment(Qt.AlignCenter)
        image_label.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding)
        image_label.setContentsMargins(0, 0, 0, 0)

        # Add widgets to the layout
        cctv_layout.addWidget(cctv_frame, 1, 1)
        cctv_layout.addWidget(label, 1, 1)
        cctv_layout.addWidget(image_label, 1, 1)

        # Add the manage_camera_button here
        manage_camera_button = QPushButton("Manage Cameras")
        manage_camera_button.setIcon(QIcon('./images/camerasettings.svg'))
        manage_camera_button.setToolTip('Add Camera')
        manage_camera_button.clicked.connect(self.show_options_popup)

        # Apply button styling
        manage_camera_button.setStyleSheet('''
            QPushButton {
                background-color: #fff;
                border: 1px solid gray;
                border-radius: 10px;
                padding: 15px; /* Adjust the padding as needed */
            }
            QPushButton:hover {
                background-color: #eeeeee;
            }
        ''')

        # Add button to the layout
        cctv_layout.addWidget(manage_camera_button, 2, 0, 1, 2)

        return widget_rt

    def confirm_exit(self):
        reply = QMessageBox.question(None, 'Message', "Are you sure to quit?", QMessageBox.Yes | QMessageBox.No,
                                     QMessageBox.No)
        if reply == QMessageBox.Yes:
            print("Exit YES")
            QApplication.quit()
        elif reply == QMessageBox.No:
            print("Exit NO")
