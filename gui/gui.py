import time
import cv2
import numpy as np
from PyQt5.QtCore import (Qt, QSize)
from PyQt5.QtGui import (QIcon, QPixmap, QImage)
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QPushButton, QVBoxLayout, QStackedWidget, QSizePolicy,
                             QGridLayout, QHBoxLayout, QMessageBox, QLabel, QMenu, QDialog, QFileDialog, QVBoxLayout, QLineEdit, QDialogButtonBox)
from dehazing.dehazing import dehazing
from dehazing.utils import CameraStream
from dehazing.utils import VideoProcessor
import configparser
import os
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = r'path/to/qt/plugins/platforms'


class GUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SeeThrough")
        self.setGeometry(100, 100, 1440, 900)
        self.setMinimumSize(1280, 720)  # Minimum width and height
        self.setWindowIcon(QIcon('gui/assets/icons/logo.svg'))
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
        # 0 is the index of Realtime Dehazing

        central_widget = QWidget()
        central_widget.setLayout(central_layout)
        self.setCentralWidget(central_widget)
        self.active_button = None  # Track the active button
        self.active_frame = 0
        self.processed_image = None

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
                pixmap = pixmap.scaled(
                    self.InputFile.width(), self.InputFile.height(), Qt.KeepAspectRatio)
                self.InputFile.setPixmap(pixmap)
                self.InputFile.setAlignment(Qt.AlignCenter)

                # Scale the processed image values to the range [0, 255] without data loss
                scaled_image = (self.processed_image *
                                255.0).clip(0, 255).astype(np.uint8)

                # Convert the NumPy array (BGR format) to an RGB image
                rgb_image = cv2.cvtColor(
                    scaled_image, cv2.COLOR_BGR2RGB)

                # Create a QImage from the RGB image
                qimage = QImage(rgb_image.data, rgb_image.shape[1],
                                rgb_image.shape[0], rgb_image.shape[1] * 3, QImage.Format_BGR888).rgbSwapped()
                qimage = qimage.scaled(
                    self.OutputFile.width(), self.OutputFile.height(), Qt.KeepAspectRatio)
                # Convert the QImage to a QPixmap
                pixmap = QPixmap(qimage)
                self.OutputFile.setPixmap(pixmap)
                self.OutputFile.setAlignment(Qt.AlignCenter)
            else:
                print("Error: Unable to open the selected image.")
        else:
            print("No image selected.")

    def save_image(self):
        """Save the image to the specified path."""
        if self.processed_image is None:
            QMessageBox.information(self, "Error", "No image to save.")
            return

        output_image_path, _ = QFileDialog.getSaveFileName(
            self, "Save Processed Image", "", "Images (*.png *.jpg *.jpeg *.bmp)")

        if output_image_path:
            cv2.imwrite(output_image_path, (self.processed_image * 255))
            QMessageBox.information(
                self, "Success", "Image saved successfully.")

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
        btn_realtime_dehazing.setObjectName(
            "realtime_button")  # Add an object name
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
        btn_static_dehazing.setObjectName("static_button")
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
        btn_video_dehazing.clicked.connect(self.video_dehazing)
        btn_exit = QPushButton()
        # btn_exit icon
        btn_exit.setIcon(QIcon('gui/assets/icons/exit.svg'))
        btn_exit.setIconSize(QSize(32, 32))
        btn_exit.setStyleSheet('''
            QPushButton {
                background-color: #fff;
                border: 1px solid gray;
                border-radius: 10px;
                padding: 5px;
            }
            QPushButton:hover {
                background-color: #eeeeee;
            }
        ''')
        btn_exit.clicked.connect(self.confirm_exit)

        # Add buttons to the navbar
        layout = QHBoxLayout(navbar)
        layout.addWidget(logo, alignment=Qt.AlignLeft)
        layout.addWidget(btn_realtime_dehazing, )
        layout.addWidget(btn_static_dehazing, )
        layout.addWidget(btn_video_dehazing, )
        layout.addWidget(btn_exit, alignment=Qt.AlignRight)

        return navbar

    def video_dehazing(self):
        # Ask the user to select a video file
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        input_video_path, _ = QFileDialog.getOpenFileName(
            self, "Select Input Video", "", "Videos (*.mp4 *.avi *.mov);;All Files (*)", options=options)

        if not input_video_path:
            print("No input video selected.")
            return

        # Ask the user to select a save location
        output_video_path, _ = QFileDialog.getSaveFileName(
            self, "Save Processed Video", "", "Videos (*.mp4 *.avi *.mov)")

        if not output_video_path:
            print("No save location selected.")
            return

        # Create a VideoProcessor object
        video_processor = VideoProcessor(input_video_path, output_video_path)

        # Start the video processing thread
        video_processor.start_processing()

    def switch_frame(self):
        frame_text = self.sender().text()

        # Define common button style
        common_style = '''
            background-color: #fff;
            border: 1px solid gray;
            border-radius: 10px;
            padding: 10px 60px;
            font-size: 13px;
        '''
        hover_style = '''
            background-color: #373030;
            color: #fff;
        '''

        if self.active_button:
            # Reset the style of the previous active button
            self.active_button.setStyleSheet(common_style)

        if frame_text == 'Realtime Dehazing':
            self.stacked_widget.setCurrentIndex(0)
            self.active_button = self.findChild(QPushButton, "realtime_button")
        elif frame_text == 'Static Dehazing':
            self.stacked_widget.setCurrentIndex(1)
            self.active_button = self.findChild(QPushButton, "static_button")

        if self.active_button:
            self.active_button.setStyleSheet(common_style + hover_style)

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

        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Enter IP address")
        layout.addWidget(self.input_field)

        # Add a button box with custom styling
        buttons = QDialogButtonBox(
            QDialogButtonBox.Save | QDialogButtonBox.Cancel,
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
                self.input_field.setText(config['DEFAULT']['input'])
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
                config['DEFAULT']['input'] = self.input_field.text()
                config['DEFAULT']['camera_name'] = camera_name.text()
                with open('settings.cfg', 'w') as configfile:
                    config.write(configfile)

                # Show a success message
                QMessageBox.information(
                    options_popup, "Success", "Settings saved successfully.")
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
            "border: 1px solid gray; border-radius: 10px; background-color: black;")

        # Add the "Select Image" button
        btn_load_image = QPushButton("Load Image")
        btn_load_image.setIcon(QIcon('gui/assets/icons/settings.svg'))
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
        btn_save_image.setIcon(QIcon('gui/assets/icons/settings.svg'))
        btn_save_image.setToolTip('Save Image')
        btn_save_image.clicked.connect(self.save_image)

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
        camera_feed = QLabel()
        camera_feed.setAlignment(Qt.AlignCenter)
        camera_feed.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding)
        camera_feed.setContentsMargins(0, 0, 0, 0)

        # Add widgets to the layout
        cctv_layout.addWidget(cctv_frame, 1, 1)
        cctv_layout.addWidget(label, 1, 1)
        cctv_layout.addWidget(camera_feed, 1, 1)

        # read the ip address from the settings.cfg file

        start_button = QPushButton("Start")
        start_button.setStyleSheet('''
            QPushButton {
                background-color: #fff;
                border: 1px solid gray;
                border-radius: 10px;
                padding: 15px;
            }
            QPushButton:hover {
                background-color: #eeeeee;
            }
        ''')

        # Create the settings button
        manage_camera_button = QPushButton()
        manage_camera_button.setIcon(QIcon('gui/assets/icons/settings.svg'))
        manage_camera_button.setToolTip('Manage Cameras')
        manage_camera_button.setStyleSheet('''
            QPushButton {
                background-color: #fff;
                border: 1px solid gray;
                border-radius: 10px;
                padding: 5px;
            }
            QPushButton:hover {
                background-color: #eeeeee;
            }
        ''')
        manage_camera_button.clicked.connect(self.show_options_popup)
        # Create a horizontal layout and add the start button and the settings button to it
        button_layout = QHBoxLayout()
        button_layout.addWidget(start_button)
        button_layout.addWidget(manage_camera_button)

        # Add the button layout to the grid layout
        cctv_layout.addLayout(button_layout, 2, 0, 1, 2, Qt.AlignCenter)
        return widget_rt

    def confirm_exit(self):
        reply = QMessageBox.question(
            self, 'Message', "Are you sure to quit?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            QApplication.quit()
