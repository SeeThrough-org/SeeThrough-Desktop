import time
import cv2
import numpy as np
from PyQt5.QtCore import (Qt, QSize)
from PyQt5.QtGui import (QIcon, QPixmap, QImage)
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QPushButton, QVBoxLayout, QStackedWidget, QSizePolicy,
                             QGridLayout, QHBoxLayout, QMessageBox, QLabel, QMenu, QDialog, QFileDialog)
from dehazing.dehazing import dehazing
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

        # Create a variable to keep track of the active button
        self.active_button = None

        self.active_frame = 0

        # Create navbar and connect buttons to frame switching
        navbar = self.create_navbar()
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
        self.InputFile = None

    def input_image_clicked(self):
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
                processed_image = dehazing_instance.image_processing(
                    open_image)
                # show image  cv2 image
                cv2.imshow("image", processed_image)
                timestamp = time.strftime("%Y%m%d%H%M%S")

                # Construct the output filename with timestamp
                output_filename = f"proccesed_{timestamp}.png"

                # Save the processed image to the specified folder with the generated filename

                cv2.imwrite(output_filename,
                            (processed_image * float(255)))

                # Convert the processed image to a QImage
                # height, width, channel = processed_image.shape
                # bytes_per_line = 3 * width
                # q_image = QImage(processed_image.data, width,
                #                  height, bytes_per_line, QImage.Format_RGB888)

                # # Create a QLabel to display the image
                # image_label = QLabel()
                # pixmap = QPixmap.fromImage(q_image)
                # image_label.setPixmap(pixmap)

                # # Add the QLabel to the InputFile widget
                # self.InputFile.layout().addWidget(image_label, 0, 0)
            else:
                # Handle the case where OpenCV couldn't read the image
                print("Error: Unable to open the selected image.")
        else:
            # Handle the case where no file was selected
            print("No image selected.")

    def set_image_in_widget(self, widget, image):
        # Assuming 'image' is a numpy array representing the image
        pixmap = QPixmap.fromImage(image)
        label = widget.findChild(QLabel)
        label.setPixmap(pixmap)
        label.show()

    def save_image_clicked(self):
        # Define the action when the "Save Image" button is clicked
        # For example, save the processed image
        # Replace this with your actual save functionality
        output_image_path, _ = QFileDialog.getSaveFileName(
            self, "Save Processed Image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if output_image_path:
            # Do something with the selected output image path
            print("Output Image Path:", output_image_path)

    def create_navbar(self):
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
        btn_exit.clicked.connect(confirm_exit)

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
            self.active_frame = 0

            print(self.active_frame)
        elif frame_text == 'Static Dehazing':
            self.stacked_widget.setCurrentIndex(1)
            self.active_frame = 1

            print(self.active_frame)

    def options_frame(self):
        options_frame = QWidget()
        options_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        options_frame.setFixedHeight(50)

        layout = QHBoxLayout(options_frame)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        for widget in options_frame.findChildren(QWidget):
            widget.deleteLater()

        if self.active_frame == 0:
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

            layout.addWidget(manage_camera_button)
        elif self.active_frame == 1:
            btn_input_image = QPushButton("Select Image")
            btn_input_image.setIcon(QIcon('./images/camerasettings.svg'))
            btn_input_image.setToolTip('Select Image')
            btn_input_image.clicked.connect(self.input_image_clicked)

            btn_save_image = QPushButton("Save Image")
            btn_save_image.setIcon(QIcon('./images/camerasettings.svg'))
            btn_save_image.setToolTip('Save Image')
            btn_save_image.clicked.connect(self.save_image_clicked)

            # Apply button styling
            btn_input_image.setStyleSheet('''
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

            layout.addWidget(btn_input_image)
            layout.addWidget(btn_save_image)

        return options_frame

    def show_options_popup(self):
        options_popup = QDialog()
        options_popup.setWindowTitle("Camera Options")
        options_popup.setWindowFlags(
            options_popup.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        options_popup.exec_()

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
        self.InputFile = QWidget()  # Assign InputFile to the instance variable
        self.InputFile.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.InputFile.setContentsMargins(0, 0, 0, 0)  # Remove any margin
        self.InputFile.setStyleSheet(
            "border: 1px solid gray; border-radius: 10px; background-color: green;")

        # Create a QLabel to display the image
        image_label = QLabel()
        layout.addWidget(image_label, 0, 0)

        # Add widgets to the layout
        layout.addWidget(self.InputFile, 0, 0)

        # Input File (Only Images and Videos)
        outputFile = QWidget()
        outputFile.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        outputFile.setContentsMargins(0, 0, 0, 0)  # Remove any margin
        outputFile.setStyleSheet(
            "border: 1px solid gray; border-radius: 10px; background-color: black;")

        # Add widgets to the layout
        layout.addWidget(outputFile, 0, 1)

        # Add the "Select Image" button
        btn_input_image = QPushButton("Select Image")
        btn_input_image.setIcon(QIcon('./images/camerasettings.svg'))
        btn_input_image.setToolTip('Select Image')
        btn_input_image.clicked.connect(self.input_image_clicked)

        # Apply button styling
        btn_input_image.setStyleSheet('''
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

        # Add the "Select Image" button
        layout.addWidget(btn_input_image, 1, 0)

        # Add the "Save Image" button
        btn_save_image = QPushButton("Save Image")
        btn_save_image.setIcon(QIcon('./images/camerasettings.svg'))
        btn_save_image.setToolTip('Save Image')
        btn_save_image.clicked.connect(self.save_image_clicked)

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
        for row in range(2):
            for col in range(2):
                # Create a widget for the CCTV frame
                cctv_frame = QWidget()
                cctv_frame.setSizePolicy(
                    QSizePolicy.Expanding, QSizePolicy.Expanding)

                cctv_frame.setContentsMargins(0, 0, 0, 0)  # Remove any margin
                cctv_frame.setStyleSheet(
                    "border: 1px solid gray; border-radius: 10px; background-color: #fff;")

                # I want to add a label here that will display the camera name
                label = QLabel("Camera" + str(row*2 + col + 1))
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
                cctv_layout.addWidget(cctv_frame, row, col)
                cctv_layout.addWidget(label, row, col)
                cctv_layout.addWidget(image_label, row, col)
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


def confirm_exit():
    reply = QMessageBox.question(None, 'Message', "Are you sure to quit?", QMessageBox.Yes | QMessageBox.No,
                                 QMessageBox.No)
    if reply == QMessageBox.Yes:
        print("Exit YES")
        QApplication.quit()
    elif reply == QMessageBox.No:
        print("Exit NO")
