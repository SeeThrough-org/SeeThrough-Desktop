from PyQt5.QtWidgets import QWidget, QLabel, QPushButton, QHBoxLayout, QVBoxLayout, QSizePolicy, QFileDialog, QMessageBox, QDialog, QDialogButtonBox, QLineEdit
from PyQt5.QtGui import QPixmap, QImage, QIcon
from PyQt5.QtCore import Qt, pyqtSlot
from dehazing.utils import CameraStream
import configparser


class Realtime_Dehazing(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        # Create widget
        widget_rt = QWidget()
        widget_rt.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Create layout
        cctv_layout = QVBoxLayout(widget_rt)
        cctv_layout.setAlignment(Qt.AlignCenter)
        cctv_layout.setContentsMargins(0, 0, 0, 0)  # Remove any margin

        # CCTV Frames

        self.cctv_frame = QLabel()
        self.cctv_frame.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.cctv_frame.setContentsMargins(0, 0, 0, 0)  # Remove any margin
        self.cctv_frame.setStyleSheet(
            "border: 1px solid gray; border-radius: 10px; background-color: black;")

        # I want to add a label here that will display the camera name
        self.label = QLabel("")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("font-size: 20px; font-weight: bold;")
        self.label.setContentsMargins(0, 0, 0, 0)  # Remove any margin

        # You can create an image label to display the camera feed
        camera_feed = QLabel()
        camera_feed.setAlignment(Qt.AlignCenter)
        camera_feed.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        camera_feed.setContentsMargins(0, 0, 0, 0)

        # Add widgets to the layout
        cctv_layout.addWidget(self.cctv_frame)
        cctv_layout.addWidget(self.label)
        cctv_layout.addWidget(camera_feed)

        # read the ip address from the settings.cfg file
        self.start_button = QPushButton("Start")
        self.start_button.setCheckable(True)  # Make it a toggle button

        # Connect the button's toggled signal to the start_camera_stream method
        self.start_button.toggled.connect(self.start_camera_stream)

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
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(manage_camera_button)

        # Add the button layout to the vertical layout
        cctv_layout.addLayout(button_layout)

        # Set the layout for the widget
        self.setLayout(cctv_layout)

    @pyqtSlot()
    def start_camera_stream(self):      
        config = configparser.ConfigParser()
        config.read('settings.cfg')
        if 'DEFAULT' in config and 'input' in config['DEFAULT']:
            ip_address = config['DEFAULT']['input']
        else:
            ip_address = '0'

        if self.start_button.isChecked():
            # Create an instance of the CameraStream class (assuming it's properly initialized)
            self.camera_stream = CameraStream(ip_address)

            # Connect the CameraStream's signal to update the cctv_frame
            self.camera_stream.run()
            self.camera_stream.ImageUpdated.connect(self.update_cctv_frame)

            # Start the camera stream
            # self.camera_stream.status = True
            self.camera_stream.start()
            self.start_button.setText("Stop")
        else:
            # Stop the camera stream if the button is unchecked
            self.start_button.setText("Start")
            if hasattr(self, 'camera_stream'):
                # self.camera_stream.status = False
                self.camera_stream.stop()

    @pyqtSlot(QImage)
    def update_cctv_frame(self, image):
        # Convert the image to QPixmap
        pixmap = QPixmap.fromImage(image)

        # Scale the pixmap while keeping the aspect ratio
        pixmap = pixmap.scaled(self.cctv_frame.width(),
                               self.cctv_frame.height(), Qt.KeepAspectRatio)

        # Update the camera feed label with the scaled pixmap
        self.cctv_frame.setPixmap(pixmap)
        self.cctv_frame.setAlignment(Qt.AlignCenter)

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
                with open('settings.cfg', 'w') as configfile:
                    config.write(configfile)

                # Show a success message
                QMessageBox.information(
                    options_popup, "Success", "Settings saved successfully.")
            except (FileNotFoundError, configparser.Error) as e:
                print(f"Error saving settings: {e}")
