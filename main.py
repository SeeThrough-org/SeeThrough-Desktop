
from PyQt5.QtCore import Qt,  QSize
from PyQt5.QtGui import QIcon, QPixmap, QPainter, QColor, QMovie
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QStackedWidget,
    QMessageBox,
    QLabel,
    QProgressDialog,
    QDialog,
    QFileDialog,
    QLineEdit,
    QDialogButtonBox,
    QSplashScreen,
    QGraphicsOpacityEffect,
)
import sys
import ctypes
import os 
import time
from gui.navbar import NavBar
from gui.realtime_frame import RealtimeFrame
from gui.static_frame import StaticFrame
from dehazing.utils import VideoProcessor
from gui.config import ConfigManager
class MovieSplashScreen(QSplashScreen):

    def __init__(self, movie, parent=None):
        movie.jumpToFrame(0)
        pixmap = QPixmap(movie.frameRect().size())

        QSplashScreen.__init__(self, pixmap)
        self.movie = movie
        self.movie.frameChanged.connect(self.repaint)
    def paintEvent(self, event):
        painter = QPainter(self)
        pixmap = self.movie.currentPixmap()
        self.setMask(pixmap.mask())
        painter.drawPixmap(0, 0, pixmap)
        
    def mousePressEvent(self, event):
        pass

class GUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SeeThrough")
        self.setGeometry(100, 100, 1440, 900)
        self.setMinimumSize(1280, 720)  # Minimum width and height
        self.setWindowIcon(QIcon("assets/logo.ico")) 
        # Check setWindowIcon path 

        self.setStyleSheet("QMainWindow {background-color: #fff;}")

        # Create a stacked widget to manage frames
        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)
        self.stacked_widget.setCurrentIndex(0)

        # Create navbar and frames
        self.navbar = NavBar(self)
        self.realtime_frame = RealtimeFrame(self)
        self.static_frame = StaticFrame(self)

        # Add frames to the stacked widget
        self.stacked_widget.addWidget(self.realtime_frame)
        self.stacked_widget.addWidget(self.static_frame)

        # Connect navbar buttons to frame switching
        self.navbar.realtime_button.clicked.connect(
            lambda: self.stacked_widget.setCurrentIndex(0)
        )
        self.navbar.static_button.clicked.connect(
            lambda: self.stacked_widget.setCurrentIndex(1)
        )
        self.navbar.video_button.clicked.connect(self.video_dehazing)
        self.navbar.exit_button.clicked.connect(self.confirm_exit)

        central_layout = QVBoxLayout()
        central_layout.addWidget(self.stacked_widget)
        central_layout.addWidget(self.navbar)

        central_widget = QWidget()
        central_widget.setLayout(central_layout)
        self.setCentralWidget(central_widget)

        self.active_button = None  # Track the active button
        self.processed_image = None
        self.image_path = None
        self.camera_stream = None

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
        video_processor.update_progress_signal.connect(
            self.update_progress_dialog)
        # Create and show a progress dialog
        self.progress_dialog = QProgressDialog(
            "Processing Video...", "Cancel", 0, 100, self)
        self.progress_dialog.setWindowTitle("Video Processing")
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        self.progress_dialog.setAutoClose(True)
        self.progress_dialog.setAutoReset(False)
        self.progress_dialog.show()
        self.progress_dialog.canceled.connect(
            video_processor.cancel_processing)
        # Start the video processing thread
        video_processor.start_processing()

    def update_progress_dialog(self, progress_percentage):
        self.progress_dialog.setValue(progress_percentage)

        if progress_percentage == 100:
            # Close the progress dialog when processing is complete
            self.progress_dialog.close()
            # Show a success message
            QMessageBox.information(
                self, "Success", "Video saved successfully.")

    def confirm_exit(self):
        reply = QMessageBox.question(
            self, "Message", "Are you sure to quit?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            QApplication.quit()
    
    def show_options_popup(self):
        config_manager = ConfigManager('settings.cfg')

        options_popup = QDialog(self)
        options_popup.setWindowTitle("Camera Options")
        options_popup.setWindowFlags(options_popup.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        options_popup.setFixedWidth(320)
        options_popup.setFixedHeight(240)

        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)

        title_label = QLabel("<h2>Camera Options</h2>")
        layout.addWidget(title_label)

        input_label = QLabel("IP Address:")
        layout.addWidget(input_label)

        input_field = QLineEdit()
        input_field.setPlaceholderText("Enter IP address")
        input_field.setText(config_manager.get_value('DEFAULT', 'input', ''))
        layout.addWidget(input_field)

        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel, options_popup)
        buttons.accepted.connect(lambda: self.save_settings(config_manager, input_field, options_popup))
        buttons.rejected.connect(options_popup.reject)
        layout.addWidget(buttons)

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
        options_popup.exec_()

    def save_settings(self, config_manager, input_field, options_popup):
        config_manager.set_value('DEFAULT', 'input', input_field.text().replace('%', '%%'))
        config_manager.save_config()
        QMessageBox.information(options_popup, "Success", "Settings saved successfully.")
        options_popup.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    myappid = 'aklas.dehazing.seethrough.1' # arbitrary string
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)

    gui = GUI()
    movie = QMovie("assets/logo.gif")
    movie.setScaledSize(QSize(600, 271))
    splash = MovieSplashScreen(movie)
    splash.show()
    splash.movie.start()

    start = time.time()
    while movie.state() == QMovie.Running and time.time() < start + 5:
        app.processEvents()

    gui.show()
    splash.finish(gui)
  
    sys.exit(app.exec_())
