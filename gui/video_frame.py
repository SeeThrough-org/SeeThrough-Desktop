import cv2
import numpy as np
import os
from PyQt5.QtCore import pyqtSlot, Qt, QThread, pyqtSignal, QObject
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QWidget, QGridLayout, QLabel, QPushButton, QHBoxLayout, QMessageBox, QSizePolicy,
    QFileDialog, QSlider, QVBoxLayout
)
import time
import shutil
from dehazing.utils import  VideoProcessor

class VideoThread(QThread):
    frame_available = pyqtSignal(np.ndarray)
    video_duration_available = pyqtSignal(float)
    current_time_changed = pyqtSignal(float)
    video_ended = pyqtSignal() 

    def __init__(self, video_file):
        super().__init__()
        self.video_file = video_file
        self.video_cap = None
        self.stopped = False
        self.paused = False
        self.seek_requested = False
        self.seek_position = 0

    def run(self):
        try:
            self.video_cap = cv2.VideoCapture(self.video_file)
            fps = self.video_cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.video_duration_available.emit(total_frames / fps)

            while True:
                if self.stopped:
                    break
                while self.paused:
                    time.sleep(0.1)
                    if self.stopped:
                        return

                if self.seek_requested:
                    self.video_cap.set(cv2.CAP_PROP_POS_MSEC, self.seek_position)
                    self.seek_requested = False

                ret, frame = self.video_cap.read()
                if not ret:
                    break

                current_time = self.video_cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
                self.current_time_changed.emit(current_time)
                self.frame_available.emit(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                time.sleep(1 / fps)
        except Exception as e:
            print(f"Error: {e}")
            QMessageBox.critical(None, "Error", f"An error occurred while processing the video: {str(e)}")
        finally:
            if self.video_cap:
                self.video_cap.release()
            self.video_ended.emit()

    def seek_to(self, position):
        self.seek_position = position
        self.seek_requested = True

    def stop(self):
        self.stopped = True

    def pause(self, paused):
        self.paused = paused



class VideoFrame(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        video_layout = QGridLayout(self)
        video_layout.setAlignment(Qt.AlignCenter)
        video_layout.setContentsMargins(0, 0, 0, 0)

        self.video_label = QLabel()
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setContentsMargins(0, 0, 0, 0)
        self.video_label.setStyleSheet("border: 1px solid gray; border-radius: 10px; background-color: black;")
        self.video_label.setAlignment(Qt.AlignCenter)

        video_layout.addWidget(self.video_label, 1, 1)

        self.load_video_button = QPushButton("Load Video")
        self.load_video_button.clicked.connect(self.load_video)
        self.play_pause_button = QPushButton("Play")
        self.play_pause_button.setEnabled(False)
        self.play_pause_button.clicked.connect(self.play_pause_video)
        self.save_video_button = QPushButton("Save Video")
        self.save_video_button.clicked.connect(self.save_video)
        self.dehaze_button = QPushButton("Dehaze Video")
        self.dehaze_button.clicked.connect(self.dehaze_video)
        self.dehaze_button.setEnabled(False)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.load_video_button)
        button_layout.addWidget(self.play_pause_button)
        button_layout.addWidget(self.save_video_button)
        button_layout.addWidget(self.dehaze_button)

        video_layout.addLayout(button_layout, 2, 0, 1, 2, Qt.AlignCenter)

        self.seekbar = QSlider(Qt.Horizontal)
        self.seekbar.setRange(0, 0)
        self.seekbar.sliderPressed.connect(self.pause_for_seek)
        self.seekbar.sliderReleased.connect(self.resume_after_seek)
        self.seekbar.sliderMoved.connect(self.set_position)
        self.seekbar.setEnabled(False)
        video_layout.addWidget(self.seekbar, 3, 0, 1, 2)

        self.progress_label = QLabel()
        self.progress_label.setAlignment(Qt.AlignCenter)
        video_layout.addWidget(self.progress_label, 4, 0, 1, 2)

        self.video_thread = None
        self.dehazing_thread = None
        self.video_duration = 0
        self.is_playing = False
        self.was_playing_before_seek = False
        self.video_file = None

        

    def load_video(self):
        try:
            file_path, _ = QFileDialog.getOpenFileName(self, "Open Video", "", "Video Files (*.mp4 *.avi *.mov)")
            if file_path:
                self.video_file = file_path
                self.video_thread = VideoThread(file_path)
                self.video_thread.frame_available.connect(self.update_frame)
                self.video_thread.video_duration_available.connect(self.set_video_duration)
                self.video_thread.current_time_changed.connect(self.update_seekbar_position)
                self.video_thread.video_ended.connect(self.handle_video_ended) 
                self.play_pause_button.setEnabled(True)
                self.seekbar.setEnabled(True)
                self.dehaze_button.setEnabled(True)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred while loading the video: {str(e)}")

    def update_seekbar_position(self, position):
        self.seekbar.blockSignals(True)
        self.seekbar.setValue(int(position * 1000))
        self.seekbar.blockSignals(False)

    def play_pause_video(self):
        if self.video_thread:
            if self.is_playing:
                self.video_thread.pause(True)
                self.play_pause_button.setText("Play")
            else:
                if not self.video_thread.isRunning():
                    self.video_thread.start()
                self.video_thread.pause(False)
                self.play_pause_button.setText("Pause")
            self.is_playing = not self.is_playing

    def update_frame(self, frame):
        try:
            qt_image = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            self.video_label.setPixmap(pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred while updating the frame: {str(e)}")

    def set_video_duration(self, duration):
        self.video_duration = duration
        self.seekbar.setRange(0, int(self.video_duration * 1000))

    def set_position(self, position):
        if self.video_thread:
            try:
                self.video_thread.seek_to(position)
                self.video_thread.current_time_changed.emit(position / 1000)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"An error occurred while seeking the video: {str(e)}")

    def pause_for_seek(self):
        """Pause the video thread when the seekbar is pressed."""
        if self.video_thread and self.is_playing:
            self.was_playing_before_seek = True
            self.video_thread.pause(True)
            self.play_pause_button.setText("Play")
        else:
            self.was_playing_before_seek = False

    def resume_after_seek(self):
        """Resume the video thread after the seekbar is released."""
        if self.video_thread and self.was_playing_before_seek:
            self.video_thread.pause(False)
            self.play_pause_button.setText("Pause")
            self.is_playing = True
        else:
            self.is_playing = False

    @pyqtSlot(int)
    def update_progress_label(self, progress):
        self.progress_label.setText(f"Dehazing Progress: {progress}%")

    def handle_progress_and_load_dehazed_video(self, progress):
        if progress == 100:
            temp_file = "temp_dehazed_video.mp4"
            if self.video_thread:
                self.video_thread.stop()
                self.video_thread.wait()

            self.video_thread = VideoThread(temp_file)
            self.video_thread.frame_available.connect(self.update_frame)
            self.video_thread.video_duration_available.connect(self.set_video_duration)
            self.video_thread.current_time_changed.connect(self.update_seekbar_position)
            self.video_thread.video_ended.connect(self.handle_video_ended) 
            self.video_thread.start()

            # Ensure video_cap is fully initialized before accessing it
            if os.path.exists(temp_file) and os.path.getsize(temp_file) > 0:
                video_cap = cv2.VideoCapture(temp_file)
                frame_count = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = video_cap.get(cv2.CAP_PROP_FPS)
                video_cap.release()

                if frame_count > 0 and fps > 0:
                    self.set_video_duration(frame_count / fps)
                    self.play_pause_button.setEnabled(True)
                    self.seekbar.setEnabled(True)
                    self.dehaze_button.setEnabled(False)
                    self.progress_label.setText("Dehazing completed.")
                    self.video_thread.pause(True)
                    self.play_pause_button.setText("Play")
                else:
                    QMessageBox.critical(self, "Error", "Failed to load the dehazed video.")
                    # self.play_pause_button.setEnabled(True)
                    # self.seekbar.setEnabled(True)
                    # self.dehaze_button.setEnabled(True)
            else:
                QMessageBox.critical(self, "Error", "Failed to load the dehazed video.")
                # self.play_pause_button.setEnabled(True)
                # self.seekbar.setEnabled(True)
                # self.dehaze_button.setEnabled(True)

    def dehaze_video(self):
        if self.video_file:
            try:
                self.progress_label.setText("Dehazing in progress...")
                self.play_pause_button.setEnabled(False)
                self.seekbar.setEnabled(False)
                self.dehaze_button.setEnabled(False)

                temp_file = "temp_dehazed_video.mp4"
                video_processor = VideoProcessor(self.video_file, temp_file)
                video_processor.update_progress_signal.connect(self.update_progress_label)
                video_processor.update_progress_signal.connect(self.handle_progress_and_load_dehazed_video)

                video_processor.start_processing()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"An error occurred while dehazing the video: {str(e)}")

    def save_video(self):
        if self.video_file:
            try:
                save_file, _ = QFileDialog.getSaveFileName(self, "Save Video", "", "Video Files (*.mp4)")
                if save_file:
                    if os.path.exists("temp_dehazed_video.mp4"):
                        shutil.copy("temp_dehazed_video.mp4", save_file)
                        QMessageBox.information(self, "Success", "Video saved successfully.")
                    else:
                        QMessageBox.critical(self, "Error", "No dehazed video available to save.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"An error occurred while saving the video: {str(e)}")
        else:
            QMessageBox.critical(self, "Error", "No video file loaded.")

    @pyqtSlot()
    def handle_video_ended(self):
        self.is_playing = False
        self.play_pause_button.setText("Play")

    def closeEvent(self, event):
        """Ensure threads stop properly when the main window is closed."""
        if self.video_thread:
            self.video_thread.stop()
            self.video_thread.wait()
        if self.dehazing_thread:
            self.dehazing_thread.quit()
            self.dehazing_thread.wait()