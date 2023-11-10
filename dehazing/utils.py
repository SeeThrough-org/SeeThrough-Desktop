import sys
import cv2
import threading
from threading import Thread
import time
import numpy as np
from dehazing.dehazing import dehazing
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot, Qt, QThread, QMutex, QWaitCondition, QSize
from PyQt5.QtGui import QImage, QPixmap


class CameraStream(QThread):
    ImageUpdated = pyqtSignal(QImage)

    def __init__(self, url) -> None:
        super(CameraStream, self).__init__()
        self.capture = cv2.VideoCapture(url)
        self.status = None
        self.frame_count = 0
        self.start_time = time.time()

    def update(self):
        while True:
            if self.capture.isOpened():
                self.status, frame = self.capture.read()
                if self.status:
                    dehazing_instance = dehazing()
                    dehazed_frame = dehazing_instance.image_processing(frame)

                    scaled_image = (
                        dehazed_frame * 255.0).clip(0, 255).astype(np.uint8)
                    rgb_image = cv2.cvtColor(scaled_image, cv2.COLOR_BGR2RGB)

                    qimage = QImage(rgb_image.data, rgb_image.shape[1], rgb_image.shape[0],
                                    rgb_image.shape[1] * 3, QImage.Format_RGB888)

                    self.ImageUpdated.emit(qimage)

                    # Calculate FPS
                    self.frame_count += 1
                    elapsed_time = time.time() - self.start_time
                    fps = self.frame_count / elapsed_time
                    print(f"Current FPS: {fps:.2f}")
                else:
                    break
            time.sleep(0.01)  # Adjust this delay as needed

    def run(self) -> None:
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def stop(self) -> None:
        self.capture.release()
        cv2.destroyAllWindows()
        self.terminate()


class VideoProcessor():
    """
    A class for processing videos, including dehazing the frames and saving the result.

    Attributes:
        input_file (str): The input video file path.
        output_file (str): The output video file path.
        total_frames (int): The total number of frames in the video.
        frames_processed (int): The number of frames processed.
        status_lock (threading.Lock): A lock for synchronizing status updates.
    """

    def __init__(self, input_file, output_file):
        """
        Initialize a VideoProcessor object.

        Args:
            input_file (str): The input video file path.
            output_file (str): The output video file path.
        """
        self.input_file = input_file
        self.output_file = output_file
        self.total_frames = 0
        self.frames_processed = 0
        self.status_lock = threading.Lock()
        self.progress_signal = None

    def set_progress_signal(self, progress_signal):
        self.progress_signal = progress_signal

    def process_video(self):
        """
        Process the input video, dehaze each frame, and save the result to the output video file.
        """
        cap = cv2.VideoCapture(self.input_file)
        if not cap.isOpened():
            print('Error opening video file')
            return

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        out = cv2.VideoWriter(self.output_file, cv2.VideoWriter_fourcc(*'mp4v'),
                              original_fps, (frame_width, frame_height))

        with self.status_lock:
            self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            dehazing_instance = dehazing()
            processed_frame = dehazing_instance.image_processing(frame)
            cv2.imshow('Processed Video', processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            processed_frame = cv2.convertScaleAbs(
                processed_frame, alpha=(255.0))

            out.write(processed_frame)
            with self.status_lock:
                self.frames_processed += 1
                print(
                    f"Processed {self.frames_processed} of {self.total_frames} frames")

        cap.release()
        out.release()
        cv2.destroyAllWindows()

    def start_processing(self):
        """
        Start processing the video in a separate thread.
        """
        processing_thread = threading.Thread(target=self.process_video)
        processing_thread.start()
