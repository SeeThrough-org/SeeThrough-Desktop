import sys
import cv2
import threading
import time
import numpy as np
from dehazing.dehazing import dehazing
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot, Qt, QThread, QMutex, QWaitCondition, QSize
from PyQt5.QtGui import QImage, QPixmap


class CameraStream(QThread):
    # Signal emitted when a new image or a new frame is ready.
    ImageUpdated = pyqtSignal(QImage)

    def __init__(self, url) -> None:
        super(CameraStream, self).__init__()
        # Declare and initialize instance variables.
        self.url = url
        self.__thread_active = True
        self.fps = 0
        self.__thread_pause = False

    def run(self) -> None:
        # Capture video from a network stream.
        cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        # Get default video FPS.
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        print(self.fps)
        frame_count = 0
        start_time = time.time()
        # If video capturing has been initialized already.q
        if cap.isOpened():
            # While the thread is active.
            while self.__thread_active:
                #
                if not self.__thread_pause:
                    # Grabs, decodes and returns the next video frame.
                    ret, frame = cap.read()
                    dehazing_instance = dehazing()
                    # If frame is read correctly.
                    if ret:
                        dehazed_frame = dehazing_instance.image_processing(
                            frame)  # P
                        frame_count += 1
                        elapsed_time = time.time() - start_time
                        fps = frame_count / elapsed_time
                        print(f"Original FPS: {self.fps}")
                        print(f"Current FPS: {fps}")

                        # Scale the processed image values to the range [0, 255] without data loss
                        scaled_image = (dehazed_frame *
                                        255.0).clip(0, 255).astype(np.uint8)

                        # Convert the NumPy array (BGR format) to an RGB image
                        rgb_image = cv2.cvtColor(
                            scaled_image, cv2.COLOR_BGR2RGB)

                        # Create a QImage from the RGB image
                        qimage = QImage(rgb_image.data, rgb_image.shape[1],
                                        rgb_image.shape[0], rgb_image.shape[1] * 3, QImage.Format_BGR888).rgbSwapped()

                        # Emit this signal to notify that a new image or frame is available.
                        self.ImageUpdated.emit(qimage)
                    else:
                        break
        # When everything done, release the video capture object.
        cap.release()
        # Tells the thread's event loop to exit with return code 0 (success).
        self.quit()

    def stop(self) -> None:
        self.__thread_active = False

    def pause(self) -> None:
        self.__thread_pause = True

    def unpause(self) -> None:
        self.__thread_pause = False


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
