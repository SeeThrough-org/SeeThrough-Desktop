import sys
import cv2
import threading
import time
import numpy as np
from dehazing.dehazing import dehazing
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread, QMutex, QWaitCondition, QSize
from PyQt5.QtGui import QImage, QPixmap


class CameraStream(object):
    ImagedUpdated = pyqtSignal(QImage)

    def __init__(self, src=0):
        """
        Initialize a CameraStream object with the specified video source.

        Args:
            src (int or str): The video source, typically a camera index (0 for the default camera) or a file path.
        """
        super().__init__()
        self.capture = cv2.VideoCapture(src)
        if not self.capture.isOpened():
            print('Error opening video source')

        # Start the thread to read frames from the video stream
        self.status = False  # Initialize the 'status' attribute
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        """
        Continuously capture frames from the video source in a separate thread.
        """
        # Read the next frame from the stream in a different thread
        while True:
            if self.capture.isOpened():
                (self.status, self.img) = self.capture.read()

            time.sleep(.01)

    def show_frame(self):
        """
        Display dehazed frames in real-time and calculate and print the frames per second (FPS).
        """
        if not self.status:
            print("Video source not working.")
            return
        # Initialize frame counter and FPS variables
        frame_count = 0
        start_time = time.time()
        dehazing_instance = dehazing()

        while True:
            # Call the update method to get the latest frame
            if self.status:
                dehazed_frame = dehazing_instance.image_processing(
                    self.img)  # Pass the frame as an argument

                # Calculate FPS
                frame_count += 1
                elapsed_time = time.time() - start_time
                fps = frame_count / elapsed_time
                print(fps)
                processed_frame = cv2.convertScaleAbs(
                    dehazed_frame, alpha=(255.0))
                # self.ImageUpdated.emit(processed_frame)
                # Display dehazed frame
                cv2.imshow('Frame', dehazed_frame)
                key = cv2.waitKey(1)
                if key == ord('q'):
                    self.capture.release()
                    cv2.destroyAllWindows()
                    exit(1)


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
