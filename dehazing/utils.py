from concurrent.futures import ThreadPoolExecutor
import cv2
import threading
from threading import Thread, Lock
import time
import numpy as np
from dehazing.dehazing import *
from PyQt5.QtCore import pyqtSignal, QThread, QObject
import logging


class CameraStream(QThread):
    frame_processed = pyqtSignal(np.ndarray)

    def __init__(self, url) -> None:
        super(CameraStream, self).__init__()
        self.url = url
        self.status = None
        self.frame_count = 0
        self.start_time = time.time()
        self.logger = self.setup_logger()
        self.use_cuda = cv2.cuda.getCudaEnabledDeviceCount() > 0
        self.thread_lock = Lock()
        self.init_video_capture()
        self.width = 640
        self.height = 480
        self.inter = cv2.INTER_AREA

    def init_video_capture(self):
        try:
            self.capture = cv2.VideoCapture(self.url)
            if not self.capture.isOpened():
                raise ValueError(
                    f"Error: Unable to open video capture from {self.url}")
        except Exception as e:
            print(f"Error initializing video capture: {e}")
            self.status = False

    def setup_logger(self):
        logger = logging.getLogger("CameraStreamLogger")
        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s')

        # Log to console
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        return logger

    def update(self):
        while True:
            if self.capture.isOpened():
                self.status, frame = self.capture.read()
                self.hazy = cv2.resize(
                    frame, (self.width, self.height), self.inter)
                self.img = cv2.resize(
                    frame, (self.width, self.height), self.inter)
            else:
                self.status = False  # Ensure status is False if the capture is not opened
            if self.status:
                # Process the frame in a separate thread
                process_thread = Thread(
                    target=self.process_and_emit_frame, args=(self.img,))
                process_thread.daemon = True
                process_thread.start()
            else:
                break

    def process_and_emit_frame(self, frame):
        try:
            if not self.use_cuda:
                dehazing_instance = DehazingCPU()
                self.frame = dehazing_instance.image_processing(frame)
            else:
                dehazing_instance = DehazingCuda()
                self.frame = dehazing_instance.image_processing(frame)

            with self.thread_lock:
                # Calculate FPS
                self.frame_count += 1
                elapsed_time = time.time() - self.start_time
                fps = self.frame_count / elapsed_time
                self.logger.debug(f"FPS: {fps}")

                self.frame_processed.emit(self.frame)

        except Exception as e:
            self.logger.error(f"Error processing frame: {e}")

    def start(self) -> None:
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def stop(self) -> None:
        if self.capture is not None:
            self.capture.release()
        cv2.destroyAllWindows()
        self.terminate()


class VideoProcessor(QObject):
    """
    A class for processing videos, including dehazing the frames and saving the result.

    Attributes:
        input_file (str): The input video file path.
        output_file (str): The output video file path.
        total_frames (int): The total number of frames in the video.
        frames_processed (int): The number of frames processed.
        status_lock (threading.Lock): A lock for synchronizing status updates.
    """
    update_progress_signal = pyqtSignal(int)

    def __init__(self, input_file, output_file):
        """
        Initialize a VideoProcessor object.

        Args:
            input_file (str): The input video file path.
            output_file (str): The output video file path.
        """
        super(VideoProcessor, self).__init__()
        self.input_file = input_file
        self.output_file = output_file
        self.total_frames = 0
        self.frames_processed = 0
        self.status_lock = threading.Lock()

    def process_frame(self, frame):
        """
        Process a single frame: dehaze it and return the processed frame.

        Args:
            frame: The input frame.

        Returns:
            processed_frame: The processed frame.
        """
        dehazing_instance = DehazingCPU()
        processed_frame = dehazing_instance.image_processing(frame)
        processed_frame = cv2.convertScaleAbs(processed_frame, alpha=(255.0))
        return processed_frame

    def process_video(self):
        """
        Process the input video, dehaze each frame, and save the result to the output video file.
        """
        start_time = time.time()
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

        # Use ThreadPoolExecutor to parallelize frame processing
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []

            while True:
                with self.status_lock:
                    if self.frames_processed >= self.total_frames:
                        break  # Break the loop if all frames have been processed

                ret, frame = cap.read()
                if not ret:
                    break

                future = executor.submit(self.process_frame, frame)
                future.add_done_callback(self.update_progress)
                futures.append(future)

            for future in futures:
                processed_frame = future.result()
                out.write(processed_frame)
                print(
                    f"Processed {self.frames_processed} of {self.total_frames} frames")

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"Processing took {time.time() - start_time} seconds")

    def start_processing(self):
        """
        Start processing the video in a separate thread.
        """
        processing_thread = threading.Thread(target=self.process_video)
        processing_thread.start()

    def update_progress(self, future):
        with self.status_lock:
            self.frames_processed += 1
            progress_percentage = int(
                (self.frames_processed / self.total_frames) * 100)
            self.update_progress_signal.emit(progress_percentage)
