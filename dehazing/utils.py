from concurrent.futures import ThreadPoolExecutor
import cv2
from threading import Thread, Lock
import time
import numpy as np
from dehazing.dehazing import *
from PyQt5.QtCore import pyqtSignal, QThread, QObject
import logging
import psutil
from multiprocessing import Manager


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
        self.stop_thread = False  # Flag to signal the threads to stop

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

    def grab_frames(self):
        while not self.stop_thread:
            if self.capture.isOpened():
                self.capture.grab()
            else:
                self.stop_thread = True

    def update(self):
        grab_thread = Thread(target=self.grab_frames, args=())
        grab_thread.daemon = True
        grab_thread.start()

        while not self.stop_thread:

            if self.capture.isOpened():
                self.status, frame = self.capture.retrieve()
                if self.status:

                    self.img = cv2.resize(
                        frame, (self.width, self.height), self.inter)

                    # Process the frame in a separate thread
                    process_thread = Thread(
                        target=self.process_and_emit_frame, args=(self.img,))
                    process_thread.daemon = True
                    process_thread.start()

                else:
                    self.status = False  # Ensure status is False if the capture is not opened

        grab_thread.join()

    def process_and_emit_frame(self, frame):
        try:
            if not self.use_cuda:
                dehazing_instance = DehazingCPU()
                self.frame = dehazing_instance.image_processing(frame)
            else:
                dehazing_instance = DehazingCPU()
                self.frame = dehazing_instance.image_processing(frame)

            with self.thread_lock:
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
        with self.thread_lock:
            self.stop_thread = True  # Set the flag to stop the threads
            if self.thread is not None:
                self.thread.join()
            if self.capture is not None:
                self.capture.release()
            self.terminate()


class VideoProcessor(QObject):
    update_progress_signal = pyqtSignal(int)

    def __init__(self, input_file, output_file, batch_size=30, num_threads=4):
        super(VideoProcessor, self).__init__()
        self.input_file = input_file
        self.output_file = output_file
        self.batch_size = batch_size
        self.num_threads = num_threads
        self.total_frames = 0
        self.frames_processed = 0
        self.status_lock = Lock()
        self.manager = Manager()
        self.frames_queue = self.manager.Queue()

    def process_frame(self, frame):
        dehazing_instance = DehazingCPU()
        processed_frame = dehazing_instance.image_processing(frame)
        processed_frame = cv2.convertScaleAbs(processed_frame, alpha=(255.0))
        return processed_frame

    def read_frames(self, cap, batch_size):
        frames_batch = []
        for _ in range(batch_size):
            ret, frame = cap.read()
            if not ret:
                break
            frames_batch.append(frame)
        return frames_batch

    def process_batch(self, frames_batch):
        with ThreadPoolExecutor(max_workers=self.num_threads // 2) as executor1, \
                ThreadPoolExecutor(max_workers=self.num_threads // 2) as executor2:
            futures1 = [executor1.submit(self.process_frame, frame)
                        for frame in frames_batch[:len(frames_batch)//2]]
            futures2 = [executor2.submit(self.process_frame, frame)
                        for frame in frames_batch[len(frames_batch)//2:]]

            processed_frames1 = [future.result() for future in futures1]
            processed_frames2 = [future.result() for future in futures2]

        return processed_frames1 + processed_frames2

    def write_frames(self, out, processed_frames):
        for frame in processed_frames:
            out.write(frame)

    def process_video(self):
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

        while True:
            with self.status_lock:
                if self.frames_processed >= self.total_frames:
                    break

            frames_batch = self.read_frames(cap, self.batch_size)
            if not frames_batch:
                break

            processed_frames = self.process_batch(frames_batch)
            print(
                f"Processed {self.frames_processed} of {self.total_frames} frames")

            self.write_frames(out, processed_frames)

            with self.status_lock:
                self.frames_processed += len(frames_batch)
                progress_percentage = int(
                    (self.frames_processed / self.total_frames) * 100)
                self.update_progress_signal.emit(progress_percentage)

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"Processing took {time.time() - start_time} seconds")

    def start_processing(self):
        processing_thread = Thread(target=self.process_video)
        processing_thread.start()

    def update_progress(self, future):
        with self.status_lock:
            self.frames_processed += 1
            progress_percentage = int(
                (self.frames_processed / self.total_frames) * 100)
            self.update_progress_signal.emit(progress_percentage)
