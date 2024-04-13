import cv2
from threading import Thread, Lock
import time
import numpy as np
from dehazing.dehazing import *
from PyQt5.QtCore import pyqtSignal, QThread, QObject, pyqtSlot
import logging
import psutil
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from collections import deque
import functools

class CameraStream(QThread):
    frame_processed = pyqtSignal(np.ndarray)
    stop_stream = pyqtSignal()

    def __init__(self, url):
        super().__init__()
        self.url = url
        self.status = True
        self.frame_count = 0
        self.start_time = time.perf_counter()
        self.logger = self.setup_logger()
        self.stop_thread = False
        self.capture = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        if not self.capture.isOpened():
            self.status = False
            self.logger.error(f"Error: Unable to open video capture from {self.url}")
        self.executor = ThreadPoolExecutor(max_workers=psutil.cpu_count(logical=False))
        self.stop_stream.connect(self.stop_update)

    def setup_logger(self):
        logger = logging.getLogger("CameraStreamLogger")
        logger.setLevel(logging.ERROR)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        ch = logging.StreamHandler()
        ch.setLevel(logging.ERROR)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        return logger

    def run(self):
        while not self.stop_thread:
            if self.capture.isOpened():
                ret, frame = self.capture.read()
                if ret:
                    process_and_emit_frame_partial = functools.partial(self.process_and_emit_frame, frame)
                    self.executor.submit(process_and_emit_frame_partial)
                else:
                    self.status = False
                    self.stop_thread = True
            else:
                self.stop_thread = True
        self.capture.release()

    def process_and_emit_frame(self, frame):
        try:
            dehazing_instance = DehazingCPU()
            frame = dehazing_instance.image_processing(frame)
            self.frame_processed.emit(frame)
            self.frame_count += 1
            elapsed_time = time.perf_counter() - self.start_time
            fps = self.frame_count / elapsed_time
            print(f"FPS: {fps:.2f}")
            self.logger.debug(f"FPS: {fps:.2f}")
        except Exception as e:
            self.logger.error(f"Error processing frame: {e}")

    @pyqtSlot()
    def stop_update(self):
        self.stop_thread = True

    def stop(self):
        self.stop_thread = True
        self.stop_stream.emit()


class VideoProcessor(QObject):
    update_progress_signal = pyqtSignal(int)

    def __init__(self, input_file, output_file):
        super(VideoProcessor, self).__init__()
        self.input_file = input_file
        self.output_file = output_file
        self.total_frames = 0
        self.frames_processed = 0
        self.status_lock = Lock()
        self.threads_count = min(psutil.cpu_count(
            logical=False), psutil.cpu_count())
        self.cancel_requested = False

    @staticmethod
    def process_frame(frame):
        dehazing_instance = DehazingCPU()
        processed_frame = dehazing_instance.image_processing(frame)
        processed_frame = cv2.convertScaleAbs(processed_frame, alpha=(255.0))
        return processed_frame

    def process_video(self):
        start_time = time.time()
        cap = cv2.VideoCapture(self.input_file)
        if cap is None or not cap.isOpened():
            print('Error opening video file')
            return

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        out = cv2.VideoWriter(self.output_file, cv2.VideoWriter_fourcc(*'H264'),
                              original_fps, (frame_width, frame_height))

        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        with ProcessPoolExecutor(max_workers=self.threads_count) as executor:
            futures = deque()

            while True:
                with self.status_lock:
                    if self.frames_processed >= self.total_frames:
                        break

                ret, frame = cap.read()
                if not ret:
                    break

                future = executor.submit(self.process_frame, frame)
                future.add_done_callback(self.update_progress)
                futures.append(future)
                if len(futures) > original_fps:
                    processed_frame = futures.popleft().result()
                    out.write(processed_frame)
                    del processed_frame  # Explicitly delete the processed frame to free up memory

                    if self.cancel_requested:
                        break

            while futures:
                processed_frame = futures.popleft().result()
                out.write(processed_frame)
                del processed_frame

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
            print(
                f"Outputting frame {self.frames_processed} of {self.total_frames}")
            self.update_progress_signal.emit(progress_percentage)

    def cancel_processing(self):
        self.cancel_requested = True
        self.update_progress_signal.disconnect()
