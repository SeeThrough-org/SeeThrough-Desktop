import cv2
from threading import Thread, Lock
import time
import numpy as np
from dehazing.dehazing import *
from PyQt5.QtCore import pyqtSignal, QThread, QObject, QMetaObject, Qt, QTimer, pyqtSlot
import logging
import psutil
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from collections import deque
import threading

class CameraStreamThread(QThread):
    def __init__(self, camera_stream):
        super().__init__()
        self.camera_stream = camera_stream

    def run(self):
        self.camera_stream.update()
    def stop(self):
        # self.camera_stream.stop()
        self.camera_stream.stop()
        self.quit()
        self.wait()


class CameraStream(QObject):
    frame_processed = pyqtSignal(np.ndarray)
    stop_stream = pyqtSignal()

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
        self.stop_thread = False  
        self.frame = None
        self.executor = ThreadPoolExecutor(max_workers=min(psutil.cpu_count(
            logical=False), psutil.cpu_count()))
        self.stop_stream.connect(self.stop_update)

    def init_video_capture(self):
        try:
            self.capture = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
            if not self.capture.isOpened():
                raise ValueError(
                    f"Error: Unable to open video capture from {self.url}")
        except cv2.error as e:
            # Check the specific error code
            if e.err == cv2.VIDEOINPUT_ERR_INVALID_ARGUMENT:
                print(f"Invalid URL: {self.url}")
            else:
                print(f"OpenCV error: {e.err}")
            self.status = False
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
                try:
                    ret = self.capture.grab()
                    if not ret:
                        print("Failed to grab frame")
                        break
                except cv2.error as e:
                    print(f"OpenCV error: {e}")
                    break
                except Exception as e:
                    print(f"Unknown error: {e}")
                    break
            else:
                print("Video capture is not opened")
                self.stop_thread = True

    def update(self):
        try:
            future = self.executor.submit(self.grab_frames)
            while not self.stop_thread:
                try:
                    if self.capture.isOpened():
                        self.status, self.frame = self.capture.retrieve()
                        if self.status:
                            future_process = self.executor.submit(self.process_and_emit_frame, self.frame)
                            future_process.result()
                        else:
                            self.status = False
                    else:
                        self.stop_thread = True
                except Exception as e:
                    print(f"Error processing frame: {e}")
                    self.status = False

            future.result()
        except Exception as e:
            print(f"Error updating camera stream: {e}")

    def process_and_emit_frame(self, frame):
        try:
            dehazing_instance = DehazingCPU()
            future_dehazing = self.executor.submit(lambda: dehazing_instance.image_processing(frame))
            self.frame = future_dehazing.result()

            if self.frame is not None:
                with self.thread_lock:
                    self.frame_count += 1
                    elapsed_time = time.time() - self.start_time
                    fps = self.frame_count / elapsed_time
                    self.logger.debug(f"FPS: {fps}")
                    self.frame_processed.emit(self.frame)
        except Exception as e:
            self.logger.error(f"Error processing frame: {e}")

    def start(self) -> None:
        try:
            self.thread = Thread(target=self.update, args=())
            self.thread.daemon = True
            self.thread.start()
        except Exception as e:
            print(f"Error starting camera stream: {e}")

    @pyqtSlot()
    def stop_update(self):
        self.stop_thread = True

    @pyqtSlot()
    def stop_timer(self):
        if self.thread is not None and isinstance(self.thread, threading.Thread):
            self.thread.join()
        if self.capture is not None:
            self.capture.release()

    def stop(self) -> None:
        try:
            with self.thread_lock:
                self.stop_thread = True  # Set the flag to stop the threads
                self.stop_stream.emit()  # Emit the signal to stop the stream
                QMetaObject.invokeMethod(self, "stop_timer", Qt.QueuedConnection)
        except Exception as e:
            print(f"Error stopping camera stream: {e}")
    
    

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
