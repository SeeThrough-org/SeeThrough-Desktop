import cv2
from threading import Thread, Lock
import time
import numpy as np
from dehazing.dehazing import *
from PyQt5.QtCore import pyqtSignal, QThread, QObject, pyqtSlot
import psutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import deque
from functools import partial
from queue import Queue
class CameraStream(QThread):
    frame_processed = pyqtSignal(np.ndarray, float)
    stop_stream = pyqtSignal()

    def __init__(self, url):
        super().__init__()
        self.url = url
        self.status = True
        self.frame_count = 0
        self.start_time = time.perf_counter()
        self.stop_thread = False
        self.capture = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        if not self.capture.isOpened():
            self.status = False
            print(f"Error: Unable to open video capture from {self.url}")
        self.executor = ThreadPoolExecutor(max_workers=psutil.cpu_count(logical=False))
        self.stop_stream.connect(self.stop_update)
        self.processing_futures = set()

    def run(self):
        try:
            while not self.stop_thread:
                if self.capture.isOpened():
                    ret, frame = self.capture.read()
                    if ret:
                        if len(self.processing_futures) < psutil.cpu_count(logical=False):
                            process_frame_partial = partial(self.process_frame, frame)
                            future = self.executor.submit(process_frame_partial)
                            self.processing_futures.add(future)
                    else:
                        self.status = False
                        break

                done_futures = {f for f in self.processing_futures if f.done()}
                self.processing_futures -= done_futures

                for future in done_futures:
                    try:
                        future.result()  
                    except Exception as e:
                        print(f"Error processing frame: {e}")

                time.sleep(0.01)

        finally:
            self.stop_thread = True
            for future in self.processing_futures:
                future.cancel()
            self.executor.shutdown(wait=True)
            self.capture.release()

    def process_frame(self, frame):
        try:
            dehazing_instance = DehazingCPU()
            self.img = frame
            self.processed_frame = dehazing_instance.image_processing(frame)
            self.frame_count += 1
            elapsed_time = time.perf_counter() - self.start_time
            fps = self.frame_count / elapsed_time
            self.frame_processed.emit(self.processed_frame, fps)
        except Exception as e:
            print(f"Error processing frame: {e}")

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
        self.threads_count = min(psutil.cpu_count(logical=False), psutil.cpu_count())
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
        out = cv2.VideoWriter(self.output_file, cv2.VideoWriter_fourcc(*'H264'), original_fps, (frame_width, frame_height))
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        with ThreadPoolExecutor(max_workers=self.threads_count) as executor:
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
            progress_percentage = int((self.frames_processed / self.total_frames) * 100)
            # print(f"Outputting frame {self.frames_processed} of {self.total_frames}")
            self.update_progress_signal.emit(progress_percentage)

    def cancel_processing(self):
        self.cancel_requested = True
        self.update_progress_signal.disconnect()