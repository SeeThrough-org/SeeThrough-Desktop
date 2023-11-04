import cv2
import threading
import time
import numpy as np
from dehazing.dehazing import dehazing


class CameraStream(object):
    def __init__(self, src=0):
        self.capture = cv2.VideoCapture(src)
        # Start the thread to read frames from the video stream
        self.status = False  # Initialize the 'status' attribute
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):

        # Read the next frame from the stream in a different thread
        while True:
            if self.capture.isOpened():
                (self.status, self.img) = self.capture.read()

            time.sleep(.01)

    def show_frame(self):
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
                print(f"Active Threads: {threading.active_count()}")

                # Display dehazed frame
                cv2.imshow('Frame', dehazed_frame)
                key = cv2.waitKey(1)
                if key == ord('q'):
                    self.capture.release()
                    cv2.destroyAllWindows()
                    exit(1)


# if __name__ == '__main__':
#     cam = CameraStream('http://192.168.1.2:4747/video')
#     while True:
#         try:
#             cam.show_frame()
#         except AttributeError:
#             pass
