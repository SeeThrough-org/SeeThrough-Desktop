from PyQt5.QtGui import QImage
from PyQt5.QtCore import pyqtSignal
import cv2
import numpy as np
import time
from threading import Thread
import cv2
import time
import math
from numba import cuda


class dehazing:
    def __init__(self):
        self.use_cuda = cv2.cuda.getCudaEnabledDeviceCount() > 0

    def DarkChannel(self, im, sz):
        b, g, r = cv2.split(im)
        dc = cv2.min(cv2.min(r, g), b)  # Use CPU min function

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (sz, sz))
        dark = cv2.erode(dc, kernel)

        return dark

    def EstimateA(self, img, darkChannel):
        h, w, _ = img.shape
        length = h * w
        num = max(int(length * 0.0001), 1)
        # convert to a row vector
        darkChannVec = np.reshape(darkChannel, length)
        index = darkChannVec.argsort()[length - num:]
        rowIdx = index // w
        colIdx = index % w
        coords = np.stack((rowIdx, colIdx), axis=1)

        sumA = np.zeros((1, 3))
        for coord in coords:
            row, col = coord
            sumA += img[row, col, :]
        A = sumA / num
        return A

    def TransmissionEstimate(self, im, A, sz):
        omega = 0.95
        im3 = np.empty(im.shape, im.dtype)

        for ind in range(0, 3):
            im3[:, :, ind] = im[:, :, ind] / A[0, ind]

        transmission = 1 - omega * self.DarkChannel(im3, sz)
        return transmission

    def GaussianTransmissionRefine(self, et):
        r = 89  # radius of the Gaussian filter

        # Apply Gaussian filtering to the transmission map
        t = cv2.GaussianBlur(et, (r, r), 0)

        return t

    def Recover(self, im, t, A, tx=0.1):
        res = np.empty(im.shape, im.dtype)
        t = cv2.max(t, tx)

        for ind in range(0, 3):
            res[:, :, ind] = (im[:, :, ind] - A[0, ind]) / t + A[0, ind]

        return res

    def image_processing(self, frame):
        I = frame.astype('float64') / 255
        # I_blurred = cv2.GaussianBlur(I, (5, 5), 0)
        dark = self.DarkChannel(I, 15)
        A = self.EstimateA(I, dark)
        te = self.TransmissionEstimate(I, A, 15)
        t = self.GaussianTransmissionRefine(te)
        J = self.Recover(I, t, A, 0.1)
        return J


class CameraStream(object):
    ImageUpdated = pyqtSignal(QImage)

    def __init__(self, src=0):
        self.capture = cv2.VideoCapture(src)
        # Start the thread to read frames from the video stream
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):

        # Read the next frame from the stream in a different thread
        while True:
            if self.capture.isOpened():
                (self.status, self.image) = self.capture.read()

            time.sleep(.01)

    @staticmethod
    @cuda.jit
    def dark_channel_cuda(image, dark_channel):
        row, col = cuda.grid(2)

        if row < image.shape[0] and col < image.shape[1]:
            min_value = image[row, col, 0]
            for channel in range(1, image.shape[2]):
                min_value = min(min_value, image[row, col, channel])

            dark_channel[row, col] = min_value

    def DarkChannel(self, image):

        d_dark_channel = cuda.to_device(
            np.zeros((self.rows, self.cols), dtype=np.float64))
        d_image = cuda.to_device(image)

        self.dark_channel_cuda[self.blockspergrid,
                               self.threadsperblock](d_image, d_dark_channel)

        h_dark_channel = d_dark_channel.copy_to_host()

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        dark_channel = cv2.erode(h_dark_channel, kernel)

        return dark_channel

    def dark_channel_cpu(self, image):
        rows, cols, _ = image.shape
        dark_channel = np.zeros((rows, cols), dtype=np.float64)

        for row in range(rows):
            for col in range(cols):
                min_value = image[row, col, 0]
                for channel in range(1, image.shape[2]):
                    min_value = min(min_value, image[row, col, channel])

                dark_channel[row, col] = min_value

        return dark_channel

    def dark_channel_filter(self, image):
        h_dark_channel = self.dark_channel_cpu(image)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        dark_channel = cv2.erode(h_dark_channel, kernel)

        return dark_channel
    # @staticmethod
    # @njit

    def EstimateA(self, img, darkChannel):
        h, w, _ = img.shape
        length = h * w
        num = max(int(length * 0.0001), 1)
        # convert to a row vector
        darkChannVec = np.reshape(darkChannel, length)
        index = darkChannVec.argsort()[length - num:]
        rowIdx = index // w
        colIdx = index % w
        coords = np.stack((rowIdx, colIdx), axis=1)

        sumA = np.zeros((1, 3))
        for coord in coords:
            row, col = coord
            sumA += img[row, col, :]
        A = sumA / num
        return A

    def TransmissionEstimate(self, im, A):
        omega = 0.95
        im3 = np.empty(im.shape, im.dtype)
        for ind in range(0, 3):
            im3[:, :, ind] = im[:, :, ind] / A[0, ind]

        transmission = 1 - omega * self.DarkChannel(im3)
        return transmission

    # def TransmissionEstimate(self, image, Alight):
    #     rows, cols, _ = image.shape
    #     radius = 7
    #     omega = 0.95
    #     tran = np.empty(image.shape, image.dtype)
    #     for i in range(self.rows):
    #         for j in range(self.cols):
    #             rmin = max(0, i - radius)
    #             rmax = min(i + radius, self.rows - 1)
    #             cmin = max(0, j - radius)
    #             cmax = min(j + radius, self.cols - 1)
    #             pixel = (image[rmin:rmax + 1, cmin:cmax + 1] / Alight).min()
    #             tran[i, j] = 1. - omega * pixel
    #     return tran

    def GaussianTransmissionRefine(self, et):
        r = 89  # radius of the Gaussian filter

        # Apply Gaussian filtering to the transmission map
        t = cv2.GaussianBlur(et, (r, r), 0)

        return t

    # @staticmethod
    # @cuda.jit
    # def gaussian_transmission_refine_cuda(et, t):
    #     row, col = cuda.grid(2)
    #
    #     if row < et.shape[0] and col < et.shape[1]:
    #         r = 4  # radius of the Gaussian filter
    #         t[row, col] = et[row, col]
    #
    #         # Apply Gaussian filter to refine transmission map
    #         for i in range(-r, r + 1):
    #             for j in range(-r, r + 1):
    #                 if row + i >= 0 and row + i < et.shape[0] and col + j >= 0 and col + j < et.shape[1]:
    #                     t[row, col] += et[row + i, col + j]
    #
    #         t[row, col] /= (2 * r + 1) ** 2
    #
    # def GaussianTransmissionRefine(self, et):
    #
    #     d_et = cuda.to_device(et)
    #     d_t = cuda.to_device(np.zeros_like(et, dtype=np.float64))
    #
    #     self.gaussian_transmission_refine_cuda[self.blockspergrid, self.threadsperblock](d_et, d_t)
    #
    #     h_t = np.empty(shape=d_t.shape, dtype=d_t.dtype)
    #     d_t.copy_to_host(h_t)
    #
    #     return h_t

    @staticmethod
    @cuda.jit
    def recover_cuda(im, t, A, res):
        row, col = cuda.grid(2)

        if row < im.shape[0] and col < im.shape[1]:
            for ind in range(0, 3):
                res[row, col, ind] = (
                    im[row, col, ind] - A[0, ind]) / t[row, col] + A[0, ind]

    def Recover(self, im, t, A, tx=0.1):
        print(f"tmap: {t.dtype}")
        t = cv2.max(t, tx)

        d_im = cuda.to_device(im)
        d_t = cuda.to_device(t)
        d_A = cuda.to_device(A)
        d_res = cuda.to_device(np.zeros(im.shape, im.dtype))

        self.recover_cuda[self.blockspergrid,
                          self.threadsperblock](d_im, d_t, d_A, d_res)

        h_res = d_res.copy_to_host()

        return h_res

    def initialize_cuda(self):
        # Initialize CUDA thread
        self.rows, self.cols, _ = self.image.shape
        self.threadsperblock = (16, 16)
        blockspergrid_x = (
            self.rows + self.threadsperblock[0] - 1) // self.threadsperblock[0]
        blockspergrid_y = (
            self.cols + self.threadsperblock[1] - 1) // self.threadsperblock[1]
        self.blockspergrid = (blockspergrid_x, blockspergrid_y)

    def process_frame(self):

        self.initialize_cuda()
        if self.use_cuda:
            # Process the frame
            image_float = self.image.astype('float64') / 255
            dark_channel = self.DarkChannel(image_float)
            A = self.EstimateA(image_float, dark_channel)
            TE = self.GaussianTransmissionRefine(
                self.TransmissionEstimate(image_float, A))
            self.frame = self.Recover(image_float, TE, A)

    def process_and_display_image(self, processed_frame):
        image = (
            processed_frame * 255.0).clip(0, 255).astype(np.uint8)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        qimage = QImage(rgb_image.data, rgb_image.shape[1], rgb_image.shape[0],
                        rgb_image.shape[1] * 3, QImage.Format_RGB888)
        return qimage

    def show_frame(self):

        # Initialize frame counter and FPS variables
        frame_count = 0
        start_time = time.time()

        dehazed_frame = self.process_frame()

        # Calculate FPS
        frame_count += 1
        elapsed_time = time.time() - start_time
        fps = math.floor(frame_count / elapsed_time)
        print(fps)
        self.ImageUpdated.emit(self.process_and_display_image(dehazed_frame))
        # # Display dehazed frame
        # cv2.imshow('Frame', self.frame)
        # key = cv2.waitKey(1)
        # if key == ord('q'):
        #     self.capture.release()
        #     cv2.destroyAllWindows()
        #     exit(1)
