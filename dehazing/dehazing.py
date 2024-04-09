from PyQt5.QtGui import QImage
from PyQt5.QtCore import pyqtSignal
import cv2
import numpy as np
from threading import Thread
import cv2
import time
from numba import cuda
from scipy.ndimage import gaussian_filter
import numba as nb

class DehazingCPU(object):
    def DarkChannel(self, im, sz):
        b, g, r = cv2.split(im)
        dc = np.min(np.stack([r, g, b]), axis=0)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (sz, sz))
        dark = cv2.erode(dc, kernel)

        return dark

    def EstimateA(self, img, darkChannel):
        h, w, _ = img.shape
        length = h * w
        num = max(int(length * 0.0001), 1)

        darkChannVec = darkChannel.reshape(length)
        index = np.argpartition(darkChannVec, -num)[-num:]
        coords = np.column_stack(np.unravel_index(index, (h, w)))

        A = np.mean(img[coords[:, 0], coords[:, 1], :], axis=0, keepdims=True)
        return A
    
    def TransmissionEstimate(self, im, A, sz):
        omega = 0.90
        im3 = im / A[0, :]

        transmission = 1 - omega * self.DarkChannel(im3, sz)
        return transmission

    def GaussianTransmissionRefine(self, et, sigma=2):
        return gaussian_filter(et, sigma=sigma)

    def Recover(self, im, t, A, tx=0.1):
        t = np.maximum(t, tx)
        t_broadcasted = np.expand_dims(t, axis=-1)  
        res = (im - A) / t_broadcasted + A
        return res


    def image_processing(self, frame):
        I = frame.astype('float64') / 255
        dark = self.DarkChannel(I, 15)
        A = self.EstimateA(I, dark)
        te = self.TransmissionEstimate(I, A, 3)
        t = self.GaussianTransmissionRefine(te)
        J = self.Recover(I, t, A, 0.1)

        return J


class DehazingCuda(object):
    def __init__(self):
        self.use_cuda = cv2.cuda.getCudaEnabledDeviceCount() > 0

    @staticmethod
    @cuda.jit
    def dark_channel_cuda(image, dark_channel):
        row, col = cuda.grid(2)

        if row < image.shape[0] and col < image.shape[1]:
            min_value = image[row, col, 0]
            for channel in range(1, image.shape[2]):
                min_value = min(min_value, image[row, col, channel])

            dark_channel[row, col] = min_value

    def DarkChannel(self, image, patch_size):
        d_dark_channel = cuda.to_device(
            np.zeros((self.rows, self.cols), dtype=np.float64))
        d_image = cuda.to_device(image)
        self.dark_channel_cuda[self.blockspergrid,
                               self.threadsperblock](d_image, d_dark_channel)
        h_dark_channel = d_dark_channel.copy_to_host()
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (patch_size, patch_size))
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

    def TransmissionEstimate(self, im, A, patch_size):

        omega = 0.95
        im3 = np.empty(im.shape, im.dtype)
        for ind in range(0, 3):
            im3[:, :, ind] = im[:, :, ind] / A[0, ind]

        transmission = 1 - omega * self.DarkChannel(im3, patch_size)
        return transmission

    def GaussianTransmissionRefine(self, et, sigma=2):

        return gaussian_filter(et, sigma=sigma)

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

    def initialize_cuda(self, image):
        # Initialize CUDA thread
        self.rows, self.cols, _ = image.shape
        self.threadsperblock = (16, 16)
        blockspergrid_x = (
            self.rows + self.threadsperblock[0] - 1) // self.threadsperblock[0]
        blockspergrid_y = (
            self.cols + self.threadsperblock[1] - 1) // self.threadsperblock[1]
        self.blockspergrid = (blockspergrid_x, blockspergrid_y)

    def image_processing(self, frame):

        self.initialize_cuda(frame)
        # Process the frame
        image_float = frame.astype('float64') / 255
        dark_channel = self.DarkChannel(image_float, 20)
        A = self.EstimateA(image_float, dark_channel)
        TE = self.GaussianTransmissionRefine(
            self.TransmissionEstimate(image_float, A, 2))
        frame = self.Recover(image_float, TE, A)
        return frame
