import cv2
import math
import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg
import pymatting
import threading
import tkinter as tk
from tkinter import filedialog
import threading
from concurrent.futures import ThreadPoolExecutor


class dehazing:
    def DarkChannel(self, im, sz):
        b, g, r = cv2.split(im)
        dc = cv2.min(cv2.min(r, g), b)  # Use CPU min function

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (sz, sz))
        dark = cv2.erode(dc, kernel)

        return dark

    # @jit
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
        # frame = simple_downsmpl(frame, 2)
        I = frame.astype('float64') / 255
        # Adjust kernel size and sigma as needed
        I_blurred = cv2.GaussianBlur(I, (5, 5), 0)
        dark = self.DarkChannel(I_blurred, 15)
        A = self.EstimateA(I, dark)
        te = self.TransmissionEstimate(I, A, 15)
        t = self.GaussianTransmissionRefine(te)
        J = self.Recover(I, t, A, 0.1)
        # J = lanczos_resampling_with_scale_factor(J, 2)
        return J
