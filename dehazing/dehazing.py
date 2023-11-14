import cv2
import numpy as np


class dehazing:
    def __init__(self):
        self.use_cuda = cv2.cuda.getCudaEnabledDeviceCount() > 0

    def DarkChannel(self, im, sz):
        if self.use_cuda:
            print("Using GPU")
            im_gpu = cv2.cuda_GpuMat()
            im_gpu.upload(im)

            # Cuda Split
            b, g, r = cv2.cuda.split(im)

            # Compute DCP
            dc = cv2.cuda.min(cv2.cuda.min(r, g), b)  # Use CPU min function

            # Create Kernel
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (sz, sz))

            # Erode image
            dark_gpu = cv2.cuda.createMorphologyFilter(
                cv2.MORPH_ERODE, dc.type(), kernel)
            result = dark_gpu.apply(dc)

            # Download from GPU
            dark = result.download()
        else:
            print("Using CPU")
            b, g, r = cv2.split(im)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (sz, sz))
            dc = cv2.min(cv2.min(r, g), b)
            dark = cv2.erode(dc, kernel)

        return dark

    def EstimateA(self, img, darkChannel):
        h, w, _ = img.shape
        length = h * w
        num = max(int(length * 0.0001), 1)
        darkChannVec = np.reshape(darkChannel, length)
        index = darkChannVec.argsort()[length - num:]
        rowIdx = index // w
        colIdx = index % w
        coords = np.stack((rowIdx, colIdx), axis=1)

        sumA = np.zeros((1, 3))

        if self.use_cuda:
            img_cuda = cv2.cuda_GpuMat()
            img_cuda.upload(img)
            for coord in coords:
                row, col = coord
                sumA += img_cuda.row(row).col(col).download()
        else:
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

        if self.use_cuda:
            et_cuda = cv2.cuda_GpuMat()
            et_cuda.upload(et)
            t_cuda = cv2.cuda_GpuMat()
            cv2.cuda.GaussianBlur(et_cuda, (r, r), 0, t_cuda)
            t = t_cuda.download()
        else:
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
        I_blurred = cv2.GaussianBlur(I, (5, 5), 0)
        dark = self.DarkChannel(I_blurred, 15)
        A = self.EstimateA(I, dark)
        te = self.TransmissionEstimate(I, A, 15)
        t = self.GaussianTransmissionRefine(te)
        J = self.Recover(I, t, A, 0.1)
        return J
