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

    def AtmLight(self, im, dark):
        [h, w] = im.shape[:2]
        imsz = h*w
        numpx = int(max(math.floor(imsz/1000), 1))
        darkvec = dark.reshape(imsz)
        imvec = im.reshape(imsz, 3)
        # cu_imvec = cp.asarray(imvec)

        indices = darkvec.argsort()
        indices = indices[imsz-numpx::]
        # cu_indices = cp.asarray(indices)

        atmsum = np.zeros([1, 3])
        for ind in range(1, numpx):
            atmsum = atmsum + imvec[indices[ind]]

        A = atmsum / numpx
        return A

    def EstimateA(self, img, DarkChann):
        h, w, _ = img.shape
        length = h*w
        num = max(int(length * 0.0001), 1)
        DarkChannVec = np.reshape(DarkChann, length)  # convert to a row vector
        index = DarkChannVec.argsort()[length-num:]
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
            im3[:, :, ind] = im[:, :, ind]/A[0, ind]

        transmission = 1 - omega*self.DarkChannel(im3, sz)
        return transmission

    def Guidedfilter_GPU(self, im, p, r, eps):
        im = im.astype(np.float32) / 255.0
        p = p.astype(np.float32) / 255.0

        # Convert images to CUDA tensors
        im_cuda = cv2.cuda_GpuMat()
        im_cuda.upload(im)
        p_cuda = cv2.cuda_GpuMat()
        p_cuda.upload(p)

        # Compute mean values
        mean_I = cv2.cuda.createContinuous(im.shape, cv2.CV_32F)
        cv2.cuda.filter2D(im_cuda, mean_I, -1,
                          cv2.cuda.createContinuous((r, r), cv2.CV_32F, 1.0 / (r * r)).mat())

        mean_p = cv2.cuda.createContinuous(p.shape, cv2.CV_32F)
        cv2.cuda.filter2D(
            p_cuda, mean_p, -1, cv2.cuda.createContinuous((r, r), cv2.CV_32F, 1.0 / (r * r)).mat())

        mean_Ip = cv2.cuda.createContinuous(im.shape, cv2.CV_32F)
        cv2.cuda.filter2D(im_cuda * p_cuda, mean_Ip, -1,
                          cv2.cuda.createContinuous((r, r), cv2.CV_32F, 1.0 / (r * r)).mat())

        # Compute covariances
        cov_Ip = mean_Ip - mean_I.mul(mean_p)
        mean_II = mean_I.mul(mean_I)
        var_I = mean_II - mean_I.mul(mean_I)

        # Compute a and b
        a = cov_Ip / (var_I + eps)
        b = mean_p - a.mul(mean_I)

        # Compute mean_a and mean_b using filtering
        mean_a = cv2.cuda.createContinuous(a.shape, cv2.CV_32F)
        cv2.cuda.filter2D(
            a, mean_a, -1, cv2.cuda.createContinuous((r, r), cv2.CV_32F, 1.0 / (r * r)).mat())

        mean_b = cv2.cuda.createContinuous(b.shape, cv2.CV_32F)
        cv2.cuda.filter2D(
            b, mean_b, -1, cv2.cuda.createContinuous((r, r), cv2.CV_32F, 1.0 / (r * r)).mat())

        # Compute the output guided filtered image
        q_cuda = mean_a.mul(im_cuda) + mean_b
        q = np.empty_like(im, dtype=np.float32)
        q_cuda.download(q)

        q = np.clip(q, 0, 1)  # Clip the values to the range [0, 1]

        return q

    def NewGuidedfilter(self, im, p, r, eps):
        te = cv2.cuda_GpuMat(p)
        gray = cv2.cuda.cvtColor(src, cv2.COLOR_BGR2GRAY)
        dst = cv2.cuda_GpuMat(src.size(), src.type())
        box_filter = cv2.cuda.createBoxFilter(
            gray.type(),
            dst.type(),
            (r, r),
            (-1, -1),
            cv2.BORDER_DEFAULT
        )

        # Define epsilon as a GPU constant
        eps_gpu = cv2.cuda.createContinuous(r, 1, cv2.CV_32F)
        eps_gpu.setTo(eps)

        # CPU
        mean_I = cv2.boxFilter(im, cv2.CV_64F, (r, r))
        mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r))
        mean_Ip = cv2.boxFilter(im*p, cv2.CV_64F, (r, r))
        cov_Ip = mean_Ip - mean_I*mean_p

        mean_II = cv2.boxFilter(im*im, cv2.CV_64F, (r, r))
        var_I = mean_II - mean_I*mean_I

        a = cov_Ip/(var_I + eps)
        b = mean_p - a*mean_I

        mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r))
        mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r))

        q = mean_a*im + mean_b

        # GPU
        # Calculate mean_I and mean_p using the box filter on the GPU
        mean_I_gpu = box_filter.apply(gray)
        mean_p_gpu = box_filter.apply(te)

        # Calculate mean_Ip using the box filter on the GPU
        mean_Ip_gpu = box_filter.apply(gray * te)

        # Calculate cov_Ip on the GPU
        cov_Ip_gpu = mean_Ip_gpu - mean_I_gpu * mean_p_gpu

        # Calculate mean_II using the box filter on the GPU
        mean_II_gpu = box_filter.apply(gray * gray)

        # Calculate var_I on the GPU
        var_I_gpu = mean_II_gpu - mean_I_gpu * mean_I_gpu

        # Calculate 'a' and 'b' on the GPU
        a_gpu = cv2.cuda.divide(cov_Ip_gpu, var_I_gpu + eps_gpu)
        b_gpu = mean_p_gpu - a_gpu * mean_I_gpu

        # Create box filters for mean_a and mean_b
        box_filter_mean_a = cv2.cuda.createBoxFilter(a_gpu.type(), -1, (r, r))
        box_filter_mean_b = cv2.cuda.createBoxFilter(b_gpu.type(), -1, (r, r))

        # Calculate mean_a and mean_b using the box filter on the GPU
        mean_a_gpu = box_filter_mean_a.apply(a_gpu)
        mean_b_gpu = box_filter_mean_b.apply(b_gpu)

        # Download the GPU results to the CPU
        cu_mean_a = mean_a_gpu.download()
        cu_mean_b = mean_b_gpu.download()

        cu_q = cu_mean_a * im + cu_mean_b

        return q

    def Guidedfilter(self, im, p, r, eps):
        mean_I = cv2.boxFilter(im, cv2.CV_64F, (r, r))
        mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r))
        mean_Ip = cv2.boxFilter(im*p, cv2.CV_64F, (r, r))
        cov_Ip = mean_Ip - mean_I*mean_p

        mean_II = cv2.boxFilter(im*im, cv2.CV_64F, (r, r))
        var_I = mean_II - mean_I*mean_I

        a = cov_Ip/(var_I + eps)
        b = mean_p - a*mean_I

        mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r))
        mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r))

        q = mean_a*im + mean_b
        return q

    def TransmissionRefine(self, im, et):
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        gray = np.float64(gray)/255
        r = 60
        eps = 0.0001
        t = self.Guidedfilter(gray, et, r, eps)

        return t

    def GaussianTransmissionRefine(self, im, et):
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        gray = np.float64(gray) / 255
        r = 59  # radius of the Gaussian filter

        # Apply Gaussian filtering to the transmission map
        t = cv2.GaussianBlur(et, (r, r), 0)

        return t

    def matting_laplacian(self, I, win_size=3, epsilon=1e-7):
        M, N, _ = I.shape
        img_size = M * N
        img_indices = np.arange(1, img_size + 1).reshape(M, N)

        row_indices = np.zeros((win_size ** 2 * img_size,))
        column_indices = np.zeros((win_size ** 2 * img_size,))
        values = np.zeros((win_size ** 2 * img_size,))
        s_len = 0

        win_radius = (win_size - 1) / 2
        U = np.eye(3)

        for rk in range(1, M + 1):
            for ck in range(1, N + 1):
                lrw_k = max(1, rk - win_radius)
                lcw_k = max(1, ck - win_radius)
                urw_k = min(M, rk + win_radius)
                ucw_k = min(N, ck + win_radius)

                w_k = I[lrw_k - 1:urw_k, lcw_k - 1:ucw_k, :]
                win_indices = img_indices[lrw_k - 1:urw_k, lcw_k - 1:ucw_k]
                Mw, Nw, _ = w_k.shape
                n_pixels = Mw * Nw
                extent = n_pixels ** 2
                w_k = w_k.reshape(n_pixels, 3)

                mu_k = np.mean(w_k, axis=0)
                Sigma_k = np.cov(w_k, rowvar=False)

                w_k = w_k - mu_k

                L_ij = (1 + np.dot(w_k, np.dot(np.linalg.inv(Sigma_k +
                        (epsilon / n_pixels) * U), w_k.T))) / n_pixels

                win_indices = np.repeat(win_indices.ravel(), n_pixels)
                row_indices[s_len:s_len + extent] = win_indices
                win_indices = win_indices.reshape((n_pixels, -1))
                column_indices[s_len:s_len + extent] = win_indices.ravel()
                values[s_len:s_len + extent] = L_ij.ravel()

                s_len += extent

        L = sp.coo_matrix((values[:s_len], (row_indices[:s_len] - 1,
                                            column_indices[:s_len] - 1)), shape=(img_size, img_size))
        L = sp.diags(L.sum(axis=1).A.ravel(), 0, (img_size, img_size)) - L

        return L

    def generate_trimap(self, t, threshold_foreground=0.1, threshold_background=0.9):
        # Create an empty trimap with the same dimensions as the transmission map
        trimap = np.zeros_like(t)

        # Set known foreground pixels
        trimap[t < threshold_foreground] = 1

        # Set known background pixels
        trimap[t > threshold_background] = 2

        # The remaining pixels are in the unknown transition region

        return trimap

    def AlphaMatTransmissionRefine(self, im, et):
        # Convert et to the range [0, 1] if it's not already
        et = np.clip(et, 0, 1)

        # Convert 'im' to float64 in the range [0, 1]
        im = im.astype(np.float64) / 255.0
        print(et)

        # Apply Closed-Form Matting
        alpha = pymatting.estimate_alpha_cf(im, et)

        # Apply the estimated alpha matte to refine the transmission map
        t = et / alpha

        return t

    def Recover(self, im, t, A, tx=0.1):
        bounded_transmission = np.zeros(im.shape)
        temp = np.copy(t)
        temp[temp < tx] = tx

        # convert to shape of the image for easier numpy computation
        for i in range(3):
            bounded_transmission[:, :, i] = temp[:, :]
        return ((im - A) / bounded_transmission) + A

    src = cv2.cuda_GpuMat()

    def process_image_part(self, image, sz, result, part_id):
        start_row = part_id * (image.shape[0] // 2)
        end_row = (part_id + 1) * (image.shape[0] // 2)

        # Get a portion of the image
        image_part = image[start_row:end_row, :]

        # Calculate the dark channel for this portion
        dark_part = self.DarkChannel(image_part, sz)

        # A = AtmLight(image_part_float, dark_part)

        # te = TransmissionEstimate(image_part_float, A, 15)

        # t = TransmissionRefine(image_part, te)

        # Store the result in the corresponding part of the result array
        result[start_row:end_row, :] = dark_part

    def single_image(self):
        import sys
        try:
            fn = sys.argv[1]
        except:
            fn = './images/1.jpg'
        start_time = time.time()
        img = cv2.imread(fn)
        # img = simple_downsmpl(img, 2)
        # Multithreading
        # start_time = time.time()
        # Number of threads (in this case, we'll use 4 threads)
        num_threads = 4

        # Initialize an array to store the results
        result = np.zeros_like(img[:, :, 0])

        # Create a list to hold thread objects
        threads = []

        # Create and start the threads
        for i in range(num_threads):
            thread = threading.Thread(
                target=self.process_image_part, args=(img, 15, result, i))
            threads.append(thread)
            thread.start()

        # Wait for all threads to finish
        for thread in threads:
            thread.join()

        # thread_time = time.time() - start_time
        # print(f"Multithreading: {thread_time}")
        # img_gray = cv2.imread(fn, cv2.IMREAD_GRAYSCALE).astype(np.float64)/255.0
        # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # GPU
        # src.upload(img)
        # empty_mat = cv2.cuda_GpuMat(src.size(), src.type())
        # start_time = time.time()
        # gaussian = cv2.cuda.createGaussianFilter(
        #     src.type(), empty_mat.type(), (5, 5), cv2.BORDER_DEFAULT)
        # gpu_time = time.time() - start_time
        # result = cu_dst.download()

        # CPU
        # start_time = time.time()
        # clahe = cv2.GaussianBlur(img, (5, 5), cv2.BORDER_DEFAULT)
        # dst = clahe.apply(img)
        # cpu_time = time.time() - start_time

        # print(f"CPU Time: {cpu_time}")
        # print(f"GPU Time: {gpu_time}")

        # cv2.imshow("GPU", result)
        # cv2.imshow("CPU", clahe)

        I = img.astype('float64')/255
        # cu_I = cp.asarray(I)
        # print("Dark Channel Stage: ")
        # start_time = time.time()
        # dark = DarkChannel(img, 15)
        # single_time = time.time() - start_time
        # print(f"Single Thread: {single_time}")
        # darkfloat = DarkChannel(I, 15)
        # start_time = time.time()
        A = self.EstimateA(I, result)
        # new = time.time() - start_time
        # start_time = time.time()
        # A2 = AtmLight(I, dark)
        # old = time.time() - start_time
        te = self.TransmissionEstimate(I, A, 15)
        # te2 = TransmissionEstimate(I, A2, 15)

        # start_time = time.time()
        # t_guided = TransmissionRefine(img, te)
        # J_guided = Recover(I, t_guided, A, 0.1)
        # guided_time = time.time() - start_time

        # start_time = time.time()
        t_gaussian = self.GaussianTransmissionRefine(img, te)
        J_gaussian = self.Recover(I, t_gaussian, A, 0.1)
        # t_gaussian2 = GaussianTransmissionRefine(img, te2)
        # J_gaussian2 = Recover(I, t_gaussian2, A2, 0.1)
        # gaussian_time = time.time() - start_time

        # start_time = time.time()
        # t_softMatt = matting_laplacian(img)
        # J_softMatt = Recover(I, t_softMatt, A, 0.1)
        # alpha_time = time.time() - start_time

        # unfiltered = Recover(I, te, A, 0.1)
        # unfiltered2 = Recover(I, te2, A2, 0.1)

        # print(f"Old AtmLight: {old}s")
        # print(f"New AtmLight: {new}s")
        # print(f"Alpha Matting Time: {alpha_time}")

        print(f"Time: {time.time() - start_time}")
        print(f"Active Threads: {threading.active_count()}")
        # cv2.imshow("Multi Threading", result)
        # cv2.imshow("Single Thread", dark)
        # cv2.imshow("t", t)
        # cv2.imshow('I', src)
        # print j_gaussian.shape
        # J_gaussian = bilinear_upsmpl(J_gaussian, 2)
        cv2.imshow('New Atmospheric Light', J_gaussian)
        # cv2.imshow('Old Atmospheric Light', J_gaussian2)
        # cv2.imshow('Guided Filter', J_guided)
        # cv2.imshow('Soft Matting', J_softMatt)
        # cv2.imshow('No Filter', unfiltered)
        # cv2.imwrite("./image/J.png", J*255)
        cv2.waitKey(0)

    def image_processing(self, frame):
        # frame = simple_downsmpl(frame, 2)
        I = frame.astype('float64') / 255
        # Adjust kernel size and sigma as needed
        I_blurred = cv2.GaussianBlur(I, (5, 5), 0)
        dark = self.DarkChannel(I_blurred, 15)
        A = self.EstimateA(I, dark)
        te = self.TransmissionEstimate(I, A, 15)
        t = self.GaussianTransmissionRefine(frame, te)
        J = self.Recover(I, t, A, 0.1)
        # J = lanczos_resampling_with_scale_factor(J, 2)
        return J 

    # Define a thread function to process a single frame

    # Function to process a single frame

    def process_frame(self, frame, processed_frames, frame_index):
        processed_frame = self.image_processing(frame)
        processed_frames[frame_index] = processed_frame

    # Function to process video frames

    def process_frames(self, video_capture, frame_count, processed_frames, num_processes):
        start_time = time.time()
        frame_index = 0
        import multiprocessing
        pool = multiprocessing.Pool(processes=num_processes)

        while True:
            ret, frame = video_capture.read()
            if not ret:
                print("Finished processing video.")
                break

            frame_count += 1
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time
            print(f"FPS: {fps}")

            # Process the frame using multiprocessing.Pool
            pool.apply_async(
                self.process_frame, (frame, processed_frames, frame_index))
            frame_index += 1
            processed_frame = pool.apply(self.image_processing, (frame,))

            cv2.imshow("Processed Video", processed_frame)
            print(f"Active Processes: {multiprocessing.active_children()}")
            print(f"Active Threads: {threading.active_count()}")
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Main thread for video display

    # Main thread for video display

    def video_input(self):
        video_capture = cv2.VideoCapture('tests.mp4')
        frame_count = 0

        if not video_capture.isOpened():
            print("Error: Could not open the video file.")
            return

        # Create a list to store processed frames
        num_processes = 16  # Adjust the number of processes as needed
        processed_frames = [None] * num_processes

        frame_count = 0

        if not video_capture.isOpened():
            print("Error: Could not open the video file.")
            exit(1)

        video_thread = threading.Thread(target=self.process_frames, args=(
            video_capture, frame_count, processed_frames, num_processes))
        video_thread.start()
        video_thread.join()
        while True:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Display the processed frames
            for i in range(num_processes):
                if processed_frames[i] is not None:
                    cv2.imshow(f"Processed Video {i}", processed_frames[i])

    def test():
        # Create sample arrays for the red, green, and blue channels
        r = np.array([[100, 50, 25],
                      [75, 30, 10]])

        g = np.array([[90, 40, 20],
                      [70, 25, 15]])

        b = np.array([[80, 35, 15],
                      [60, 20, 5]])

        # Compute the element-wise minimum operation
        dc = np.minimum(np.minimum(r, g), b)

        # Print the original arrays and the result
        print("Red Channel:")
        print(r)
        print("\nGreen Channel:")
        print(g)
        print("\nBlue Channel:")
        print(b)
        print("\nResult (Darkest Channel):")
        print(dc)
