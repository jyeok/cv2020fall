import os
import numpy as np
import cv2
from scipy.interpolate import RectBivariateSpline
from skimage.filters import apply_hysteresis_threshold

def affine(x, y, p):
    # 17p: p = [dxx dxy dyx dyy dx dy]^T
    p = np.array([[1 + p[0], p[2], p[4]], [p[1], 1 + p[3], p[5]]])
    x = np.array([x, y, 1])

    return p @ x.T

def jacobian_of_warp(x, y):
    # 21p: J = [(x, 0, y, 0, 1, 0), (0, x, 0, y, 0, 1)]
    J = np.array([[x, 0, y, 0, 1, 0], [0, x, 0, y, 0, 1]], dtype=np.float64)
    
    return J

def approx_hessian(Gx, Gy):
    # 23p: H = sum_x ((grad I) * J)^T * ((grad I) * J)
    H = np.zeros((6, 6), dtype=np.float64)
    (h, w) = Gx.shape

    for y in range(h):
        for x in range(w):
            J = jacobian_of_warp(x, y)
            grad = np.array([[Gx[y, x], Gy[y, x]]])
            H += (grad @ J).T @ (grad @ J)

    return H

def downsample(img):
    # half the image, by picking maximum one because we wanted to intensify pixel difference; edge is better than uniform region.

    (orig_h, orig_w) = img.shape
    new_shape = (orig_h // 2, orig_w // 2)

    downsampled_image = np.zeros(new_shape)

    for y in range(new_shape[0]):
        for x in range(new_shape[1]):
            downsampled_image[y, x] = np.max(img[y * 2 : (y + 1) * 2, x * 2 : (x + 1) * 2])

    return downsampled_image

def img_normalize(img):
    return img.astype(np.float64) / 255.0

def lucas_kanade_affine(img1, img2, p, Gx, Gy):
    # Preprocess images
    img1_norm = downsample(img_normalize(img1))
    img2_norm = downsample(img_normalize(img2))
    (Gx, Gy) = (downsample(Gx), downsample(Gy))

    # Declaration and Constants
    (h, w) = img1_norm.shape
    summ = np.zeros((1, 6), dtype=np.float64)

    # Compute Hessian
    H_inv = np.linalg.inv(approx_hessian(Gx, Gy))

    # Compute Summation
    for y in range(h):
        for x in range(w):
            # Warping
            x_, y_ = affine(x, y, p[0])
            (x_, y_) = (int(x_), int(y_))

            # Out of Bounds
            if not (0 <= y < h and 0 <= x < w):
                continue

            # Eval Jacobian
            J = jacobian_of_warp(x, y)

            grad = np.array([[Gx[y, x], Gy[y, x]]], dtype=np.float64)
            img_diff = float(img1_norm[y, x] - img2_norm[y_, x_])

            # Accumulation
            summ += img_diff * grad @ J

    # Compute dp
    dp = summ @ H_inv
    return dp

def subtract_dominant_motion(img1, img2):
    # Sobel Operators
    Gx = cv2.Sobel(I, cv2.CV_64F, 1, 0, ksize = 5)
    Gy = cv2.Sobel(I, cv2.CV_64F, 0, 1, ksize = 5)

    # Thresholds and Parameters
    th_hi = 0.2 * 256  # you can modify this
    th_lo = 0.15 * 256  # you can modify this
    th_diff = 2e-7
    th_norm = 9e-6
    MAX_ITER = 100

    # Initialization
    moving_img = np.zeros_like(img1)
    (h, w) = img1.shape
    curr_iter = 0

    p = np.zeros((1, 6), dtype = np.float64)
    dp = np.zeros((1, 6), dtype = np.float64)
    prev_dp = np.zeros((1, 6), dtype = np.float64)

    # Update p
    while curr_iter <= 1 or (np.linalg.norm(dp) > th_norm and np.linalg.norm(dp - prev_dp) > th_diff):
        prev_dp = dp
        dp = lucas_kanade_affine(img1, img2, p, Gx, Gy)
        
        p += dp
        curr_iter += 1

        print(f'update #{curr_iter}\nupdated p: {p}\ndp: {dp}\nnorm of dp: {np.linalg.norm(dp)}\nnorm difference: {np.linalg.norm(dp - prev_dp)}')

        # prevent infinite loop
        if curr_iter > MAX_ITER:
            break

    for y in range(h):
        for x in range(w):            
            x_, y_ = affine(x, y, p[0])
            (x_, y_) = (int(x_), int(y_))

            if 0 <= y_ < h and 0 <= x_ <w:
                moving_img[y_, x_] = abs(int(img2[y_, x_]) - int(img1[y, x]))

    hyst = apply_hysteresis_threshold(moving_img, th_lo, th_hi)
    return hyst

# Parameters
NUM_PICS = 150

# Files and Paths
data_dir = 'data'
video_path = 'motion.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(video_path, fourcc, 150/20, (320, 240))
tmp_path = os.path.join(data_dir, "{}.jpg".format(0))
T = cv2.cvtColor(cv2.imread(tmp_path), cv2.COLOR_BGR2GRAY)

# Main Iteration
for i in range(NUM_PICS):
    img_path = os.path.join(data_dir, "{}.jpg".format(i))
    print(f'picture #{i+1}: {img_path}')

    I = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    clone = I.copy()

    moving_img = subtract_dominant_motion(T, I)
    
    clone = cv2.cvtColor(clone, cv2.COLOR_GRAY2BGR)
    clone[moving_img, 2] = 255
    
    out.write(clone)
    T = I

out.release()