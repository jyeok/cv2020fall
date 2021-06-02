import os
import numpy as np
import cv2
from scipy.interpolate import RectBivariateSpline
from skimage.filters import apply_hysteresis_threshold

def jacobian(x, y):
    return np.array([[x, 0, y, 0, 1, 0], [0, x, 0, y, 0, 1]], dtype=np.float64)

def hessian(Gx, Gy):
    H = np.zeros(shape=(6, 6), dtype=np.float64)
    h, w = Gx.shape
    for y in range(h):
        for x in range(w):
            J = jacobian(x, y)
            img_gradient = np.array([Gx[y, x], Gy[y, x]]).reshape(1, 2)
            H += (img_gradient @ J).T @ (img_gradient @ J)

    return H

def get_affine(x, y, p):
    return np.array([[1+p[0], p[2], p[4]], [p[1], 1+p[3], p[5]]]) @ np.array([x, y, 1]).T

def downsample(img):
    h, w = img.shape

    new_img = np.zeros(shape=(h//2, w//2))

    for y in range(h//2):
        for x in range(w//2):
            new_img[y, x] = max(img[y*2,x*2], img[y*2+1,x*2], img[y*2,x*2+1], img[y*2,x*2+1])
    return new_img

def lucas_kanade_affine(img1, img2, p, Gx, Gy):
    normalized_img1 = downsample(img1.astype(np.float64) / 255.0)
    normalized_img2 = downsample(img2.astype(np.float64) / 255.0)
    normalized_Gx = downsample(Gx)
    normalized_Gy = downsample(Gy)

    inverse_hessian = np.linalg.inv(hessian(normalized_Gx, normalized_Gy))

    h, w = normalized_img1.shape

    tmp = np.zeros((1, 6), dtype=np.float64)
    for y in range(h):
        for x in range(w):
            J = jacobian(x, y)
            x_, y_ = get_affine(x, y, p[0])
            img_gradient = np.array([normalized_Gx[y, x], normalized_Gy[y, x]], dtype=np.float64).reshape(1, 2)
            x_ = int(x_)
            y_ = int(y_)
            if 0<= x_ < w and 0 <=y_ < h:
                tmp += float(normalized_img1[y, x] - normalized_img2[y_, x_]) * img_gradient @ J

    return tmp @ inverse_hessian

def subtract_dominant_motion(img1, img2):
    # Sobel Operators
    Gx = cv2.Sobel(I, cv2.CV_64F, 1, 0, ksize = 5)
    Gy = cv2.Sobel(I, cv2.CV_64F, 0, 1, ksize = 5)

    # Thresholds and Parameters
    th_hi = 0.2 * 256  # you can modify this
    th_lo = 0.15 * 256  # you can modify this
    th_diff = 2e-5
    th_norm = 9e-6

    # Initialization
    img = np.zeros_like(img1)
    (h, w) = img1.shape
    curr_iter = 0

    p = np.zeros((1, 6))
    dp = np.zeros((1, 6))
    prev_dp = np.zeros((1, 6))

    # Update p
    while curr_iter <= 1 or (np.linalg.norm(dp) > th_norm and np.linalg.norm(dp - prev_dp) > th_diff):
        prev_dp = dp
        dp = lucas_kanade_affine(img1, img2, p, Gx, Gy)
        
        p += dp
        curr_iter += 1

        print(f'update #{curr_iter}\nupdated p: {p}\ndp: {dp}\nnorm of dp: {np.linalg.norm(dp)}\nnorm difference: {np.linalg.norm(dp - prev_dp)}')

        if np.linalg.norm(dp) < th_norm:
            break

        if curr_iter > 100:
            break

    for y in range(h):
        for x in range(w):
            x_, y_ = get_affine(x, y, p[0])
            x_ = int(x_)
            y_ = int(y_)
            if 0<=x_<w and 0<=y_<h:
                img[y_, x_] = abs(int(img2[y_, x_]) - int(img1[y, x]))

    hyst = apply_hysteresis_threshold(img, th_lo, th_hi)
    return hyst

# Parameters
NUM_PICS = 150

# Files and Paths
data_dir='data'
video_path='motion.mp4'
fourcc=cv2.VideoWriter_fourcc(*'mp4v')
out=cv2.VideoWriter(video_path, fourcc, 150/20, (320, 240))
tmp_path=os.path.join(data_dir, "{}.jpg".format(0))
T=cv2.cvtColor(cv2.imread(tmp_path), cv2.COLOR_BGR2GRAY)

# Main Iteration
for i in range(NUM_PICS):
    img_path=os.path.join(data_dir, "{}.jpg".format(i))
    print(f'picture #{i+1}: {img_path}')

    I = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    clone = I.copy()

    moving_img = subtract_dominant_motion(T, I)
    
    clone = cv2.cvtColor(clone, cv2.COLOR_GRAY2BGR)
    clone[moving_img, 2] = 255
    
    out.write(clone)
    T = I

out.release()

# import os
# import numpy as np
# import cv2
# from scipy.interpolate import RectBivariateSpline
# from skimage.filters import apply_hysteresis_threshold

# def affine(x, y, p):
#     # 17p: p = [dxx dxy dyx dyy dx dy]^T
#     p = np.array([[1 + p[0], p[2], p[4]], [p[1], 1 + p[3], p[5]]])
#     x = np.array([x, y, 1])

#     return p @ x.T

# def jacobian_of_warp(x, y):
#     # 21p: J = [(x, 0, y, 0, 1, 0), (0, x, 0, y, 0, 1)]
#     J = np.array([[x, 0, y, 0, 1, 0], [0, x, 0, y, 0, 1]], dtype=np.float64)
    
#     return J

# def approx_hessian(Gx, Gy):
#     # 23p: H = sum_x ((grad I) * J)^T * ((grad I) * J)
#     H = np.zeros((6, 6), dtype=np.float64)
#     (h, w) = Gx.shape

#     for y in range(h):
#         for x in range(w):
#             J = jacobian_of_warp(x, y)
#             grad = np.array([[Gx[y, x], Gy[y, x]]])
#             H += (grad @ J).T @ (grad @ J)

#     return H

# def downsample(img):
#     # half the image, by picking maximum one because we wanted to intensify pixel difference; edge is better than uniform region.

#     (orig_h, orig_w) = img.shape
#     new_shape = (orig_h // 2, orig_w // 2)

#     downsampled_image = np.zeros(new_shape)

#     for y in range(new_shape[0]):
#         for x in range(new_shape[1]):
#             downsampled_image[y, x] = np.max(img[y*2:(y+1)*2, x*2:(x+1)*2])

#     return downsampled_image

# def img_normalize(img):
#     return img.astype(np.float64) / 255.0


# def lucas_kanade_affine(img1, img2, p, Gx, Gy):
#     # Preprocess images
#     img1_norm = downsample(img_normalize(img1))
#     img2_norm = downsample(img_normalize(img2))
#     (Gx_norm, Gy_norm) = (Gx, Gy)
#     # Gx_norm = downsample(Gx)
#     # Gy_norm = downsample(Gy)
#     (h, w) = img1_norm.shape
#     summ = np.zeros(6, dtype=np.float64)

#     # Compute Hessian
#     H_inv = np.linalg.inv(approx_hessian(Gx_norm, Gy_norm))


#     # Compute Summation
#     for y in range(h):
#         for x in range(w):
#             # Warping
#             x_, y_ = affine(x, y, p)
#             (x_, y_) = (np.uint8(x_), np.uint8(y_))

#             # Out of Bounds
#             if not (0 <= y < h and 0 <= x < w):
#                 continue

#             # Eval Jacobian
#             J = jacobian_of_warp(x, y)

#             grad = np.array([Gx_norm[y, x], Gy_norm[y, x]], dtype=np.float64)
#             img_diff = img1_norm[y, x] - img2_norm[y_, x_]

#             summ += (grad @ J).T * img_diff
#             # summ += img_diff * grad @ J

#     # Compute dp
#     dp = H_inv @ summ
#     return dp

# def subtract_dominant_motion(img1, img2):
#     # Sobel Operators
#     Gx = cv2.Sobel(I, cv2.CV_64F, 1, 0, ksize = 5)
#     Gy = cv2.Sobel(I, cv2.CV_64F, 0, 1, ksize = 5)

#     # Thresholds and Parameters
#     th_hi = 0.2 * 256  # you can modify this
#     th_lo = 0.15 * 256  # you can modify this
#     th_diff = 2e-5
#     th_norm = 6e-5
#     MAX_ITER = 500

#     # Initialization
#     img = np.zeros_like(img1)
#     (h, w) = img1.shape
#     curr_iter = 0

#     p = np.zeros(6)
#     dp = np.zeros(6)
#     prev_dp = np.zeros(6)

#     # Update p
#     while curr_iter <= 1 or (np.linalg.norm(dp) > th_norm and np.linalg.norm(dp - prev_dp) > th_diff):
#         prev_dp = dp
#         dp = lucas_kanade_affine(img1, img2, p, Gx, Gy)
        
#         p += dp
#         curr_iter += 1

#         # print(f'update #{curr_iter}\nupdated p: {p}\ndp: {dp}\nnorm of dp: {np.linalg.norm(dp)}\nnorm difference: {np.linalg.norm(dp - prev_dp)}')

#         # prevent infinite loop
#         if curr_iter > MAX_ITER:
#             break

#     for y in range(h):
#         for x in range(w):            
#             x_, y_ = affine(x, y, p)
#             (x_, y_) = (np.uint8(x_), np.uint8(y_))

#             if 0 <= y_ < h and 0 <= x_ <w:
#                 img[y_, x_] = abs(np.float64(img2[y_, x_]) - np.float64(img1[y, x]))

#     hyst = apply_hysteresis_threshold(img, th_lo, th_hi)
#     return hyst

# # Parameters
# NUM_PICS = 150

# # Files and Paths
# data_dir='data'
# video_path='motion.mp4'
# fourcc=cv2.VideoWriter_fourcc(*'mp4v')
# out=cv2.VideoWriter(video_path, fourcc, 150/20, (320, 240))
# tmp_path=os.path.join(data_dir, "{}.jpg".format(0))
# T=cv2.cvtColor(cv2.imread(tmp_path), cv2.COLOR_BGR2GRAY)

# # Main Iteration
# for i in range(NUM_PICS):
#     img_path=os.path.join(data_dir, "{}.jpg".format(i))
#     print(f'picture #{i+1}: {img_path}')

#     I = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
#     clone = I.copy()

#     moving_img = subtract_dominant_motion(T, I)
    
#     clone = cv2.cvtColor(clone, cv2.COLOR_GRAY2BGR)
#     clone[moving_img, 2] = 255
    
#     out.write(clone)
#     T = I

# out.release()