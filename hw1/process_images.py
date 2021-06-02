import numpy as np
import math
from utils import *

def get_pixel_at(pixel_grid, i, j):
    '''
    Get pixel values at row i and column j.

    Args:
        pixel_grid (ndarray): 3D numpy array representing an RGB image.
        i (int): Row index.
        j (int): Column index.

    Returns:
        ndarray: 1Da numpy array representing RGB values.
    '''
    return (pixel_grid[i][j] if pixel_grid.shape[0] > i and pixel_grid.shape[1] > j else np.array([0, 0, 0]))

def get_patch_at(pixel_grid, i, j, size):
    '''
    Get an image patch at row i and column j.

    Args:
        pixel_grid (ndarray): 3D numpy array representing an RGB image.
        i (int): Row index.
        j (int): Column index.
        size (int): Patch size.

    Returns:
        ndarray: 3D numpy array representing an image patch.
    '''
    if size // 2 <= i and i < pixel_grid.shape[0] - size // 2 and size // 2 <= j and j < pixel_grid.shape[1] - size // 2:
        return pixel_grid[i - size // 2 : i + size // 2 + 1, j - size // 2 : j + size // 2 + 1, :]
    
    stride = size // 2

    (x_start, y_start) = (max(0, i - stride), max(0, j - stride))
    (x_end, y_end) = (min(i + stride, pixel_grid.shape[0] - 1), min(j + stride, pixel_grid.shape[1] - 1))
    (x_left, x_right) = (x_start - (i - stride), (i + stride) - x_end)
    (y_up, y_down) = (y_start - (j - stride), (j + stride) - y_end)

    before_patch = pixel_grid[x_start:x_end + 1, y_start:y_end + 1, :]

    return (np.pad(before_patch, ((x_left, x_right), (y_up, y_down), (0,0)), 'constant', constant_values = 0))

def apply_gaussian_filter(pixel_grid, size):
    '''
    Apply gaussian filter for every pixel in pixel_grid, and return the
    resulting pixel grid.

    Args:
        pixel_grid (ndarray): 3D numpy array representing an RGB image.
        size (int): Kernel size.

    Returns:
        ndarray: 3D numpy array representing an RGB image after f iltering.
    '''
    sigma = (size - 1)/3 

    ranges = np.arange(-math.trunc(size / 2), math.ceil(size / 2))
    gaussians = np.exp(-(ranges ** 2) / 2 * sigma ** 2)

    gaussian_filter = (gaussians / gaussians.sum()).reshape(-1, 1)

    new_image = np.zeros_like(pixel_grid, dtype=np.float32)

    pad = size // 2
    padded_image = np.pad(pixel_grid, ((0, 0), (pad, pad), (0, 0)))

    for x in range(pixel_grid.shape[1]):
        new_image[:,x] = np.sum(padded_image[:, x:x+size, :] * gaussian_filter, axis = 1)

    padded_image = np.pad(new_image, ((pad, pad), (0, 0), (0, 0)))

    for y in range(pixel_grid.shape[0]):
        new_image[y, :] = np.round(np.sum(padded_image[y:y+size, :, :] * gaussian_filter[:, None], axis = 0))

    return new_image.astype(np.uint8)

def apply_median_filter(pixel_grid, size):
    '''
    Apply median filter for every pixel in pixel_grid, and return the
    resulting pixel grid.

    Args:
        pixel_grid (ndarray): 3D numpy array representing an RGB image.
        size (int): Kernel size.

    Returns:
        ndarray: 3D numpy array representing an RGB image after filtering.
    '''

    new_image = np.zeros_like(pixel_grid)

    for x in range(pixel_grid.shape[0]):
        for y in range(pixel_grid.shape[1]):
            sub_block = get_patch_at(pixel_grid, x, y, size)
            reshaped_block = sub_block.reshape(-1, 3)
            new_image[x][y] = np.median(reshaped_block, axis=0)

    return new_image

def build_gaussian_pyramid(pixel_grid, size, levels=5):
    '''
    Build and return a Gaussian pyramid.

    Args:
        pixel_grid (ndarray): 3D numpy array representing an RGB image.
        size (int): Kernel size.
        levels (int): Number of levels.

    Returns:
        list of ndarray: List of 3D numpy arrays representing Gaussian
        pyramid.
    '''

    pyramids = [pixel_grid]

    for _ in range(levels):
        blurred_image = apply_gaussian_filter(pyramids[-1], size)
        downsampled_image = downsample(blurred_image)
        pyramids.append(downsampled_image)

    return pyramids

def build_laplacian_pyramid(gaussian_pyramid):
    '''
    Build and return a Laplacian pyramid.

    Args:
        gaussian_pyramid (list of ndarray): Gaussian pyramid. 

    Returns:
        list of ndarray: List of 3D numpy arrays representing Laplacian
        pyramid
    '''
    laplacians = []

    for i in range(len(gaussian_pyramid) - 1):
        l = gaussian_pyramid[i].astype(np.float32) - upsample(gaussian_pyramid[i+1].astype(np.uint8)).astype(np.float32)
        laplacians.append(l)

    laplacians.append(gaussian_pyramid[-1].astype(np.float32))
    return laplacians

def blend_images(left_image, right_image):
    (h, w, d) = left_image.shape
    size = 5
    levels = 5

    mask = concat(np.ones((h, w, d), dtype=np.uint8), np.zeros((h, w, d), dtype=np.uint8))

    mask_gaussian = build_gaussian_pyramid(mask, size, levels)
    left_gaussian = build_gaussian_pyramid(left_image, size, levels)
    right_gaussian = build_gaussian_pyramid(right_image, size, levels)

    left_laplacian = build_laplacian_pyramid(left_gaussian)
    right_laplacian = build_laplacian_pyramid(right_gaussian)

    LS = []
    for g, l, r in zip(mask_gaussian, left_laplacian, right_laplacian):
        ls = l.astype(np.float32) * g.astype(np.float32) + r.astype(np.float32) * (1.0 - g.astype(np.float32))
        LS.append(ls)
    
    result_image = LS[-1].astype(np.float32)

    for i in range(len(LS) - 2, -1, -1):
        prev_image = upsample(result_image.astype(np.uint8))
        ls_ = LS[i]
        
        result_image = np.clip(ls_.astype(np.float32) + prev_image.astype(np.float32), 0, 255)

    return result_image.astype(np.uint8)
    '''
    Smoothly blend two images by concatenation.
    
    Tip: This function should build Laplacian pyramids for both images,
    concatenate left half of left_image and right half of right_image
    on all levels, then start reconstructing from the smallest one.

    Args:
        left_image (ndarray): 3D numpy array representing an RGB image.
        right_image (ndarray): 3D numpy array representing an RGB image.

    Returns:
        ndarray: 3D numpy array representing an RGB image after blending.
    '''




if __name__ == "__main__":
    ### Test Gaussian Filter ###
    # dog_gaussian_noise = load_image('./images/dog_gaussian_noise.png')
    # after_filter = apply_gaussian_filter(dog_gaussian_noise, 5)
    # save_image(after_filter, './dog_gaussian_noise_after.png')
    
    # ### Test Median Filter ###
    # dog_salt_and_pepper = load_image('./images/dog_salt_and_pepper.png')
    # after_filter = apply_median_filter(dog_salt_and_pepper, 5)
    # save_image(after_filter, './dog_salt_and_pepper_after.png')

    # ### Test Image Blending ###
    player1 = load_image('./images/1_resized.jpg')
    player2 = load_image('./images/2_resized.jpg')
    after_blending = blend_images(player1, player2)
    save_image(after_blending, './player3.png')

    # Simple concatenation for comparison.
    save_image(concat(player1, player2), './player_simple_concat.png')
