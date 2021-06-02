import math
import glob
import numpy as np
from PIL import Image, ImageDraw

# parameters

datadir = './data'
resultdir='./results'

sigma = 2
threshold = 1
rhoRes = 1.5
thetaRes = math.pi/180
nLines = 20

NMS_ITER = 2
NMS_BLOCK_SIZE = 8

'''
helper functions
'''

def GetImagePatch(pixel_grid, i, j, size):
    if size // 2 <= i and i < pixel_grid.shape[0] - size // 2 and size // 2 <= j and j < pixel_grid.shape[1] - size // 2:
        return pixel_grid[i - size // 2 : i + size // 2 + 1, j - size // 2 : j + size // 2 + 1]
    
    stride = size // 2

    (x_start, y_start) = (max(0, i - stride), max(0, j - stride))
    (x_end, y_end) = (min(i + stride, pixel_grid.shape[0] - 1), min(j + stride, pixel_grid.shape[1] - 1))
    (x_left, x_right) = (x_start - (i - stride), (i + stride) - x_end)
    (y_up, y_down) = (y_start - (j - stride), (j + stride) - y_end)

    before_patch = pixel_grid[x_start:x_end + 1, y_start:y_end + 1]

    return (np.pad(before_patch, ((x_left, x_right), (y_up, y_down)), 'constant', constant_values = 0))

def PadImg(Igs, G):
    return np.pad(Igs, ((G.shape[0] // 2, G.shape[0] // 2), (G.shape[1] // 2, G.shape[1] // 2)), 'edge')

def save_image(ndarray, filepath):
    image = Image.fromarray(ndarray, 'L')
    image.save(filepath)

def ToImage(Igs_float, round = False, norm = True):
    if norm:
        Igs_float = 255 * (Igs_float / np.max(Igs_float))
    if round:
        Igs_float = np.clip(Igs_float, 0, 255)
    return Igs_float.astype(np.uint8)


def GetGaussian(sigma, size):
    ranges = np.arange(-math.trunc(size / 2), math.ceil(size / 2))
    gaussians = np.exp(-(ranges ** 2) / 2 * sigma ** 2)
    gaussians = gaussians / gaussians.sum()

    gaussian_filter = gaussians.reshape(-1, 1) * gaussians.reshape(1, -1)

    return gaussian_filter

def NMS(Im, Io, y, x):
    for _ in range(NMS_ITER):
        tan_value = np.tan(Io[y][x])
        (h, w) = Im.shape

        if -1/2 <= tan_value <= 1/2: # horizontal cases
            if x > 0 and Im[y][x] < Im[y][x-1]: Im[y][x] = 0
            if x < w - 1 and Im[y][x] < Im[y][x+1]: Im[y][x] = 0
        elif 1/2 < tan_value < 2: # right-up diagnoal cases
            if x < w - 1 and y > 0 and Im[y][x] < Im[y-1][x+1]: Im[y][x] = 0
            if x > 0 and y < h - 1 and Im[y][x] < Im[y+1][x-1]: Im[y][x] = 0
        elif -2 < tan_value < -1/2: # left-up diagnoal cases
            if x > 0 and y > 0 and Im[y][x] < Im[y-1][x-1]: Im[y][x] = 0
            if x < w - 1 and y < h - 1 and Im[y][x] < Im[y+1][x+1]: Im[y][x] = 0
        else: # vertical cases
            if y > 0 and Im[y][x] < Im[y-1][x]: Im[y][x] = 0
            if y < h - 1 and Im[y][x] < Im[y+1][x]: Im[y][x] = 0

def NMS_block(H, NMS_BLOCK_SIZE, nLines):
    num_nonzeros = np.array(np.nonzero(H)).T.shape[0]
    
    while True:
        sorted_indicies = np.array(np.unravel_index(H.argsort(axis = None), H.shape)).T
        nonzero_indicies_sorted = np.array([i for i in sorted_indicies if H[i[0], i[1]] > 0])

        if num_nonzeros <= nLines or num_nonzeros == nonzero_indicies_sorted.shape[0]: break
        else: num_nonzeros = nonzero_indicies_sorted.shape[0]
        
        for indicies in nonzero_indicies_sorted:
            curr_value = H[indicies[0], indicies[1]]
            H_block = GetImagePatch(H, indicies[0], indicies[1], NMS_BLOCK_SIZE)
            
            if(H_block.max() != curr_value):
                H[indicies[0], indicies[1]] = 0

    return np.flip(nonzero_indicies_sorted, axis = 0)[:nLines, :]

def distance(p1, p2):
    return (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2

def params_to_endpoints(lrho, ltheta, shape):
    endpoints = []
    big_constant = 5000

    for rh, th in zip(lrho, ltheta):
        (a, b) = (np.cos(th), np.sin(th))
        (x_0, y_0) = (a * rh, b * rh)
        
        start = (int(x_0 + big_constant * (-b)), int(y_0 + big_constant * (a)))
        end = (int(x_0 - big_constant * (-b)), int(y_0 - big_constant * a))

        endpoints.append({'start': start, 'end': end})

    return endpoints

def draw_lines(image, points):
    drawer = ImageDraw.Draw(image)
    
    for p in points:
        drawer.line(p['start'] + p['end'], fill = 0, width = 2)

    return image

'''
functions to implement
'''
def ConvFilter(Igs, G):
    Igs_padded = PadImg(Igs, G)

    image_size = Igs.shape
    filter_size = G.shape
    iconv = np.zeros_like(Igs, dtype=np.float32)

    for y in range(image_size[0]):
        for x in range(image_size[1]):
            iconv[y][x] = (Igs_padded[y: y + filter_size[1], x : x + filter_size[0]] * G).sum()

    return iconv

def EdgeDetection(Igs, sigma):
    size = int(3 * sigma + 1)
    (h, w) = Igs.shape

    gaussian_filter = GetGaussian(sigma, size)
    smoothed_image = ConvFilter(Igs, gaussian_filter)

    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    Ix = ConvFilter(smoothed_image, sobel_x)
    Iy = ConvFilter(smoothed_image, sobel_y)
    Im = np.sqrt(Ix ** 2 + Iy ** 2)
    Io = np.arctan2(Iy, Ix)

    for y in range(h):
        for x in range(w):
            NMS(Im, Io, y, x)

    return Im, Io, Ix, Iy

def HoughTransform(Im, threshold, rhoRes, thetaRes):
    over_threshold = np.where(Im >= threshold, Im, 0)

    (y, x) = Im.shape
    (theta_len, rho_len) = (int(2 * np.pi // thetaRes), int(np.sqrt(x ** 2 + y ** 2) // rhoRes))

    H = np.zeros((rho_len, theta_len))

    nonzero_indicies = np.nonzero(over_threshold)
    points = np.array(nonzero_indicies).T

    for (y, x) in points:
        for th in range(theta_len):
            curr_theta = th * thetaRes
            curr_rho = x * np.cos(curr_theta) + y * np.sin(curr_theta)

            if 0 <= curr_rho < np.sqrt(x ** 2 + y ** 2):
                H[int(curr_rho // rhoRes)][th] += 1

    return H

def HoughLines(H, rhoRes, thetaRes, nLines):
    indicies = NMS_block(H, NMS_BLOCK_SIZE, nLines)

    lRho = [i[0] * rhoRes for i in indicies]
    lTheta = [i[1] * thetaRes for i in indicies]

    return lRho, lTheta

def HoughLineSegments(lRho, lTheta, Im, threshold):
    threshold_near = 2.25
    threshold_discrete_pixel = 144

    valid_n_lines = len(lRho)
    
    over_threshold = np.where(Im >= threshold, Im, 0)
    nonzero_indicies = np.nonzero(over_threshold)
    points = np.array(nonzero_indicies).T

    l = [{'contains': []} for _ in range(valid_n_lines)]

    for (i, rh, th) in zip(range(valid_n_lines), lRho, lTheta):
        for (y, x) in points:
            if abs(x * np.cos(th) + y * np.sin(th) - rh) <= threshold_near:
                l[i]['contains'].append((x, y))

    for line in range(len(l)):
        points_in_line = sorted(l[line]['contains'])
        max_start_point = curr_point = points_in_line[0]
        l[line]['start'] = l[line]['end'] = max_start_point
        
        max_len = -1
        curr_len = 1

        for p in range(1, len(points_in_line)):
            if distance(points_in_line[p], curr_point) <= threshold_discrete_pixel:
                curr_len += 1

                if curr_len > max_len:
                    l[line]['start'] = max_start_point
                    l[line]['end'] = points_in_line[p]
                    max_len = curr_len
            else:
                curr_len = 1
                max_start_point = points_in_line[p]

            curr_point = points_in_line[p]

    return l

def main():
    # read images
    for img_path in glob.glob(datadir+'/*.jpg'):
        # load grayscale image
        filename = img_path.split('/')[-1].split('.')[0]
        img = Image.open(img_path).convert("L")

        print(f'open {filename}...')

        Igs = np.array(img)
        Igs = Igs / 255.

        # Hough function
        print(f'Handling Edge Detection of {filename}...')
        (Im, Io, Ix, Iy) = EdgeDetection(Igs, sigma)
        save_image(ToImage(Im), f'{resultdir}/{filename}_magnitude.png')
        # save_image(ToImage(Io), f'{resultdir}/{filename}_orientation.png')

        print(f'Handling Hough Transform of {filename}...')
        H = HoughTransform(Im, threshold, rhoRes, thetaRes)
        save_image(ToImage(H), f'{resultdir}/{filename}_hough.png')

        print(f'Handling Hough Lines of {filename}...')
        (lRho, lTheta) = HoughLines(H, rhoRes, thetaRes, nLines)
        
        print(f'calculating lines of {filename}...')
        line_endpoints = params_to_endpoints(lRho, lTheta, img.size)

        print(f'Handling Hough Line Segments of {filename}...')
        segments = HoughLineSegments(lRho, lTheta, Im, threshold)

        print(f'Drawing lines of {filename}...')
        image_lines = draw_lines(img.copy(), line_endpoints)
        save_image(ToImage(image_lines), f'{resultdir}/{filename}_lines.png')
        
        print(f'Drawing line Segments of {filename}...')
        image_line_segments = draw_lines(img.copy(), segments)
        save_image(ToImage(image_line_segments), f'{resultdir}/{filename}_segments.png')

if __name__ == '__main__':
    main()
