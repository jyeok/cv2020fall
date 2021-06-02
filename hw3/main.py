import math
import numpy as np
from PIL import Image

# IMG_SIZE = (1608, 1068)
def set_cor_mosaic():
    # selected appropriate points by matplotlib
    fig1 = np.array([(693, 466), (801, 595), (896, 829), (1069, 674), (1133, 886), (1192, 482), (1198, 608), (1289, 761), (1301, 571), (1423, 555), (1496, 981), (1505, 381), (1582, 546)])

    fig2 = np.array([(95, 437), (218, 591), (300, 847), (503, 696), (546, 882), (644, 489), (634, 609), (704, 754), (732, 578), (834, 569), (859, 950), (920, 411), (964, 563)])

    return (fig2, fig1)

def set_cor_rec():
    c_in = np.array(((158, 18), (255, 27), (159, 255), (255, 243)), dtype=np.float64)
    c_ref = np.array(((40, 54), (135, 54), (40, 223), (135, 223)), dtype=np.float64)

    return (c_ref, c_in)

def compute_h(p1, p2):
    N = p1.shape[0]
    A = np.zeros((2 * N, 9), dtype=np.float64)

    # matrix for calculating homography hard-coded
    for i in range(N):
        (x, y) = (p1[i][0], p1[i][1])
        (x_, y_) = (p2[i][0], p2[i][1])

        A[2*i, 0] = x
        A[2*i, 1] = y
        A[2*i, 2] = 1
        A[2*i, 6] = -x_ * x
        A[2*i, 7] = -x_ * y
        A[2*i, 8] = -x_

        A[2*i+1, 3] = x
        A[2*i+1, 4] = y
        A[2*i+1, 5] = 1
        A[2*i+1, 6] = -y_* x
        A[2*i+1, 7] = -y_ * y
        A[2*i+1, 8] = -y_

    # we can approximate H with svd
    (_, _, v) = np.linalg.svd(A)
    return v[-1].reshape(3, 3) # already normalized so simply reshape

def compute_h_norm(p1, p2, img_size_tuple):
    # I modified parameter so that we can normalize by image size, not by maximum point I chose.

    img_size = np.array(img_size_tuple)
    p1_norm = p1 / img_size
    p2_norm = p2 / img_size

    return compute_h(p1_norm, p2_norm)

def warp_image(igs_in, igs_ref, H, img_size_tuple):
    # hyperparameters and constants chosen by hand
    (h_warp, w_warp) = (2000, 2400) # height / width of warp image
    (h_ref, w_ref, _) = igs_ref.shape # size of ref img
    (trans_x, trans_y) = (1200, 380) # amount of transposition    
    
    img_size = np.array(img_size_tuple)
    ref_position = (trans_x, trans_y + 5)
    igs_warp = np.zeros((h_warp, w_warp, 3), dtype=np.uint8)
    merge_img_size = (2 * w_ref - 300, h_warp - 500)
    igs_merge = Image.new('RGB', merge_img_size)

    H_inv = np.linalg.inv(H)

    for y in range(h_warp):
        for x in range(w_warp):
            curr_cord_normalized = np.array([(x - trans_x) / w_ref, (y - trans_y) / h_ref, 1])

            curr_mapped_cord = H_inv @ curr_cord_normalized
            curr_mapped_cord = curr_mapped_cord / curr_mapped_cord[-1]

            (x_resized, y_resized) = (curr_mapped_cord * np.append(img_size, 1))[:2]

            # no interpolation due to overhead @ image border
            x_resized = int(x_resized)
            y_resized = int(y_resized)

            if 0 <= x_resized < w_ref and 0 <= y_resized < h_ref:
                igs_warp[y, x] = igs_in[y_resized, x_resized, :]

    # image blend with Image.paste
    igs_merge.paste(Image.fromarray(igs_warp), (0, 0))
    igs_merge.paste(Image.fromarray(igs_ref), ref_position)

    return igs_warp, np.array(igs_merge)

def rectify(igs, p1, p2):
    # hyperparameters and constants chosen by hand
    (h, w, _) = igs.shape
    (h_res, w_res) = (300, 200)

    igs_warp = np.zeros((h_res, w_res, 3), dtype=np.float64)
    
    H = compute_h_norm(p1, p2, (w, h))

    for y in range(h_res):
        for x in range(w_res):
            curr_cord_normalized = np.array([x / w, y / h, 1])

            curr_mapped_cord = H @ curr_cord_normalized
            curr_mapped_cord = curr_mapped_cord / curr_mapped_cord[-1]

            (x_resized, y_resized, _) = (curr_mapped_cord * np.array([w, h, 0]))

            # no interpolation due to overhead @ image border
            x_resized = int(x_resized)
            y_resized = int(y_resized)

            if 0 <= x_resized < w and 0 <= y_resized < h:
                igs_warp[y, x] = igs[y_resized, x_resized, :]
    return igs_warp

def main():
    ##############
    # step 1: mosaicing
    ##############

    # read images
    img_in = Image.open('data/porto1.png').convert('RGB')
    img_ref = Image.open('data/porto2.png').convert('RGB')
    img_size = (1608, 1067)

    # shape of igs_in, igs_ref: [y, x, 3]
    igs_in = np.array(img_in)
    igs_ref = np.array(img_ref)
    # lists of the corresponding points (x,y)
    # shape of p_in, p_ref: [N, 2]
    p_in, p_ref = set_cor_mosaic()

    # p_ref = H * p_in
    H = compute_h_norm(p_ref, p_in, img_size)
    igs_warp, igs_merge = warp_image(igs_in, igs_ref, H, img_size)


    # plot images
    img_warp = Image.fromarray(igs_warp.astype(np.uint8))
    img_merge = Image.fromarray(igs_merge.astype(np.uint8))

    # save images
    img_warp.save('porto1_warped.png')
    img_merge.save('porto_merged.png')

    ##############
    # step 2: rectification
    ##############

    img_rec = Image.open('data/iphone.png').convert('RGB')
    igs_rec = np.array(img_rec)

    (c_ref, c_in) = set_cor_rec()

    igs_rec = rectify(igs_rec, c_ref, c_in)

    img_rec_output = Image.fromarray(igs_rec.astype(np.uint8))
    img_rec_output.save('iphone_rectified.png')


if __name__ == '__main__':
    main()