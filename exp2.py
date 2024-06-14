import cv2
import numpy as np
import matplotlib.pyplot as plt
def dct2(a):
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(a)))
def idct2(a):
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(a)))
def whole_img_dct(img_f32):
    img_dct = dct2(img_f32)
    img_dct_log = np.log(abs(img_dct) + 1e-10)
    img_idct = idct2(img_dct)
    return img_dct, img_dct_log, img_idct

def block_img_dct(img_f32, f_ac, f_dc):
    height, width = img_f32.shape[:2]
    block_y = height // 8
    block_x = width // 8
    height_ = block_y * 8
    width_ = block_x * 8
    img_f32_cut = img_f32[:height_, :width_]
    img_dct = np.zeros((height_, width_), dtype=np.float32)
    new_img = img_dct.copy()

    for h in range(block_y):
        for w in range(block_x):
            img_block = img_f32_cut[8 * h: 8 * (h + 1), 8 * w: 8 * (w + 1)]
            img_dct[8 * h: 8 * (h + 1), 8 * w: 8 * (w + 1)] = dct2(img_block)

            # Change the number of preserved harmonics
            img_dct[8 * h: 8 * (h + 1), 8 * w + f_ac:] = 0
            # Zero out AC
            img_dct[8 * h + f_dc:, 8 * w: 8 * (w + 1)] = 0
            # Zero out DC
            dct_block = img_dct[8 * h: 8 * (h + 1), 8 * w: 8 * (w + 1)]
            img_block = idct2(dct_block)
            new_img[8 * h: 8 * (h + 1), 8 * w: 8 * (w + 1)] = img_block

    img_dct_log = np.log(abs(img_dct) + 1e-10)
    return img_dct, img_dct_log, new_img

if __name__ == '__main__':
    img = cv2.imread(r"C:\Users\Rheina Trudy\Documents\UNI\SEMESTER 5\Multimedia\Experiments\images\1.jpg")
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_f32 = img_gray.astype(float)

    # (a) Perform DCT
    img_dct, img_dct_log, img_idct = whole_img_dct(img_f32)

    # (b) Change the number of preserved harmonics
    f_ac = 10
    f_dc = 5
    img_dct_block, img_dct_log_block, new_img = block_img_dct(img_f32.copy(), f_ac, f_dc)

    # (c) Perform Inverse DCT
    I_DCT = dct2(img_f32)

    # Visualization
    plt.figure(figsize=(12, 8))
    plt.subplot(231)
    plt.imshow(img_gray, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])

    plt.subplot(232)
    plt.imshow(img_dct_log)
    plt.title('DCT (Whole Image)'), plt.xticks([]), plt.yticks([])

    plt.subplot(233)
    plt.imshow(abs(img_idct), cmap='gray')
    plt.title('Inverse (Whole Image)'), plt.xticks([]), plt.yticks([])


    plt.subplot(234)
    plt.imshow(np.log(abs(I_DCT) + 1e-10), cmap='gray')
    plt.title('DCT Coefficients'), plt.xticks([]), plt.yticks([])

    plt.subplot(235)
    plt.imshow(img_dct_log_block)
    plt.title('DCT for 8x8'), plt.xticks([]), plt.yticks([])

    plt.subplot(236)
    plt.imshow(abs(new_img), cmap='gray')
    plt.title('Restored image'), plt.xticks([]), plt.yticks([])

    plt.show()
