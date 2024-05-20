import numpy as np
import skimage
from PIL import Image


def psnr(img1, img2):
    return skimage.metrics.peak_signal_noise_ratio(img1, img2)  # type: ignore


def ssim(img1, img2):
    assert img1.shape[0] == img2.shape[0] == 3
    return skimage.metrics.structural_similarity(  # type: ignore
        img1, img2, channel_axis=0)


def main():
    path1 = '/root/01_1_front.jpg'
    path2 = '/root/02_1_front.jpg'
    img1 = np.asarray(Image.open(path1)).transpose(2, 0, 1)
    img2 = np.asarray(Image.open(path2)).transpose(2, 0, 1)

    print(img1.shape, img2.shape)
    print(img1.dtype, img2.dtype)
    print(psnr(img1, img2))
    print(ssim(img1, img2))


if __name__ == '__main__':
    main()
