"""Transform Functions.

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""

from __future__ import print_function
import numpy as np
from scipy.misc import imresize
import transyou


def trans_you(ori_image, img_db, target_size=(8, 8)):
    """Transfer original image to composition of images.

    Parameters
    ----------
    ori_image : numpy.ndarray
        the original image
    img_db : h5py.File
        image datasets
    target_size : tuple

    Returns
    -------
    res_img : numpy.ndarray
        result image
    """
    tot_pixels = ori_image.shape[0]*ori_image.shape[1]
    image_idx = img_idx(tot_pixels)

    res_img = np.zeros_like(ori_image)
    res_img = imresize(res_img,
                       (res_img.shape[0]*target_size[0],
                        res_img.shape[1]*target_size[1]))
    for i in xrange(ori_image.shape[0]):
        for j in xrange(ori_image.shape[1]):
            idx = image_idx[i*ori_image.shape[1]+j]
            img = get_img(img_db, idx)
            pixel = ori_image[i, j, :]
            img = trans_img(img, pixel, target_size)
            res_img[i*target_size[0]:(i+1)*target_size[0],
                    j*target_size[1]:(j+1)*target_size[1]] = img
        print ("[MESSAGE] Row %i is processed." % (i+1))

    return res_img


def get_img(img_db, idx):
    """Get image from index.

    Parameters
    ----------
    img_db : h5py.File
        image datasets
    idx : int
        The index of the image

    Returns
    -------
    img_out : numpy.ndarray
        the output image
    """
    if (idx < 60000):
        return img_db["CIFAR10"][idx]
    elif (idx > 60000 and idx < 120000):
        return img_db["CIFAR100"][idx-60000]
    elif (idx > 120000):
        return img_db["STL10"][idx-120000]


def img_idx(num_pixels, tot_imgs=233000):
    """Select indices of images.

    Parameters
    ----------
    num_pixels : int
        number of pixels
    tot_imgs : int
        total number of images in the image datasets

    Returns
    -------
    idx : numpy.ndarray
        array of image index
    """
    num_idx = num_pixels % tot_imgs

    return np.array(np.random.choice(tot_imgs, num_idx), dtype="uint32")


def trans_img(img, pixel, target_size=(8, 8)):
    """Transform a image to a targeted pixel.

    Parameters
    ----------
    img : numpy.ndarray
        The original image
    pixel : numpy.ndarray
        the pixel, typically has 3 values

    Returns
    new_img : numpy.ndarray
        The transformed image.
    """
    img = np.array(img, dtype="int16")
    if img.ndim == 2:
        # gray image
        avg_pixel = img.mean()
        pixel_diff = np.array([pixel-avg_pixel], dtype="int16")
        img += pixel_diff[None]
        img *= (img > 0)
        img = 255*(img > 255)+img*(img <= 255)
        img = np.array(img, dtype="uint8")
    elif img.ndim == 3 and pixel.size == 3:
        # color image
        avg_pixel = img.mean(axis=(0, 1))
        pixel_diff = np.array(pixel-avg_pixel, dtype="int16")
        img += pixel_diff[None][None]
        img *= (img > 0)
        img = 255*(img > 255)+img*(img <= 255)
        img = np.array(img, dtype="uint8")
    else:
        raise ValueError("The input is not valid.")

    return imresize(img, target_size)
