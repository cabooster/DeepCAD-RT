"""
Suite of functions that help display movie

"""
import tifffile as tiff
import cv2
import numpy as np
from csbdeep.utils import normalize


def display(filename, display_length, norm_min_percent, norm_max_percent):
    """
    Display movie using opencv lib

    Args:
       filename : display image file name
       display_length : display frames number
       norm_min_percent : minimum percentile of the image you want to retain
       norm_max_percent : maximum percentile of the image you want to retain
    """
    img = tiff.imread(filename)
    img = img.astype(np.float32)
    img = img[0:display_length, :, :]
    img = normalize(img, norm_min_percent, norm_max_percent)
    cv2.namedWindow('Raw video')
    for i in range(display_length):
        tempimg = img[i, :, :]
        cv2.imshow('Raw video', tempimg)
        cv2.waitKey(33)
    cv2.destroyWindow('Raw video')


def test_img_display(img, display_length, norm_min_percent, norm_max_percent):
    """
    Display movie using opencv lib

    Args:
       img : display image file
       display_length : display frames number
       norm_min_percent : minimum percentile of the image you want to retain
       norm_max_percent : maximum percentile of the image you want to retain
    """
    img = img[50:display_length-50, :, :] # display the middle frames
    img = normalize(img, norm_min_percent, norm_max_percent)
    cv2.namedWindow('Denoised video')
    for i in range(display_length - 100):
        tempimg = img[i, :, :]
        cv2.imshow('Denoised video', tempimg)
        cv2.waitKey(33)
    cv2.destroyWindow('Denoised video')
