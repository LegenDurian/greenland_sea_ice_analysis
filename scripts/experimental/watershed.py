"""watershed detector based on https://www.youtube.com/watch?v=3MUxPn3uKSk

does not work great. try threshold 100 w/ cv.THRESH_BINARY_INV for okay-ish results
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

def watershed():
    img_path = str(Path(__file__).parent.parent.parent / "data" / "images" / "sea_ice_thermal.jpg")
    img = cv.imread(img_path)
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_RGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    # added bilateral filter
    plt.figure()
    plt.subplot(231)
    plt.imshow(img_gray, cmap='gray')
    
    # Apply CLAHE
    clahe = cv.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    img_gray_eq = clahe.apply(img_gray)
    cv.imwrite('clahe_output.jpg', img_gray_eq)

    # img_bilateral = cv.bilateralFilter(img_gray, d=9, sigmaColor=75, sigmaSpace=75)
    # plt.imshow(img_bilateral, cmap='gray')  
    #cv.imwrite('bilateral_output.jpg', img_bilateral)

    # apply gaussian blur
    img_blur = cv.GaussianBlur(img_gray_eq, (5, 5), 0)
    plt.imshow(img_blur, cmap='gray')

    plt.subplot(232)
    _, img_threshold = cv.threshold(img_blur, 40, 255, cv.THRESH_BINARY_INV)
    plt.imshow(img_threshold, cmap='gray')

    # plt.subplot(233)
    # kernel = np.ones((3, 3), np.uint8)
    # img_dilated = cv.morphologyEx(img_threshold, cv.MORPH_DILATE, kernel)
    # plt.imshow(img_dilated)

    plt.subplot(234)
    dist_transform = cv.distanceTransform(img_threshold, cv.DIST_L2, 5)
    plt.imshow(dist_transform)

    plt.subplot(235)
    _, dist_threshold = cv.threshold(dist_transform, 5, 255, cv.THRESH_BINARY)
    plt.imshow(dist_threshold)

    plt.subplot(236)
    dist_threshold = np.uint8(dist_threshold)
    _, labels = cv.connectedComponents(dist_threshold)
    plt.imshow(labels)

    plt.figure()
    plt.subplot(121)
    labels = np.int32(labels)
    labels = cv.watershed(img_RGB, labels)

    plt.imshow(labels)

    plt.subplot(122)
    img_RGB[labels == -1] = [255, 0, 0]
    plt.imshow(img_RGB)

    plt.show()

if __name__ == "__main__":
    watershed()