import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.io import imread
from skimage.color import rgb2gray
import time
import math


def congeal(img1, img2):
    imageA, imageB = img1, img2
    array_size = np.shape(imageA)
    # print(array_size)
    start, end = int(array_size[1]*3/7), int(array_size[1]*4/7)

    # Calculate initial error between the two images
    current_error = mse(imageA[:, start:end], imageB[:, start:end])
    imageT = imageB

    for i in range(50):
        # Transform
        imageT = A_Trans(imageT)
        new_error = mse(imageA[:, start:end], imageT[:, start:end])
        print('Current Error ' + str(i) + ': ' + str(current_error))
        print('New Error ' + str(i) + ': ' + str(new_error))
        if new_error > current_error:
            print('hello')
            cv2.imwrite('./imgT.png', imageT)
            i *= 10
            return i
        else:
            current_error = new_error

    # imageA[:, start:end] = imageA[:, start:end]-imageA[:, start:end]
    cv2.imwrite('./imgT.png', imageT)


def mse(imageA, imageB):
    err = np.sum((imageA.astype('float') - imageB.astype('float'))**2)
    err /= float(imageA.shape[0] * imageB.shape[1])

    return err


def A_Trans(image):
    M = np.float32([[1, 0, 10], [0, 1, 0]])
    imageT = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

    return imageT


def depth_estimation(pixels, displacement, focal):
    depth = focal*(displacement/pixels)
    angle = math.atan(displacement/depth)
    print(angle)

    return depth


def main():
    path1 = './img1.png'
    path2 = './img3.png'

    imageA = cv2.imread(path1)
    imageB = cv2.imread(path2)
    ouput = congeal(imageA, imageB)
    print(ouput)


if __name__ == '__main__':
    main()
