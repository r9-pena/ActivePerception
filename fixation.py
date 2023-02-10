import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.io import imread
from skimage.color import rgb2gray
import time
import math
from scipy.stats import _entropy


def congeal(img1, img2):
    imageA, imageB = img1, img2
    array_size = np.shape(imageA)
    # print(array_size)
    w_start, w_end = int(array_size[1]*3/7), int(array_size[1]*5/7)
    h_start, h_end = int(array_size[0]*2/5), int(array_size[0]*3/5)

    # Calculate initial error between the two images
    current_error = mse(imageA[h_start:h_end, w_start:w_end],
                        imageB[h_start:h_end, w_start:w_end])
    imageT = imageB

    for i in range(20):
        # Transform
        imageT = A_Trans(imageT, i)
        new_error = mse(imageA[h_start:h_end, w_start:w_end],
                        imageT[h_start:h_end, w_start:w_end])
        print('Current Error ' + str(i) + ': ' + str(current_error))
        print('New Error ' + str(i) + ': ' + str(new_error))
        if new_error > 1.05*current_error:
            print('hello')
            cv2.imwrite('./imgT.png', imageT)
            angle = i-1
            return angle
        else:
            current_error = new_error

    # imageA[:, start:end] = imageA[:, start:end]-imageA[:, start:end]
    cv2.imwrite('./imgT.png', imageT)


def mse(imageA, imageB):
    err = np.sum((imageA.astype('float') - imageB.astype('float'))**2)
    err /= float(imageA.shape[0] * imageB.shape[1])

    return err


def translate(image, offset):
    M = np.float32([[1, 0, offset], [0, 1, 0]])
    imageT = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

    return imageT


def A_Trans(image, theta):
    print('-------------------------------------')
    print('Angle (degrees): ' + str(theta))
    theta = theta * math.pi/180
    h, w = image.shape[:2]
    # print('h=' + str(h))
    # print('w=' + str(w))
    diag = (h ** 2 + w ** 2) ** 0.5
    f = diag
    # print('f=' + str(f))
    sin_r, cos_r = np.sin(theta), np.cos(theta)
    Identity = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    H_M = np.array([
        [1, 0, -w / 2],
        [0, 1, -h / 2],
        [0, 0,      1],
        [0, 0,      1]
    ])
    Hp_M = np.array([
        [f, 0, w / 2, 0],
        [0, f, h / 2, 0],
        [0, 0,     1, 0]
    ])
    R_M = np.array([
        [cos_r, 0, -sin_r, 0],
        [0, 1,       0, 0],
        [sin_r, 0,  cos_r, 0],
        [0, 0,       0, 1]
    ])
    inv_M = np.array([
        [1/f, 0, (f-1)*w/(2*f)],
        [0, 1/f, (f-1)*h/(2*f)],
        [0, 0, 1]
    ])
    M = Identity
    # M = np.dot(R_M,  M)
    M = np.dot(Hp_M, np.dot(M, H_M))
    # M = np.dot(M, inv_M)
    dsize = (w, h)
    imageT = cv2.warpPerspective(
        image, M, dsize)
    mid = imageT[200, :]
    zrs = w*3-np.count_nonzero(mid)
    offset = int(zrs/6)
    print(zrs)
    imageT = translate(imageT, offset)

    return imageT


def depth_estimation(theta, displacement):
    depth = displacement/math.tan(theta*math.pi/180)

    return depth


def objectDetect(img):

    red = img.reshape(-1, img.shape[-1])
    red = np.zeros(img.shape[1])
    green, blue = red, red

    # Extract the color channels
    blue = img[:, :, 0]
    green = img[:, :, 1]
    red = img[:, :, 2]

    # PNG extension images have BGR format
    if type == 'png':
        blue, red = red, blue

    # rgbIndex1 = blue+2.5*green-5.05*red
    rgbIndex = 4*(red-blue)-(0.1*green + 2.75*blue)

    # Extract water mask calculated from Index
    th, mask = cv2.threshold(
        rgbIndex, 0, 255, cv2.THRESH_BINARY)

    image_filter = np.zeros(img.shape)

    # Recreate image with water
    image_filter[:, :, 2] = mask                    # Water
    # image_filter[:, :, 1] = np.absolute(mask-255)   # Land

    cv2.imwrite('./mask.png', mask)

    return mask


def bgr_hsv(img):
    hsv1 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    result1 = img.copy()

    lower1 = np.array([0, 170, 80])
    upper1 = np.array([10, 255, 255])
    lower2 = np.array([160, 170, 80])
    upper2 = np.array([180, 255, 255])

    lower_mask = cv2.inRange(hsv1, lower1, upper1)
    upper_mask = cv2.inRange(hsv1, lower2, upper2)
    mask = lower_mask + upper_mask

    cv2.imwrite('./mask.png', mask)

    # result = cv2.bitwise_and(result, result, mask = full_mask)
    return mask


def main():
    path1 = './Dist_40/img5.jpeg'
    path2 = './Dist_40/img6.jpeg'

    imageA = cv2.imread(path1)
    imageB = cv2.imread(path2)
    # imageA = bgr_hsv(imageA)
    # imageB = bgr_hsv(imageB)
    output = congeal(imageA, imageB)
    print(output)
    depth = depth_estimation(output, 5.18)
    print('Depth: ' + str(depth))


if __name__ == '__main__':
    main()
