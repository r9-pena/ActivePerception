import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.io import imread
from skimage.color import rgb2gray
import time
import math
from scipy.stats import _entropy

# This method receives two images as input to perform congealing


def congeal(img1, img2):
    imageA, imageB = img1, img2
    array_size = np.shape(imageA)

    # Adjust window size to be compared. Ignore everything outside.
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
            cv2.imwrite('./pics/imgT.png', imageT)
            angle = i-1
            depth = depth_estimation(angle, 5.18)
            return angle
        else:
            current_error = new_error

    # imageA[:, start:end] = imageA[:, start:end]-imageA[:, start:end]
    cv2.imwrite('./pics/imgT.png', imageT)


# This function compares two images and returns the Mean-Squared Error Value
def mse(imageA, imageB):
    err = np.sum((imageA.astype('float') - imageB.astype('float'))**2)
    err /= float(imageA.shape[0] * imageB.shape[1])

    return err


# This function is used to re-center the transformed image
def translate(image, offset):
    M = np.float32([[1, 0, offset], [0, 1, 0]])
    imageT = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

    return imageT


# This funtion rotates the image by the input angle
def A_Trans(image, theta):
    print('-------------------------------------')
    print('Angle (degrees): ' + str(theta))
    theta = theta * math.pi/180
    h, w = image.shape[:2]
    diag = (h ** 2 + w ** 2) ** 0.5
    f = diag
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
    M = np.dot(R_M,  M)
    M = np.dot(Hp_M, np.dot(M, H_M))
    M = np.dot(M, inv_M)
    dsize = (w, h)
    imageT = cv2.warpPerspective(
        image, M, dsize)
    mid = imageT[200, :]
    zrs = w*3-np.count_nonzero(mid)
    offset = int(zrs/6)
    print(zrs)
    imageT = translate(imageT, offset)

    return imageT


# This function calculates the depth based on the robot displacement.
# The displacement can be measured or calculated from the wheel rotation.
def depth_estimation(theta, displacement):
    depth = displacement/math.tan(theta*math.pi/180)

    return depth


def main():
    path1 = './Dist_20/img1.jpeg'
    path2 = './Dist_20/img2.jpeg'

    imageA = cv2.imread(path1)
    imageB = cv2.imread(path2)
    output = congeal(imageA, imageB)
    print(output)
    # 5.18 is the calculated displacement. Adjust as needed.
    displacement = 5.18
    depth = depth_estimation(output, displacement)
    print('Depth: ' + str(depth))


if __name__ == '__main__':
    main()
