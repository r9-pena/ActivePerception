#!/usr/bin/env python3

import numpy as np
import cv2
import math

def contour (img1, img2):
    
    img1 = cv2.imread('img1.jpeg', 1)
    cv2.imshow("img1", img1)
    img2 = cv2.imread('img2.jpeg', 1)
    cv2.imshow('img2', img2)

    def bgr_hsv (img1, img2):
        hsv1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
        hsv2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
        result1 = img1.copy()
        result2 = img2.copy()

        lower1 = np.array([0, 170, 80])
        upper1 = np.array([10, 255, 255])
        lower2 = np.array([160, 170, 80])
        upper2 = np.array([180, 255, 255]) 

        lower_mask1 = cv2.inRange(hsv1, lower1, upper1)
        upper_mask1 = cv2.inRange(hsv1, lower2, upper2)
        lower_mask2 = cv2.inRange(hsv2, lower1, upper1)
        upper_mask2 = cv2.inRange(hsv2, lower2, upper2)
        full_mask1 = lower_mask1 + upper_mask1
        full_mask2 = lower_mask2 + upper_mask2

        result1 = cv2.bitwise_and(result1, result1, mask = full_mask1)
        result2 = cv2.bitwise_and(result2, result2, mask = full_mask2)
        cv2.imshow('mask', full_mask1)
        return full_mask1, full_mask2

    print(bgr_hsv(img1,img2)[0].shape)

    #binary image to grayscale
    mask1, mask2 = bgr_hsv(img1, img2)[0], bgr_hsv(img1, img2)[1]
    cv2.imwrite('mask1.jpeg', mask1)
    mask1_gray = cv2.imread('mask1.jpeg', 0)
    cv2.imwrite('mask2.jpeg', mask2)
    mask2_gray = cv2.imread('mask2.jpeg', 0)
    cv2.imshow('mask1_gray',mask1_gray)
    print(mask1_gray.shape)

    #finding the contour
    ret1, thresh1 = cv2.threshold(mask1_gray, 127, 255, 0)
    contour1, hierarchy1 = cv2.findContours(thresh1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    ret2, thresh2 = cv2.threshold(mask2_gray, 127, 255, 0)
    contour2, hierarchy2 = cv2.findContours(thresh2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(img1, contour1, -1, (0,255,0), 3)
    cv2.drawContours(img2, contour2, -1, (0,255,0), 3)
    cv2.imwrite('contour1.jpeg', img1)
    cv2.imwrite('contour2.jpeg', img2)

    #finding the center coordinate positions
    for c_pixel1 in contour1:
        (x,y), radius1 = cv2.minEnclosingCircle(c_pixel1)
        center1 = (int(x), int(y))
        radius1 = int(radius1)
        print(center1)

    for c_pixel2 in contour2:
        (x,y), radius2 = cv2.minEnclosingCircle(c_pixel2)
        center2 = (int(x), int(y))
        radius2 = int(radius2)
        print(center2)


    #pixel distance
    x1, y1 = center1[0], center1[1]
    x2, y2 = center2[0], center2[1]
    pixel_dist = x2-x1
    print(pixel_dist)

    #Center Marking
    img_c1 = img1
    img_c1[y1, x1] = (0,0,0)
    cv2.imwrite('img_c1.jpeg', img_c1)
    img_c2 = img2
    img_c2[y2, x2] = (0,0,0)
    cv2.imwrite('img_c2.jpeg', img_c2)

    #Image Translation
    img_T = img_c2
    T = np.float32([[1, 0, -pixel_dist], [0, 1, 0]])
    img_T = cv2.warpAffine(img_T, T, (500,400))
    cv2.imwrite('img_T.jpeg', img_T)
    #Depth Calculation
    d = 5.5
    c = math.pi * d
    Dist = c * 0.3
    f = 411.8125
    z = f*(Dist/pixel_dist)
    print(z)

    #Rotation through optical flow
    angle = -Dist/z
    angle_deg_optic = angle*180/math.pi
    print(angle*180/math.pi)

    #Rotation through trigonometry
    angle_T = math.atan(Dist/z)
    print(angle_T*180/math.pi)
    
    k = cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return angle_deg_optic, z