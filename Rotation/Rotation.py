import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from scipy.stats import entropy

class ImageCongealling:

    def __init__(self, image_shape):
        old_index = np.array(list(np.ndindex((image_shape[1], image_shape[0]))))
        ones = np.ones((old_index.shape[0],))
        self.old_index = np.column_stack((old_index, ones))
        self.img_width = image_shape[1]
        self.img_height= image_shape[0]

    def red_color_tracker(self, image):
        lower1 = np.array([0, 100, 20])
        upper1 = np.array([10, 255, 255])

        # upper boundary RED color range values; Hue (160 - 180)
        lower2 = np.array([160, 100, 20])
        upper2 = np.array([180, 255, 255])

        hsv_red = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        mask1 = cv2.inRange(hsv_red, lower1, upper1)
        mask2 = cv2.inRange(hsv_red, lower2, upper2)
        mask = mask2

        mask = cv2.medianBlur(mask, 5)

        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return mask, contours


    def rotate_on_y(self, img, angle_in_rad, focal_lenght=411.8125):

        old_index = self.old_index - np.array([self.img_width/2, self.img_height/2, 0]).transpose()

        Z = 1
        rotation_matrix = np.array([
            [np.cos(angle_in_rad), 0, np.sin(angle_in_rad)],
            [0, 1, 0],
            [-np.sin(angle_in_rad), 0, np.cos(angle_in_rad)]
        ])

        new_idx = np.multiply(old_index, np.array([Z/focal_lenght, Z/focal_lenght, 1]))
        new_idx = np.dot(new_idx, rotation_matrix)
        Z_rotated = new_idx[:, 2] #400 x 400
        trans3d_2d = np.array(
            [focal_lenght/Z_rotated, focal_lenght/Z_rotated]
        ).transpose()
        new_idx = new_idx[:, :-1]
        new_idx = np.multiply(new_idx, trans3d_2d)
        new_idx = new_idx + np.array([self.img_width / 2, self.img_height / 2]).transpose()

        new_idx = np.asarray(new_idx, dtype='int')

        const1 = img.shape[1] >new_idx[:, 0]
        const2 = img.shape[0] >new_idx[:, 1]
        const3 = new_idx[:, 0]>=0
        const4 = new_idx[:, 1]>=0

        in_frame_bool = const1*const2*const3*const4
        # TODO: UPPER CONSTRAINT IS MISSING

        new_img = np.zeros(img.shape)
        mask = np.zeros(img.shape)

        valid_new_idx = new_idx[in_frame_bool, :]
        valid_new_idx[:, [0, 1]] = valid_new_idx[:, [1, 0]]

        valid_old_idx = self.old_index[in_frame_bool, :][:, :-1]
        valid_old_idx[:, [0, 1]] = valid_old_idx[:, [1, 0]]

        valid_old_idx= np.asarray(valid_old_idx, dtype="int")

        new_img[tuple(zip(*valid_new_idx))] = img[tuple(zip(*valid_old_idx))]
        mask[tuple(zip(*valid_new_idx))] = 1

        return new_img, mask

    def rotate_on_y_uneff(self, img, angle_in_rad, focal_lenght):

        mask = np.zeros(img.shape)
        new_image = np.zeros(img.shape)
        Z = 10
        rotation_matrix = np.array([
            [np.cos(angle_in_rad), 0, np.sin(angle_in_rad)],
            [0, 1, 0],
            [-np.sin(angle_in_rad), 0, np.cos(angle_in_rad)]
        ])
        for i, j in np.ndindex(img.shape):
            X = (j - self.img_height/2) * Z /focal_lenght
            Y = (i - self.img_width/2)* Z / focal_lenght
            point_3d = np.array([X, Y, Z])
            rotated_point = point_3d @ rotation_matrix
            x =int((focal_lenght * rotated_point[0] / rotated_point[2]) + self.img_height/2)
            y = int((focal_lenght * rotated_point[1]/ rotated_point[2]) + self.img_width/2)
            if 0 <= x <img.shape[1] and 0 <= y <img.shape[0]:
                new_image[y, x] = img[int(i), int(j)]
                mask[y, x] = 1
        return new_image, mask

    def congealling(self, img1, img2, init_value, plots, num_iter=15, max_step_size = 3):

        img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
        img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)

        img1, _ = self.red_color_tracker(img1)
        img2, _ = self.red_color_tracker(img2)
        
        if plots:
            fig = plt.figure()
            fig.suptitle("Masked images")
            plt.subplot(1,2,1)
            plt.imshow(img1)
            plt.subplot(1,2,2)
            plt.imshow(img2)
            plt.show()

        previous_error = mean_squared_error(img1, img2)
        previous_angle = 0
        new_angle = init_value
        max_error = 0
        angle = []
        cost = []
        for i in range(num_iter):
            rot_img, mask = self.rotate_on_y(img1,  np.radians(new_angle))
            current_error = mean_squared_error(rot_img[mask==1], img2[mask==1])
            cost.append(current_error)
            angle.append(new_angle)
            max_error = max(current_error, max_error)
            delta_error = current_error-previous_error
            delta_angle = new_angle-previous_angle
            previous_error = current_error
            previous_angle = new_angle
            learning_rate = (current_error/max_error) * max_step_size
            new_angle = new_angle - learning_rate * np.sign(delta_error/delta_angle)
            print(f"#{i}: new_angle ={new_angle} // cost = {current_error}")

        if plots:
            plt.title("Learning curve")
            plt.plot(np.array(angle), np.array(cost))
            plt.xlabel("Angle")
            plt.ylabel("Error")
            plt.show()
            fig =plt.figure()
            fig.suptitle("Aligned images")
            plt.subplot(1, 2, 1)
            plt.imshow(img2)
            result_img, _ = self.rotate_on_y(img1, np.radians(angle[np.argmin(cost)]))
            plt.subplot(1, 2, 2)
            plt.imshow(result_img)
            plt.show()


        max_cost_idx = np.argmin(cost)
        return angle[max_cost_idx], cost[max_cost_idx]


