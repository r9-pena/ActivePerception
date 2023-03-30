from picamera import PiCamera
from buildhat import MotorPair
from buildhat import Motor
import numpy as np
import time
from image_congealling import ImageCongealling
import cv2

pair = MotorPair('A', 'B')
pair.set_default_speed(50)
#pair.run_to_position(0,0)

motor_c = Motor('C')
#motor_b = Motor('B')
motor_c.run_to_position(0)
#motor_b.run_to_position(0)

#Initialize Image Congealling
image_congealling = ImageCongealling((400, 500))

#initial picture taking
camera = PiCamera()
camera.vflip=True
camera.hflip=True
camera.resolution = (500,400)
camera.start_preview()
time.sleep(10)
img1 = camera.capture('./pic1.jpg')
camera.stop_preview()

#inital displacement movement
N_rotation = 0.25
pair.run_for_rotations(N_rotation, speedl=30, speedr=-30)


#second picture taking
time.sleep(1)
img2 = camera.capture('./pic2.jpg')


# Use
path1 = './pic1.jpg'
path2 = './pic2.jpg'
img1,img2 = cv2.imread(path1),cv2.imread(path2)

angle, cost = image_congealling.congealling(img1, img2, init_value=-3, plots=True, num_iter=30)
print(f"The camera must be rotated {angle}°")

motor_c.run_to_position(0)

if angle>0:
    print(angle)
    motor_c.run_to_position(np.abs(int(angle)), 15, blocking = False, direction = 'anticlockwise')
else:
    motor_c.run_to_position(np.abs(int(angle)), 15, blocking = False, direction = 'clockwise')

# Wheel Diameter = 5.5cm
# Circumference = 17.2782cm
circunference_wheel=17.2782 #cm 
    
estimated_depth = (circunference_wheel * N_rotation)/np.radians(angle)
print(f"Estimated depth = {estimated_depth}")
camera.start_preview()
time.sleep(10)
camera.stop_preview()

print(f"Process time: {time.process_time()}")



