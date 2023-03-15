from picamera import PiCamera
from buildhat import MotorPair
from buildhat import Motor
import time
import Contour
import cv2
import math

#Depth Calculation
DIAMETER = 5.5
CIRCUMFERENCE = math.pi * DIAMETER



class LegoRobot():
    def __init__(self):
        self.pair = MotorPair('A', 'B') # Wheel motors
        self.pair.set_default_speed(50) # Set wheel motor speed
        self.motor_c = Motor('C') # Camera motor
        self.motor_c.run_to_position(0) # Reset camera to position 0
        #initial picture taking
        self.camera = PiCamera()
        self.camera.vflip=True
        self.camera.hflip=True
        self.camera.resolution = (500,400)
        self.counter = 0
        self.algorithm = {
            'contour':Contour,
            #'matrix':Matrix,
            #'rotation':Rotation
            }


    def object_align(self, wheel_rotation, select):
        # first picture taking
        self.check_centering()
        path1 = './pic' + str(self.counter) +'.jpg'
        img1 = self.camera.capture(path1)
        self.counter+=1
        
        # inital displacement movement
        self.pair.run_for_rotations(wheel_rotation, speedl=30, speedr = -30)
        time.sleep(1)
        
        # second picture taking
        path2 = './pic' + str(self.counter) +'.jpg'
        img2 = self.camera.capture(path2)
        self.counter+=1
        
        # process image
        self.image_process(select,path1,path2)


    def check_centering(self):
        self.camera.start_preview()
        input('Are you ready??')
        self.camera.stop_preview()
        return


    def image_process(self,select,path1,path2):
        img1,img2 = cv2.imread(path1),cv2.imread(path2)
        rot, depth = self.algorithm[select].congeal(img1,img2)
        print(rot,depth)

def main():
    wheel_rotation = 0.3
    distance_traveled = CIRCUMFERENCE * wheel_rotation
    robot = LegoRobot()
    robot.object_align(wheel_rotation,'contour')

if __name__ == "__main__":
    main()






#################################################





f = 411.8125
#z = f*(Dist/pixel)
#print(z)

#Rotation with trig
#angle = 90 - math.atan(z/Dist)*180/math.pi
#print (angle)

#print(time.time())
#print(time.process_time())

#focal length calculation
#focal length (mm) = 3.04mm
#Sony IMX219 sensor width = 3.691mm
#New image width in pixels = 500 pxs
# focal_pixel = (focal length/sensorwidth)*imagewidth
#             = 411.8125 pixels

#Robot Displacement
#Wheel Diameter = 5.5cm
#Circumference = 17.2782cm