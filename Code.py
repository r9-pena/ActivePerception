#from picamera import PiCamera
from buildhat import MotorPair
from buildhat import Motor
import time
from Contour import Contour
from Matrix import Matrix
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
        self.camera = cv2.VideoCapture(0)
        # self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 500)
        # self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 400)
        #self.camera.flip()
        #self.camera.hflip=True
        #self.camera.resolution = (500,400)
        self.counter = 0
        self.algorithm = {
            '1':Contour,
            '2':Matrix,
            #'rotation':Rotation
            }


    def object_align(self, wheel_rotation, select):
        # first picture taking
        if self.check_centering() == False:
            print('Exiting program')
            return False
        path1 = './pics/pic' + str(self.counter) +'.jpeg'
        ret, frame = self.camera.read()
        print(self.camera.get(cv2.CAP_PROP_FRAME_COUNT))
        self.camera.release()
        cv2.imwrite(path1,frame)
        self.counter+=1
        
        # inital displacement movement
        self.pair.run_for_rotations(wheel_rotation, speedl=30, speedr = -30)
        time.sleep(5)
        
        # second picture taking
        print(self.camera.get(cv2.CAP_PROP_FRAME_COUNT))
        path2 = './pics/pic' + str(self.counter) +'.jpeg'
        self.camera = cv2.VideoCapture(0)
        ret, frame2 = self.camera.read()
        self.camera.release()
        cv2.imwrite(path2,frame2)
        self.counter+=1
        
        # process image
        self.image_process(select,path1,path2)


    def check_centering(self):
        #self.camera.start_preview()
        if self.camera.isOpened() == False:
            print('Camera not working')
            return False
        input('Are you ready??')
        while(True):
            # Capture the video frame
            # by frame
            ret, frame = self.camera.read()
        
            # Display the resulting frame
            cv2.imshow('frame', frame)   
            # the 'q' button is set as the
            # quitting button you may use any
            # desired button of your choice
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        return True


    def image_process(self,select,path1,path2):
        img1,img2 = cv2.imread(path1),cv2.imread(path2)
        img1 = cv2.flip(img1,-1)
        img2 = cv2.flip(img2,-1)
        rot, depth = self.algorithm[select].congeal(img1,img2)
        print(rot,depth)

    def user_selection(self):
        print('Please select algorithm to run')
        print('1.Contour 2.Matrix 3.Rotation')
        sel = input()
        sel = str(sel)
        return sel

def main():
    wheel_rotation = 0.3
    distance_traveled = CIRCUMFERENCE * wheel_rotation
    robot = LegoRobot()
    if robot == False:
        return
    # Ask user to choose algorithm to be used
    selection = robot.user_selection()
    # Begin objec tracking
    robot.object_align(wheel_rotation,selection)
    # Release camera object after program conclusion
    robot.camera.release()
    # Destroy all the windows
    cv2.destroyAllWindows()

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