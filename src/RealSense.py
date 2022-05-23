#! /usr/bin/env python3

import time
import rospy
import message_filters
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import CameraInfo, Image

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pyrealsense2 as rs2
from math import floor

# Realsense Topic
COLOR_FRAME_TOPIC = '/camera/color/image_raw'
DEPTH_ALIGNED_TOPIC = '/camera/aligned_depth_to_color/image_raw'
CAMERA_INFO_TOPIC = '/camera/aligned_depth_to_color/camera_info'

CAMERA_FRAME = "camera_color_optical_frame"

# Costant loginfo
PARAMETERS_LOG = 'Camera Parameters acquired \n  Parameters:{}'

class RealSense():
    """
    RealSense class for Subscribe interesting topic.
    """

    def __init__(self):
        """
        Class builder
        @param -
        @return RealSense RealSense object
        """
        self.bridge = CvBridge()
        self.colorFrame = None
        self.depthFrame = None

        # Gestione camera pyrealsense2
        self.intrinsics = None
        self.cameraInfoReceived = False
        self.frameAcquired = False
        
        self.frame_number = 0

    def callback(self,frameRgb,frameDepth):
        """
        Callback method to retrieve the content of the topic and convert it in cv2 format. Identify human KeyPoints.
        @param frameRgb : camera msg rgb
        @param frameDepth : camera msg depth
        """
        # Convertion from ros msg image to cv2 image
        colorFrame = self.bridge.imgmsg_to_cv2(frameRgb, desired_encoding="passthrough")
        depthFrame = self.bridge.imgmsg_to_cv2(frameDepth, desired_encoding="passthrough")
        frameDistance = self.bridge.imgmsg_to_cv2(frameDepth, desired_encoding="32FC1")

        self.colorFrame = colorFrame.copy()
        self.depthFrame = depthFrame.copy()
        self.frameDistance = frameDistance.copy()
       
    def callbackOnlyRgb(self,frameRgb):
        colorFrame = self.bridge.imgmsg_to_cv2(frameRgb, desired_encoding="passthrough")
        self.colorFrame = colorFrame.copy()
        
    def cameraInfoCallback(self,cameraInfo):
        """
        Callback for get Intrinsic Parameter of Camera and create intrinsics object (pyrealsense2 library)
        """
        self.intrinsics = rs2.intrinsics()
        self.intrinsics.width = cameraInfo.width
        self.intrinsics.height = cameraInfo.height
        self.intrinsics.ppx = cameraInfo.K[2]
        self.intrinsics.ppy = cameraInfo.K[5]
        self.intrinsics.fx = cameraInfo.K[0]
        self.intrinsics.fy = cameraInfo.K[4]

        if cameraInfo.distortion_model == 'plumb_bob':
            self.intrinsics.model = rs2.distortion.brown_conrady
        elif cameraInfo.distortion_model == 'equidistant':
            self.intrinsics.model = rs2.distortion.kannala_brandt4
        self.intrinsics.coeffs = [i for i in cameraInfo.D]
        self.cameraInfoReceived = True

        #Reference frame
        self.frame_id=cameraInfo.header.frame_id
        rospy.loginfo("Camera frame id: {}".format(self.frame_id))

    def waitCameraInfo(self):
        while not self.cameraInfoReceived:
            pass
        self.sub_info.unregister()
        rospy.loginfo(PARAMETERS_LOG.format(self.intrinsics))

    def acquire(self):
        """
        Method for acquiring in syncronization way rgb and depth frame
        """
        print("Dentro Acquire")
        self.subcriberColorFrame = message_filters.Subscriber(COLOR_FRAME_TOPIC, Image)
        self.subcriberDepthFrame = message_filters.Subscriber(DEPTH_ALIGNED_TOPIC, Image)
        # Subscriber Synchronization
        subSync = message_filters.TimeSynchronizer([self.subcriberColorFrame, self.subcriberDepthFrame], queue_size=10)
        #Call callback sincronized
        subSync.registerCallback(self.callback)

        rospy.spin()

    def acquireOnlyRgb(self):
        """
        Method for acquiring in syncronization way rgb
        """
        self.subcriberColor = rospy.Subscriber(COLOR_FRAME_TOPIC, Image, self.callbackOnlyRgb, queue_size=1)

    def acquireOnce(self):
        """Method for acquiring only once frame rgb
        """
        rospy.loginfo("Waiting frame ...")
        frameRgb = rospy.wait_for_message(COLOR_FRAME_TOPIC, Image, timeout=None)
        colorFrame = self.bridge.imgmsg_to_cv2(frameRgb, desired_encoding="passthrough")
        self.colorFrame = colorFrame.copy()
        rospy.loginfo("Frame recived...")
    
    def saveImage(self, folder_path):
        """Method for saving only one frame to  desired location

        Args:
            filename (String): path of the image to saved
        """
        
        self.acquireOnce()
        cv2.imwrite(folder_path + "frame_" +str(self.frame_number) + ".png",cv2.cvtColor(self.colorFrame, cv2.COLOR_RGB2BGR))
        # cv2.imwrite(folder_path + "frame_" +str(self.frame_number) + ".png", self.colorFrame)
        self.frame_number += 1
    
    def showColorFrame(self,nameWindowRgb):
        """Show RGB Frame in a windows

        Args:
            nameWindowRgb (String): Name of windows
        """
        imgImshow = cv2.cvtColor(self.colorFrame, cv2.COLOR_RGB2BGR)
        cv2.imshow(nameWindowRgb, imgImshow)        
        
    def showImage(self,nameWindowRgb,nameWindowDepth):
        """
        Method for showing the image
        """
        #Rgb -> Bgr convertion for cv2 imshow
        imgImshow = cv2.cvtColor(self.colorFrame, cv2.COLOR_RGB2BGR)
        cv2.imshow(nameWindowRgb, imgImshow)
        cv2.imshow(nameWindowDepth,self.depthFrame)
    
    def getCameraParam(self):
        self.sub_info = rospy.Subscriber(CAMERA_INFO_TOPIC,CameraInfo,self.cameraInfoCallback)
    
    def stop(self):
        '''Method to disconnect the subscribers from kinect2_bridge topics, to release
            some memory and avoid filling up the queue.'''
        self.subcriberColorFrame.unregister()
        self.subcriberDepthFrame.unregister()

    def deproject(self, x,y,depth):
        # Deprojection : Image frame -> Camera frame (camera_color_optical_frame)
        # print(self.intrinsics)
        # print([x,y])
        # print(depth)
        deprojection=rs2.rs2_deproject_pixel_to_point(self.intrinsics,[x,y], depth)
        # print(deprojection)
        return deprojection
