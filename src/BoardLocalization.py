#! /usr/bin/env python3

import rospy
from std_srvs.srv import SetBool,SetBoolResponse

from RealSense import RealSense

import tf2_ros
import tf
import geometry_msgs.msg

import numpy as np

import cv2

from utils import *

GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
BOLD = '\033[1m'
END = '\033[0m'


SERVICE_CALLBACK = GREEN + "Service call {} received" + END
PARAM_NOT_DEFINED_ERROR = "Parameter: {} not defined"
SUCCESSFUL = "Successfully executed"
NOT_SUCCESSFUL = "Not Successfully executed"

SERVICE_NAME = "/robothon2022/board_localization"

class BoardLocalization:

    def __init__(self):


        self.realsense=RealSense()

        # Retrieve camera parameters
        rospy.loginfo(YELLOW + "Waiting camera parameters" + END)
        self.realsense.getCameraParam()
        self.realsense.waitCameraInfo()
        rospy.loginfo(GREEN + "Camera parameters retrived correctly" + END)

        #Estimated parameters
        self.depth = 589    # Estimated distance

        rospy.loginfo(GREEN + "Service alive ...." + END)

    def callback(self,request):

        rospy.loginfo(SERVICE_CALLBACK.format(SERVICE_NAME))

        #Acquire the rgb-frame
        self.realsense.acquireOnce()

        rospy.loginfo(RED + "INIZIO A IDENTIFICARE" + END)
        rgb_frame = self.realsense.getColorFrame()
        cv2.imshow("frame acq", rgb_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        ################# Macro ROI #################
        crop_img = getRoi(rgb_frame,144,669,360,1150)
        hsv, lab, bw = getAllColorSpaces(crop_img)
        #############################################

        hue,saturation,value = cv2.split(hsv)
        l_col,a_col,b_col    = cv2.split(lab)

        ret,value_th = cv2.threshold(value,90,255,0)

        ################ Board contour ##############
        board_cnt, contours_limited, contours = getBoardContour(value_th)
        #############################################

        ################ Identify screen ############

        ScreenPos, idx_screen = getScreen(a_col,contours_limited)

        contours_limited.pop(idx_screen)     #Remove screen contour from list

        rospy.loginfo(GREEN + "Screen position: {}".format(ScreenPos) + END)
        new_img = getRoi(rgb_frame,144,669,360,1150)
        new_img = cv2.circle(crop_img, (ScreenPos[0][0],ScreenPos[0][1]), 5, color = (255, 0, 0), thickness = 2)
        # new_img = cv2.circle(crop_img, (RedBlueButPos[0][0],RedBlueButPos[0][1]), 5, color = (255, 0, 0), thickness = 2)
        # new_img = cv2.circle(crop_img, (KeyLockPos[0][0],KeyLockPos[0][1]), 5, color = (255, 0, 0), thickness = 2)

        cv2.imshow("Identificato",new_img)
        cv2.imshow("TH",value_th)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        #crop_img = getRoi(rgb_frame,144,669,360,1150)

        #new_img = cv2.circle(crop_img, (ScreenPos[0][0],ScreenPos[0][1]), 5, color = (255, 0, 0), thickness = 2)
        #new_img = cv2.circle(crop_img, (RedBlueButPos[0][0],RedBlueButPos[0][1]), 5, color = (255, 0, 0), thickness = 2)

        ##############################################


        ################## Identify red button #######
        rospy.loginfo(GREEN +"Identifying red button..."+ END)
        RedBlueButPos, id_red_blue_contour  = getRedBlueButtons(saturation,b_col,contours_limited,crop_img, ScreenPos)
        rospy.loginfo(GREEN +"Identified red button"+ END)

        contours_limited.pop(id_red_blue_contour)

        crop_img = getRoi(rgb_frame,144,669,360,1150)



        #############<##### Identify key lock   #######
        rospy.loginfo(GREEN +"Identifying KeyLock..."+ END)
        KeyLockPos , id_circle    = getKeyLock(l_col,contours_limited,crop_img,ScreenPos,crop_img)
        rospy.loginfo(GREEN +"Identified red button"+ END)

        ##############################################

        # Crop image (in order to get the interested area)
        new_img = getRoi(rgb_frame,144,669,360,1150)
        new_img = cv2.circle(crop_img, (ScreenPos[0][0],ScreenPos[0][1]), 5, color = (255, 0, 0), thickness = 2)
        new_img = cv2.circle(crop_img, (RedBlueButPos[0][0],RedBlueButPos[0][1]), 5, color = (255, 0, 0), thickness = 2)
        new_img = cv2.circle(crop_img, (KeyLockPos[0][0],KeyLockPos[0][1]), 5, color = (255, 0, 0), thickness = 2)

        cv2.imshow("Identificato",new_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


        ############### Computing board tf ##################
        rospy.loginfo(GREEN + "Computing tf" + END)

        red_button_camera = np.array(self.realsense.deproject(RedBlueButPos[0][0],RedBlueButPos[0][1],self.depth))/1000.0
        key_lock_camera = np.array(self.realsense.deproject(KeyLockPos[0][0],KeyLockPos[0][1],self.depth))/1000.0
        screen_camera = np.array(self.realsense.deproject(ScreenPos[0][0],ScreenPos[0][1],self.depth))/1000.0

        listener = tf.TransformListener()
        rospy.sleep(1.0)
        found = False
        while not found:
            print("prova")
            try:
                (trans,rot) = listener.lookupTransform('base_link', 'camera_color_optical_frame', rospy.Time(0))
                found = True
                print("Trovata")
            except (tf.LookupException, tf.ConnectivityException):
                print("non ce un cazzoo")
                rospy.sleep(0.5)

        rospy.loginfo(YELLOW + "Trasformata camera_color_optical_frame -> base_link \n :{}".format(trans) + RED)
        rospy.loginfo(YELLOW + "Trasformata camera_color_optical_frame -> base_link \n :{}".format(rot) + RED)
        
        trans_world_camera = tf.transformations.translation_matrix(trans)
        rot_world_camera = tf.transformations.quaternion_matrix(rot)
        print(trans_world_camera)
        print(rot_world_camera)
        M_world_camera = np.dot(trans_world_camera,rot_world_camera)
        print(M_world_camera)


        self.pubTF(red_button_camera,"red_prima","camera_color_optical_frame")
        self.pubTF(key_lock_camera,"keuy_prima","camera_color_optical_frame")
        self.pubTF(screen_camera,"screen_prima","camera_color_optical_frame")
        print("red button respect camera")
        print(red_button_camera)
        red_button_world = np.dot(M_world_camera, self.get4Vector(red_button_camera))
        key_lock_world = np.dot(M_world_camera, self.get4Vector(key_lock_camera))
        screen_world = np.dot(M_world_camera, self.get4Vector(screen_camera))


        red_button_world = red_button_world[0:-1]
        red_button_world[-1] = 0.1
        key_lock_world = key_lock_world[0:-1]
        key_lock_world[-1] = 0.1
        screen_world = screen_world[0:-1]
        screen_world[-1] = 0.1
        
        print(red_button_world)
        print(key_lock_world)
        print(screen_world)

        self.pubTF(red_button_world,"red_dopo","base_link")
        self.pubTF(key_lock_world,"key","base_link")
        self.pubTF(screen_world,"screen","base_link")


        # red_button_camera = np.array(self.realsense.deproject(RedBlueButPos[0][0],RedBlueButPos[0][1],self.depth))
        # key_lock_camera = np.array(self.realsense.deproject(KeyLockPos[0][0],KeyLockPos[0][1],self.depth))
        # screen_camera = np.array(self.realsense.deproject(ScreenPos[0][0],ScreenPos[0][1],self.depth))

        x_axis = ( key_lock_world - red_button_world ) / np.linalg.norm( key_lock_world - red_button_world )
        print(x_axis)
        y_axis_first_approach = ( screen_world - red_button_world )
        y_axis_norm = y_axis_first_approach - np.dot(y_axis_first_approach,x_axis)/(np.dot(x_axis,x_axis))*x_axis
        y_axis_norm = y_axis_norm / np.linalg.norm(y_axis_norm)
        print(y_axis_norm)

        z_axis = np.cross(x_axis,y_axis_norm)       
        print(z_axis)
        
        rot_mat_camera_board = np.array([x_axis,y_axis_norm,z_axis]).T
        M_camera_board_only_rot = tf.transformations.identity_matrix()
        M_camera_board_only_rot[0:-1,0:-1]=rot_mat_camera_board
        
        M_camera_board_only_tra = tf.transformations.identity_matrix()
        M_camera_board_only_tra[0:3,-1]=np.array([red_button_world[0],red_button_world[1],red_button_world[2]])
        
        M_camera_board = np.dot(M_camera_board_only_tra,M_camera_board_only_rot)
        
        rotation_quat = tf.transformations.quaternion_from_matrix(M_camera_board)


        ################## Broadcast board tf ############
        rospy.loginfo(GREEN + "Publishing tf" + END)
        broadcaster = tf2_ros.StaticTransformBroadcaster()
        static_transformStamped = geometry_msgs.msg.TransformStamped()
        static_transformStamped.header.stamp = rospy.Time.now()
        static_transformStamped.header.frame_id = "base_link"
        static_transformStamped.child_frame_id = "board"

        static_transformStamped.transform.translation.x = M_camera_board[0,-1]
        static_transformStamped.transform.translation.y = M_camera_board[1,-1]
        static_transformStamped.transform.translation.z = M_camera_board[2,-1]
        static_transformStamped.transform.rotation.x = rotation_quat[0]
        static_transformStamped.transform.rotation.y = rotation_quat[1]
        static_transformStamped.transform.rotation.z = rotation_quat[2]
        static_transformStamped.transform.rotation.w = rotation_quat[3]

        broadcaster.sendTransform(static_transformStamped)



        ######################## anche con altro

        x_axis = ( key_lock_camera - red_button_camera ) / np.linalg.norm( key_lock_camera - red_button_camera )

        y_axis_first_approach = ( screen_camera - red_button_camera )
        y_axis_norm = y_axis_first_approach - np.dot(y_axis_first_approach,x_axis)/(np.dot(x_axis,x_axis))*x_axis
        y_axis_norm = y_axis_norm / np.linalg.norm(y_axis_norm)

        z_axis = np.cross(x_axis,y_axis_norm)       
        
        
        rot_mat_camera_board = np.array([x_axis,y_axis_norm,z_axis]).T
        M_camera_board_only_rot = tf.transformations.identity_matrix()
        M_camera_board_only_rot[0:-1,0:-1]=rot_mat_camera_board
        
        M_camera_board_only_tra = tf.transformations.identity_matrix()
        M_camera_board_only_tra[0:3,-1]=np.array([red_button_camera[0],red_button_camera[1],red_button_camera[2]])
        
        M_camera_board = np.dot(M_camera_board_only_tra,M_camera_board_only_rot)
        
        rotation_quat = tf.transformations.quaternion_from_matrix(M_camera_board)


        ################## Broadcast board tf ############
        # rospy.loginfo(GREEN + "Publishing tf" + END)
        # broadcaster = tf2_ros.StaticTransformBroadcaster()
        # static_transformStamped = geometry_msgs.msg.TransformStamped()
        # static_transformStamped.header.stamp = rospy.Time.now()
        # static_transformStamped.header.frame_id = "camera_color_optical_frame"
        # static_transformStamped.child_frame_id = "board_respect_camera"

        # static_transformStamped.transform.translation.x = M_camera_board[0,-1]
        # static_transformStamped.transform.translation.y = M_camera_board[1,-1]
        # static_transformStamped.transform.translation.z = M_camera_board[2,-1]
        # static_transformStamped.transform.rotation.x = rotation_quat[0]
        # static_transformStamped.transform.rotation.y = rotation_quat[1]
        # static_transformStamped.transform.rotation.z = rotation_quat[2]
        # static_transformStamped.transform.rotation.w = rotation_quat[3]

        # broadcaster.sendTransform(static_transformStamped)

        rospy.loginfo(GREEN + "Published tf" + END)


        # try:
        return SetBoolResponse(True,SUCCESSFUL)
        # except :
        #     return SetBoolResponse(False,NOT_SUCCESSFUL)

    def get4Vector(self,vect):
        vet = np.array([0.0,0.0,0.0,1.0])
        vet[:-1] = vect
        return vet

    def pubTF(self,punto,nome, header_frame_id):
        import tf2_ros
        from geometry_msgs.msg import Pose, Point, Quaternion, PoseArray, Transform, Vector3, TransformStamped

        print("dentro")
        t0=Transform()
        t0TS=TransformStamped()

        t0.rotation=Quaternion(0,0,0,1)
        t0.translation=Vector3(punto[0],punto[1],punto[2])
        t0TS.header.frame_id=header_frame_id
        t0TS.header.stamp= rospy.Time.now()
        t0TS.child_frame_id=nome
        t0TS.transform=t0
        br = tf2_ros.TransformBroadcaster()
        rospy.sleep(2.0)
        br.sendTransform(t0TS)
        # broadcaster = tf2_ros.StaticTransformBroadcaster()
        # static_transformStamped = geometry_msgs.msg.TransformStamped()
        # static_transformStamped.header.stamp = rospy.Time.now()
        # static_transformStamped.header.frame_id = "base_link"
        # static_transformStamped.child_frame_id = nome
        #
        # static_transformStamped.transform.translation.x = punto[0]
        # static_transformStamped.transform.translation.y = punto[1]
        # static_transformStamped.transform.translation.z = punto[2]
        # static_transformStamped.transform.rotation.x = 0
        # static_transformStamped.transform.rotation.y = 0
        # static_transformStamped.transform.rotation.z = 0
        # static_transformStamped.transform.rotation.w = 1
        #
        # broadcaster.sendTransform(static_transformStamped)


def main():

    rospy.init_node("board_localization")

    # Retrieve nedded rosparam
    # try:
    #     param_name=rospy.get_param("~rosparam_name")
    # except KeyError:
    #     rospy.logerr(RED + PARAM_NOT_DEFINED_ERROR.format("mongo_database") + END)
    #     return 0
    board_localization = BoardLocalization()
    # Rosservice

    rospy.Service(SERVICE_NAME,SetBool,board_localization.callback)
    # print("dentro")
    # try:
    #     print("prova")
    #     board_localization = BoardLocalization()
    # except:
    #     print("eccez")
    #     return 0     #Connection to db failed



    rospy.spin()

if __name__ == "__main__":
    main()
