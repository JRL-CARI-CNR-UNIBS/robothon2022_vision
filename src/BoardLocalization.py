#! /usr/bin/env python3

import rospy
from std_srvs.srv import SetBool,SetBoolResponse
from manipulation_msgs.msg import Location
from manipulation_msgs.srv import AddLocations, AddLocationsRequest
from geometry_msgs.msg import Pose, Point, Quaternion, PoseArray, Transform, Vector3, TransformStamped

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
SERVICE_ADD_LOCATION_NAME = "/go_to_location_server/add_locations"

class BoardLocalization:

    def __init__(self):


        self.realsense=RealSense()
        # Retrieve camera parameters
        rospy.loginfo(YELLOW + "Waiting camera parameters" + END)

        self.realsense.getCameraParam()
        self.realsense.waitCameraInfo()

        # camera_parameters = {"height": 720,"width": 1280,"distortion_model": "plumb_bob", "D": [0.0, 0.0, 0.0, 0.0, 0.0],"K": [902.4752197265625, 0.0, 633.35498046875, 0.0, 901.7176513671875, 379.5818786621094, 0.0, 0.0, 1.0]}
        
        # self.realsense.setcameraInfo(camera_parameters)

        rospy.loginfo(GREEN + "Camera parameters retrived correctly" + END)

        #Estimated parameters
        self.depth = 589    # Estimated distance

        # rospy.wait_for_service(SERVICE_ADD_LOCATION_NAME)
        # self.add_location = rospy.ServiceProxy(SERVICE_ADD_LOCATION_NAME, AddLocations)

        #TfBroadcaster
        self.broadcaster = tf2_ros.StaticTransformBroadcaster()
        
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
        buttons_not_found = True
        while buttons_not_found:
            try:
                rospy.loginfo(GREEN +"Identifying red button..."+ END)
                RedButPos, BlueButtonPos, id_red_blue_contour  = getRedBlueButtonsNewVersion(saturation,b_col,contours_limited,crop_img, ScreenPos)
                rospy.loginfo(GREEN +"Identified red button"+ END)
                buttons_not_found=False
            except:
                rospy.loginfo(GREEN +"Repeting acquisition, screen identification buttons"+ END)   
                ##### Cosa orrenda         ######
                self.realsense.acquireOnce()
                rgb_frame = self.realsense.getColorFrame()
                crop_img = getRoi(rgb_frame,144,669,360,1150)
                hsv, lab, bw = getAllColorSpaces(crop_img)
                hue,saturation,value = cv2.split(hsv)
                l_col,a_col,b_col    = cv2.split(lab)
                ret,value_th = cv2.threshold(value,90,255,0)
                board_cnt, contours_limited, contours = getBoardContour(value_th)
                ScreenPos, idx_screen = getScreen(a_col,contours_limited)
                contours_limited.pop(idx_screen)     #Remove screen contour from list        
                ###### FIne cosa orrenda   ####
        print(RedButPos)
        print(BlueButtonPos)
        
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
        new_img = cv2.circle(crop_img, (RedButPos[0],RedButPos[1]), 5, color = (0, 0, 255), thickness = 2)
        new_img = cv2.circle(crop_img, (BlueButtonPos[0],BlueButtonPos[1]), 5, color = (255, 0, 0), thickness = 2)
        new_img = cv2.circle(crop_img, (KeyLockPos[0][0],KeyLockPos[0][1]), 5, color = (255, 0, 0), thickness = 2)

        cv2.imshow("Identificato",new_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


        ############### Computing board tf ##################
        rospy.loginfo(GREEN + "Computing tf" + END)

        red_button_camera = np.array(self.realsense.deproject(RedButPos[0],RedButPos[1],self.depth))/1000.0
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


        # self.pubTF(red_button_camera,"red_prima","camera_color_optical_frame")
        # self.pubTF(key_lock_camera,"keuy_prima","camera_color_optical_frame")
        # self.pubTF(screen_camera,"screen_prima","camera_color_optical_frame")
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
        static_transformStamped_board = self.getStaticTrasformStamped("base_link", "board",M_camera_board[0:3,-1] ,rotation_quat)

        # broadcaster.sendTransform(static_transformStamped_board)

        # self.broadcastTF(Quaternion(rotation_quat[0],rotation_quat[1],rotation_quat[2],rotation_quat[3]), Vector3(M_camera_board[0,-1],M_camera_board[1,-1],M_camera_board[2,-1]),"board", "base_link")

        ######################## anche con altro

        ###### Questo è il calcolo della tf della board rispetto camera senza sistemare la z, orientato male#############
        # x_axis = ( key_lock_camera - red_button_camera ) / np.linalg.norm( key_lock_camera - red_button_camera )

        # y_axis_first_approach = ( screen_camera - red_button_camera )
        # y_axis_norm = y_axis_first_approach - np.dot(y_axis_first_approach,x_axis)/(np.dot(x_axis,x_axis))*x_axis
        # y_axis_norm = y_axis_norm / np.linalg.norm(y_axis_norm)

        # z_axis = np.cross(x_axis,y_axis_norm)       
        
        
        # rot_mat_camera_board = np.array([x_axis,y_axis_norm,z_axis]).T
        # M_camera_board_only_rot = tf.transformations.identity_matrix()
        # M_camera_board_only_rot[0:-1,0:-1]=rot_mat_camera_board
        
        # M_camera_board_only_tra = tf.transformations.identity_matrix()
        # M_camera_board_only_tra[0:3,-1]=np.array([red_button_camera[0],red_button_camera[1],red_button_camera[2]])
        
        # M_camera_board = np.dot(M_camera_board_only_tra,M_camera_board_only_rot)
        
        # rotation_quat = tf.transformations.quaternion_from_matrix(M_camera_board)


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
        
        
        ######################################################################################

        rospy.loginfo(GREEN + "Published tf" + END)

        # location_to_add = Location()
        # location_to_add.name = "reference"
        # location_to_add.frame = "board"
        # pose_to_add = Pose()
        # pose_to_add.position = Point(0.137, 0.094, -0.155)
        # pose_to_add.orientation = Quaternion(0.000, 0.000, 0.959, -0.284)
        # location_to_add.pose = pose_to_add

        # locations_request = AddLocationsRequest()
        # locations_request.locations = [location_to_add]
        # # print(location_to_add)
        # try:
        #     srv_response = self.add_location(locations_request)
        # except rospy.ServiceException as e:
        #     rospy.loginfo(RED + "Error on add location srbv call" + END) 

        # if srv_response.results == 1:
        #     rospy.loginfo(GREEN + "Location added succesfully" + END)

        static_transformStamped_reference = self.getStaticTrasformStamped("board", "reference", [0.137, 0.094,-0.155],[0.0, 0.0, 0.959,-0.284])
        tf_key_lock = self.getStaticTrasformStamped("base_link", "key_lock", key_lock_world,rotation_quat)
        
        self.broadcaster.sendTransform([static_transformStamped_board,static_transformStamped_reference,tf_key_lock])
        # self.broadcastTF(Quaternion(0,0,0.959,-0.284), Vector3(0.137,0.094,-0.155), "reference","board")

        return SetBoolResponse(True,SUCCESSFUL)


    def get4Vector(self,vect):
        vet = np.array([0.0,0.0,0.0,1.0])
        vet[:-1] = vect
        return vet

    def getStaticTrasformStamped(self,header_frame_id_name, child_frame_id_name, tra,quat):
        static_transformStamped_board = geometry_msgs.msg.TransformStamped()
        static_transformStamped_board.header.stamp = rospy.Time.now()
        static_transformStamped_board.header.frame_id = header_frame_id_name
        static_transformStamped_board.child_frame_id = child_frame_id_name

        static_transformStamped_board.transform.translation.x = tra[0]
        static_transformStamped_board.transform.translation.y = tra[1]
        static_transformStamped_board.transform.translation.z = tra[2]
        static_transformStamped_board.transform.rotation.x = quat[0]
        static_transformStamped_board.transform.rotation.y = quat[1]
        static_transformStamped_board.transform.rotation.z = quat[2]
        static_transformStamped_board.transform.rotation.w = quat[3]
        return static_transformStamped_board



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
