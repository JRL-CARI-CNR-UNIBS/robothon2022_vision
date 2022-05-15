#! /usr/bin/env python3

import rospy
import cv2
import os
import numpy as np

GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
BOLD = '\033[1m'
END = '\033[0m'


SERVICE_CALLBACK = GREEN + "Service call {} received" + END
PARAM_NOT_DEFINED_ERROR = "Parameter: {} not defined"
SUCCESSFUL = "Successfully executed"
NOT_SUCCESSFUL = "Not Successfully executed"
USER_QUESTION = YELLOW + "Do you want to save another image?" + END


        
def main():

    rospy.init_node("offline_board_localization")

    # Retrieve nedded rosparam     
    # try:
    #     images_folder_path=rospy.get_param("~images_path")   
    # except KeyError:   
    #     rospy.logerr(RED + PARAM_NOT_DEFINED_ERROR.format("images_path") + END)
    #     return 0
    
    images_folder_path = "/home/samuele/projects/robothon_2022_ws/src/robothon2022_vision/file"
    
    for single_image_name in os.listdir(images_folder_path):
        img = cv2.imread(os.path.join(images_folder_path,single_image_name))      

        crop_img      = img[144:669, 358:1150]
        crop_img_copy = img[144:669, 358:1150].copy()
    
        print(img.shape)
        cv2.imshow("img croppata",crop_img)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
        bw  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        
        
        hue,saturation,value = cv2.split(hsv)
        l_col,a_col,b_col    = cv2.split(lab)

        
        # Threshold and contours 
        ret,value_th = cv2.threshold(value,90,255,0)
        cv2.imshow('VALUE TH', value_th)
        contours, hierarchy = cv2.findContours(value_th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        contours = np.asarray(contours)
        #TODO controlla che estrai veramente la board
        contours = sorted(contours, key=lambda x: cv2.contourArea(x))
        contours_limited = contours[-8:]


        board_cnt = contours[-2]

        x,y,w,h = cv2.boundingRect(board_cnt)
        ROI_board = value[y:y+h, x:x+w]
 
        
        cv2.drawContours(img, board_cnt, -1, (0,255,0), 3)
        cv2.imshow(single_image_name,img)
        cv2.imshow('HSV image', hsv)
        cv2.imshow('LAB image', lab)    
        cv2.imshow('BW image', bw)
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        
if __name__ == "__main__":
    main()
    
    
    