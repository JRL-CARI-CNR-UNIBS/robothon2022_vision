#! /usr/bin/env python3

import rospy
import cv2
import os
import numpy as np
from sklearn.cluster import KMeans
from math import cos,sin
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

def getScreen(a_col_in,contours_in):
    passed_imgs = 0
    max_idx     = 0
    max_val     = 0
    set_butt    = False
    for idx,cnt in enumerate(contours_in):
        area = cv2.contourArea(cnt)
        if (area > 15000) or (area < 1000):
            continue
        passed_imgs += 1
        mask = np.zeros(a_col_in.shape[:2],np.uint8)
        cv2.drawContours(mask, [cnt], 0, (255,255,255), -1)
        #cv2.imshow("screen", mask)
        
        ROI = cv2.bitwise_and(a_col_in,a_col_in,mask = mask)
        dst = cv2.inRange(ROI, 150, 255)
        no_brown = float(cv2.countNonZero(dst))
        tmp_val = no_brown/float(ROI.shape[0] * ROI.shape[1])
        # print(tmp_val)

        if tmp_val > max_val:
            max_val = tmp_val
            max_idx = idx
        else:
            continue

    M = cv2.moments(contours_in[max_idx])
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    return np.array([[cX,cY]])

def getRedBlueButtons(value_in_gray,b_col_in,contours_in):
    passed_imgs = 0
    butt_idx    = 0
    std_max     = 0
    set_butt    = False
    for idx,cnt in enumerate(contours_in):
        area = cv2.contourArea(cnt)
        if (area > 5000) or (area < 1000):
            continue
        passed_imgs += 1
        x,y,w,h = cv2.boundingRect(cnt)
        ROI = b_col_in[y:y+h, x:x+w]
        # cv2.imshow('ROI', ROI)
        # while cv2.waitKey(33) != ord('a'):
        #     time.sleep(0.5)

        flattened = ROI.reshape((ROI.shape[0] * ROI.shape[1], 1))
        cv2.imshow("flattened red blue button",ROI )
        clt = KMeans(n_clusters = 3)
        clt.fit(flattened)
        # print(np.std(clt.cluster_centers_))
        if np.std(clt.cluster_centers_) > std_max:
            butt_idx = idx
            std_max  = np.std(clt.cluster_centers_)
        else:
            continue


    x,y,w,h           = cv2.boundingRect(contours_in[butt_idx])
    butt_image_b      = b_col_in[y:y+h, x:x+w]
    shift             = np.asarray([x,y])
    butt_image_gray   = value_in_gray[y:y+h, x:x+w]
    # val_mid           = value_in_gray[int(value_in_gray.shape[1]/2),int(value_in_gray.shape[0]/2)]
    # corn_width        = 3

    circles = []

    ret, thresh = cv2.threshold(butt_image_gray, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)    

    contours = sorted(contours, key=lambda x: cv2.contourArea(x))
    mask = np.zeros(butt_image_gray.shape[:2],np.uint8)
    for cnt in contours:
        if cv2.contourArea(cnt) < 200:
            continue

        M = cv2.moments(cnt)
        try:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            circles.append(np.array([cX,cY]))
            cv2.circle(mask,(cX,cY),2,(255,255,255),1)  
        except:
            pass

    if len(circles) != 2:
        print("More or less than 2 buttons found!!!!!!",len(circles))


    if False:
        cv2.namedWindow('image')
        def nothing(x):
            pass
        cv2.createTrackbar('Param 1','image',100,200,nothing)
        cv2.createTrackbar('Param 2','image',30,100,nothing)

        while(1):
            butt_image_gray_copy=butt_image_gray.copy()
            cv2.imshow('image',butt_image_gray_copy)
            #To Get Parameter values from Trackbar Values
            para1 = cv2.getTrackbarPos('Param 1','image')
            para2 = cv2.getTrackbarPos('Param 2','image')

            
            try:
                print(para1,para2)
                circles = cv2.HoughCircles(butt_image_gray_copy,cv2.HOUGH_GRADIENT,1,15,param1=para1,param2=para2,minRadius=10,maxRadius=20)
                circles = np.uint16(np.around(circles))
                print(circles)
                for i in circles[0,:]:
                    cv2.circle(butt_image_gray_copy,(int(i[0]),int(i[1])),i[2],(0,0,0),1)
                    cv2.circle(butt_image_gray_copy,(int(i[0]),int(i[1])),2,(0,0,0),1)
            except:
                traceback.print_exc()
                pass
            #For drawing Hough Circles
            
            cv2.imshow('image', butt_image_gray_copy)
            cv2.waitKey(5)

    DEBUG_BUTTONS = False
    if DEBUG_BUTTONS:
        for i in circles:
            cv2.circle(butt_image_gray,(int(i[0]),int(i[1])),2,(0,0,0),1)   
    
        cv2.imshow('ROI_buttons_cluster', butt_image_b)
        cv2.imshow('ROI_buttons_v', butt_image_gray)



    # if butt_image_b[circles[0][1],circles[0][0]] > butt_image_b[circles[1][1],circles[1][0]]:
    #     return(np.array([circles[0] + shift,circles[1] + shift]))
    # else:
    #     return(np.array([circles[1] + shift,circles[0] + shift]))

    
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

        crop_img      = img[144:669, 360:1150]
        crop_img_copy = img[144:669, 360:1150].copy()
    
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
 
        #non fatto il get opt threshold
        
        # #Get Screen position
        # cv2.imshow("A COL PER GET SCREEN", a_col)
        # ScreenPos = getScreen(a_col,contours_limited)

        ################# BOARD MASK ON REAL IMAGE ##################
        board_mask = np.zeros(value.shape[:2],np.uint8)
        print(value.shape)
        cv2.drawContours(board_mask, [board_cnt], 0, (255,255,255), -1)
        # board_mask = cv2.bitwise_and(a_col,a_col,mask = board_mask)
        cv2.imshow("Board Mask",board_mask)
        
        new_img = cv2.bitwise_and(img,img,mask = board_mask)
        cv2.imshow('board mask rgb', new_img)
        

        ##############################################
        
        cv2.imshow("saturation", saturation)
        cv2.imshow("b_col", b_col)
        # cv2.imshow("saturation", saturation)
        RedBlueButPos  = getRedBlueButtons(saturation,b_col,contours_limited)
            
        cv2.drawContours(img, board_cnt, -1, (0,255,0), 3)
        # cv2.imshow(single_image_name,img)
        # cv2.imshow('HSV image', hsv)
        # cv2.imshow('LAB image', lab)    
        # cv2.imshow('BW image', bw)
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        
if __name__ == "__main__":
    main()
    
    
    