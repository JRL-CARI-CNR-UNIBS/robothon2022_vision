#! /usr/bin/env python3

from turtle import Screen
import rospy
import cv2
import os
import numpy as np
from sklearn.cluster import KMeans
from math import cos,sin
from RealSense import RealSense
import tf2_ros
import tf
import geometry_msgs.msg
        
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
        cv2.imshow("screen", mask)
        
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

    return np.array([[cX,cY]]), max_idx

def getRedBlueButtons(value_in_gray,b_col_in,contours_in,new_img,ScreenPos):
    rospy.loginfo(RED + "DENTRO RED BLUE" + END)
    distance_from_screen_acceptable = False
    while not distance_from_screen_acceptable:
        passed_imgs = 0
        butt_idx    = 0
        std_max     = 0
        set_butt    = False
        roi_list = []
        print(len(contours_in))
        buttons_found = False
        ind=0
        while not buttons_found:
            len(contours_in)
            ind +=1
            for idx,cnt in enumerate(contours_in):
                print(idx)
                area = cv2.contourArea(cnt)
                print("area")
                print(area)
                if (area > 5000) or (area < 700):       #era 1000
                    continue
                passed_imgs += 1
                x,y,w,h = cv2.boundingRect(cnt)
                ROI = b_col_in[y:y+h, x:x+w]
                cv2.imshow('ROI', ROI)
                # while cv2.waitKey(33) != ord('a'):
                #     rospy.sleep(1)

                flattened = ROI.reshape((ROI.shape[0] * ROI.shape[1], 1))
                #cv2.imshow("flattened red blue button",ROI)
                clt = KMeans(n_clusters = 3)
                clt.fit(flattened)

                # print(np.std(clt.cluster_centers_))
                if np.std(clt.cluster_centers_) > std_max:
                    butt_idx = idx
                    std_max  = np.std(clt.cluster_centers_)
                else:
                    continue
            print("button idx")
            print(butt_idx)
            
            x,y,w,h           = cv2.boundingRect(contours_in[butt_idx])
            butt_image_b      = b_col_in[y:y+h, x:x+w]
            shift             = np.asarray([x,y])
            butt_image_gray   = value_in_gray[y:y+h, x:x+w]
            
            # val_mid           = value_in_gray[int(value_in_gray.shape[1]/2),int(value_in_gray.shape[0]/2)]
            # corn_width        = 3

            circles = []
            cv2.imshow("BOTTONIIIIIII"+str(ind), butt_image_gray)
            contours_test, hierarchy_test = cv2.findContours(butt_image_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
            
            img = new_img[y:y+h, x:x+w]
            cv2.drawContours(img, contours_test, -1, (0, 255, 0), 3)
            cv2.imshow("CONTORNI Nuovo"+ str(ind),img)        
            
            ret, thresh = cv2.threshold(butt_image_gray, 100, 255, 0)           # era 127
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
            # cv2.drawContours(new_img, contours_in[butt_idx], -1, (0, 255, 0), 3)
            # cv2.imshow("CONTORNI Nuovo"+ str(ind),new_img) 
            
            img_appoggio = new_img.copy()
            cv2.imshow("appoggio",img_appoggio)

            cv2.imshow("BOTTONI"+str(ind),thresh)
            # input("aspetta")
            contours = sorted(contours, key=lambda x: cv2.contourArea(x))
            mask = np.zeros(butt_image_gray.shape[:2],np.uint8)
            
            all_contour_area_little = True
            # if len(contours)>30:
            #     print("Reiterate too mutch contours...")
            #     contours_in.pop(butt_idx)
            #     continue   
            for cnt in contours:
                print("contourArea: ")
                print(cv2.contourArea(cnt))
                if cv2.contourArea(cnt) < 150:              # very little circle , se ci saranno casi prendere i 2 più grossi
                    continue
                M = cv2.moments(cnt)
                try:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    circles.append(np.array([cX,cY]))
                    cv2.circle(mask,(cX,cY),2,(255,255,255),1)  
                except:
                    pass
                all_contour_area_little = False
            if all_contour_area_little == False:
                buttons_found = True
                print("Buttons found")
            else:
                print("Reiterate...")
                if contours_in:
                    contours_in.pop(butt_idx)
                    continue            
                else:
                    rospy.loginfo("No buttons found")
                    break
            different_from_two = False
            if len(circles) != 2:
                print("More or less than 2 buttons found!!!!!!",len(circles))
                different_from_two = True
            # cv2.imshow("bot", mask)
            # cv2.imshow('ROI_buttons_cluster', butt_image_b)
            # cv2.imshow('ROI_buttons_v', butt_image_gray)
        print(circles)
        # return butt_idx
        if not different_from_two:
            if butt_image_b[circles[0][1],circles[0][0]] > butt_image_b[circles[1][1],circles[1][0]]:
                center_coordinate = np.array([circles[0] + shift,circles[1] + shift])
                #return(np.array([circles[0] + shift,circles[1] + shift]), butt_idx)
            else:
                center_coordinate = np.array([circles[1] + shift,circles[0] + shift])
                #return(np.array([circles[1] + shift,circles[0] + shift]), butt_idx)
        else:
            center_coordinate = np.array([circles[0] + shift])
                                         
        distance_from_screen = np.linalg.norm(ScreenPos[0]-center_coordinate[0][:2])
        
        print("Screen pos:")
        print(ScreenPos[0])
        print("Center coordinate: ")

        print(center_coordinate[0][:2])
        print(distance_from_screen)
        if distance_from_screen>240:
            distance_from_screen_acceptable = True
            rospy.loginfo(GREEN + "Distante giusto" + END)
        else:
            rospy.loginfo(GREEN + "Troppo poco distante" + END)
            if contours_in:
                contours_in.pop(butt_idx)
            else:
                rospy.loginfo("No buttons found")
                break 
             
    
    
    return center_coordinate, butt_idx
    # rospy.loginfo(RED + "Red blue button identification" + END)
    # passed_imgs = 0
    # butt_idx    = 0
    # std_max     = 0
    # set_butt    = False
    # roi_list = []
    # print(len(contours_in))
    
    # cv2.drawContours(new_img, contours_in, -1, (0, 255, 0), 3)
    # cv2.imshow("PRIMA DI RED",new_img) 
    
    # ind=0
    # buttons_found = False
    # distance_acceptable = False
    # while not distance_acceptable:   
    #     while not buttons_found:        # not little
    #         for idx,cnt in enumerate(contours_in):
    #             print(idx)
    #             area = cv2.contourArea(cnt)
    #             print("area")
    #             print(area)
    #             if (area > 5000) or (area < 700):       #era 1000
    #                 continue
    #             passed_imgs += 1
    #             x,y,w,h = cv2.boundingRect(cnt)
    #             ROI = b_col_in[y:y+h, x:x+w]
    #             cv2.imshow('ROI'+ str(idx), ROI)
    #             # while cv2.waitKey(33) != ord('a'):
    #             #     rospy.sleep(1)

    #             flattened = ROI.reshape((ROI.shape[0] * ROI.shape[1], 1))
    #             #cv2.imshow("flattened red blue button",ROI)
    #             clt = KMeans(n_clusters = 3)
    #             clt.fit(flattened)

    #             print(np.std(clt.cluster_centers_))
    #             if np.std(clt.cluster_centers_) > std_max:
    #                 butt_idx = idx
    #                 std_max  = np.std(clt.cluster_centers_)
    #             # else:
    #             #     continue
            
    #         rospy.loginfo(RED+"id bottoneeeee" + END)
    #         print("button idx")
    #         print(butt_idx)
    #         len(contours_in)
    #         std_max     = 0
    #         ind +=1
            
    #         x,y,w,h           = cv2.boundingRect(contours_in[butt_idx])
    #         butt_image_b      = b_col_in[y:y+h, x:x+w]
    #         shift             = np.asarray([x,y])
    #         butt_image_gray   = value_in_gray[y:y+h, x:x+w]
            
    #         # val_mid           = value_in_gray[int(value_in_gray.shape[1]/2),int(value_in_gray.shape[0]/2)]
    #         # corn_width        = 3

    #         circles = []
    #         cv2.imshow("BOTTONIIIIIII"+str(ind), butt_image_gray)
    #         contours_test, hierarchy_test = cv2.findContours(butt_image_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
            
    #         img = new_img[y:y+h, x:x+w]
    #         cv2.drawContours(img, contours_test, -1, (0, 255, 0), 3)
    #         cv2.imshow("CONTORNI Nuovo"+ str(ind),img)        
            
    #         ret, thresh = cv2.threshold(butt_image_gray, 100, 255, 0)           # era 127
    #         contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
    #         cv2.drawContours(new_img, contours_in[butt_idx], -1, (0, 255, 0), 3)
    #         cv2.imshow("CONTORNI Nuovo"+ str(ind),new_img) 
            
    #         img_appoggio = new_img.copy()
    #         cv2.imshow("appoggio",img_appoggio)

    #         cv2.imshow("BOTTONI"+str(ind),thresh)
    #         # input("aspetta")
    #         contours = sorted(contours, key=lambda x: cv2.contourArea(x))
    #         mask = np.zeros(butt_image_gray.shape[:2],np.uint8)
            
    #         all_contour_area_little = True
    #         # if len(contours)>30:
    #         #     print("Reiterate too mutch contours...")
    #         #     contours_in.pop(butt_idx)
    #         #     continue   
    #         for cnt in contours:
    #             print("contourArea: ")
    #             print(cv2.contourArea(cnt))
    #             if cv2.contourArea(cnt) < 150:              # very little circle , se ci saranno casi prendere i 2 più grossi
    #                 continue
    #             M = cv2.moments(cnt)
    #             try:
    #                 cX = int(M["m10"] / M["m00"])
    #                 cY = int(M["m01"] / M["m00"])
    #                 circles.append(np.array([cX,cY]))
    #                 cv2.circle(mask,(cX,cY),2,(255,255,255),1)  
    #             except:
    #                 pass
    #             all_contour_area_little = False
    #         if all_contour_area_little == False:
    #             buttons_found = True
    #             print("Buttons found")
    #         else:
    #             print("Reiterate...")
    #             if contours_in:
    #                 contours_in.pop(butt_idx)
    #             continue            
            
    #         if len(circles) != 2:
    #             print("More or less than 2 buttons found!!!!!!",len(circles))
    #         # cv2.imshow("bot", mask)
    #         # cv2.imshow('ROI_buttons_cluster', butt_image_b)
    #         # cv2.imshow('ROI_buttons_v', butt_image_gray)

    #     # return butt_idx

    #     if butt_image_b[circles[0][1],circles[0][0]] > butt_image_b[circles[1][1],circles[1][0]]:
    #         center_coordinate = np.array([circles[0] + shift,circles[1] + shift])
    #         #return(np.array([circles[0] + shift,circles[1] + shift]), butt_idx)
    #     else:
    #         center_coordinate = np.array([circles[1] + shift,circles[0] + shift])
    #         #return(np.array([circles[1] + shift,circles[0] + shift]), butt_idx)

    #     print("Screen pos:")
    #     print(ScreenPos[0])
    #     print("Center coordinate: ")
    #     print(center_coordinate)
    #     print(center_coordinate[0][:2])
    #     distance_from_screen = np.linalg.norm(ScreenPos[0]-center_coordinate[0][:2])
    #     print
    #     print(center_coordinate)
    #     if distance_from_screen>240:
    #         distance_acceptable = True
    #     rospy.loginfo(RED + "Troppo poco distante" + END)
        
    #     print(distance_from_screen)
    #     # print(contours_in)
    #     #contours_in.pop(butt_idx)
    # return center_coordinate, butt_idx

def getButtonCell(lab_l, contours_in, crop_img, ScreenPos, KeyLockPos,new_img):
    rospy.loginfo(RED + "Get button cell identification" + END)
    distance_from_screen_acceptable = False
    print(len(contours_in))
    cv2.drawContours(new_img,contours_in,-1, (0, 255, 0), 3)
    cv2.imshow("contorni button cell",new_img)
    
    attempt_max = len(contours_in)
    attempt = 0
    attempt_max_reached = False
    while not distance_from_screen_acceptable:    
        ecc_list = []
        for idx,cnt in enumerate(contours_in):
            hull = cv2.convexHull(cnt)
            area = cv2.contourArea(hull)
            rospy.loginfo(RED + "--------AREA-------" + END)
            print(area)
            peri = cv2.arcLength(hull,True)
            ecc  = (4 * np.pi * area)/(peri*peri)
            ecc_list.append(ecc)
        print(ecc_list)    
        id_circle = ecc_list.index(max(ecc_list))
        
        #### inizio debug
        for id,circle in enumerate(ecc_list):
            x,y,w,h = cv2.boundingRect(contours_in[id])
            ROI_key = lab_l[y-10:y+h+10, x-10:x+w+10]
            cv2.imshow("prova button cell"+str(id),ROI_key)    
        #### fine debug
        x,y,w,h = cv2.boundingRect(contours_in[id_circle])
        extend_ = 10
        shift   = np.array([x-extend_,y-extend_,0])
        print("shift****************")
        print(shift)
        ROI_key = lab_l[y-extend_:y+h+extend_, x-extend_:x+w+extend_]
        
        param2  = 60
        circles = []

        while len(circles) == 0:
            circles = cv2.HoughCircles(ROI_key,cv2.HOUGH_GRADIENT,1.2,15, param1=150,param2=param2,minRadius=10,maxRadius=22)
            if circles is None:
                circles = []
            param2 = param2 - 1

        circles = np.uint16(np.around(circles))
        print("circles*******************************")
        print(circles[0,:])
        
        for i in circles[0,:]:
            cv2.circle(ROI_key,(int(i[0]),int(i[1])),i[2],(255,250,250),1)
            cv2.circle(ROI_key,(int(i[0]),int(i[1])),2,(250,250,250),1)
            cv2.imshow("pila bottone",ROI_key)
        import traceback    
        if 0:
            cv2.namedWindow('image')
            def nothing(x):
                pass
            cv2.createTrackbar('Param 1','image',150,200,nothing)
            cv2.createTrackbar('Param 2','image',80,100,nothing)

            while cv2.waitKey(33) != ord('a'):
                butt_image_gray_copy=ROI_key.copy()
                cv2.imshow('image',butt_image_gray_copy)
                #To Get Parameter values from Trackbar Values
                para1 = cv2.getTrackbarPos('Param 1','image')
                para2 = cv2.getTrackbarPos('Param 2','image')

                
                try:
                    print(para1,para2)
                    circles = cv2.HoughCircles(butt_image_gray_copy,cv2.HOUGH_GRADIENT,1.2,15,param1=para1,param2=para2,minRadius=10,maxRadius=25)
                    circles = np.uint16(np.around(circles))
                    print(circles)
                    for i in circles[0,:]:
                        cv2.circle(butt_image_gray_copy,(int(i[0]),int(i[1])),i[2],(255,250,250),1)
                        cv2.circle(butt_image_gray_copy,(int(i[0]),int(i[1])),2,(250,250,250),1)
                except:
                    traceback.print_exc()
                    pass
                #For drawing Hough Circles
                
                cv2.imshow('image', butt_image_gray_copy)
        
        
        circles = np.uint16(np.around(circles))
        print("circles*******************************")
        print(circles[0,:])
        
        button_cell_pos = np.array([circles[0,0] + shift])
    
        print("Button cell pos: ")
        print(button_cell_pos[0][:2])
        print("Screen pos: ")
        print(ScreenPos)
        print("Distanza normale *************************")
        
        distance = np.linalg.norm(button_cell_pos[0][:2]-ScreenPos)
        print(distance)
        attempt+=1
        if(distance > 250):
            distance_from_screen_acceptable = True
            # if attempt>=attempt_max:
            #     print("Button cell not identified")
        if(attempt>=attempt_max):
            attempt_max_reached = True
            print("Button cell not identified")
            break      
        contours_in.pop(id_circle)    
    
    if not attempt_max_reached:
        rospy.loginfo(RED + "ID button cell" + END)
        print(id_circle)
        return(np.array([circles[0,0] + shift]),id_circle)                                              ####Da verificare
    else:
        raise Exception            
        
        
def getKeyLock(lab_l_in,contours_in,orig,ScreenPos,new_img):
    rospy.loginfo(RED + "Get key lock identification" + END)
    import traceback
    print("N contorni:")
    print(len(contours_in))
    
    distance_from_screen_acceptable = False

    cv2.drawContours(new_img,contours_in,-1, (0, 255, 0), 3)
    cv2.imshow("contorni chiaveeeeee",new_img)
    while not distance_from_screen_acceptable:    
        ecc_list = []
        for idx,cnt in enumerate(contours_in):
            hull = cv2.convexHull(cnt)
            area = cv2.contourArea(hull)
            peri = cv2.arcLength(hull,True)
            ecc  = (4 * np.pi * area)/(peri*peri)
            ecc_list.append(ecc)
            # print(area,peri,ecc)
        print(ecc_list)
        id_circle = ecc_list.index(max(ecc_list))
        
        #### DEBUG
        for id,circle in enumerate(ecc_list):
            x,y,w,h = cv2.boundingRect(contours_in[id])
            ROI_key = lab_l_in[y-10:y+h+10, x-10:x+w+10]
            cv2.imshow("prova chiave"+str(id),ROI_key)
        #### DEBUG
                
        x,y,w,h = cv2.boundingRect(contours_in[id_circle])
        extend_ = 10
        shift   = np.array([x-extend_,y-extend_,0])
        print("shift****************")
        print(shift)
        ROI_key = lab_l_in[y-extend_:y+h+extend_, x-extend_:x+w+extend_]
        
        param2  = 60
        circles = []

        while len(circles) == 0:
            circles = cv2.HoughCircles(ROI_key,cv2.HOUGH_GRADIENT,1.2,15, param1=150,param2=param2,minRadius=10,maxRadius=22)
            if circles is None:
                circles = []
            param2 = param2 - 1

        circles = np.uint16(np.around(circles))
        print("circles*******************************")
        print(circles[0,:])
        
        for i in circles[0,:]:
            cv2.circle(ROI_key,(int(i[0]),int(i[1])),i[2],(255,250,250),1)
            cv2.circle(ROI_key,(int(i[0]),int(i[1])),2,(250,250,250),1)
            cv2.imshow("chiave",ROI_key)
        if 0:
            cv2.namedWindow('image')
            def nothing(x):
                pass
            cv2.createTrackbar('Param 1','image',150,200,nothing)
            cv2.createTrackbar('Param 2','image',80,100,nothing)

            while cv2.waitKey(33) != ord('a'):
                butt_image_gray_copy=ROI_key.copy()
                cv2.imshow('image',butt_image_gray_copy)
                #To Get Parameter values from Trackbar Values
                para1 = cv2.getTrackbarPos('Param 1','image')
                para2 = cv2.getTrackbarPos('Param 2','image')

                
                try:
                    print(para1,para2)
                    circles = cv2.HoughCircles(butt_image_gray_copy,cv2.HOUGH_GRADIENT,1.2,15,param1=para1,param2=para2,minRadius=10,maxRadius=25)
                    circles = np.uint16(np.around(circles))
                    print(circles)
                    for i in circles[0,:]:
                        cv2.circle(butt_image_gray_copy,(int(i[0]),int(i[1])),i[2],(255,250,250),1)
                        cv2.circle(butt_image_gray_copy,(int(i[0]),int(i[1])),2,(250,250,250),1)
                except:
                    traceback.print_exc()
                    pass
                #For drawing Hough Circles
                
                cv2.imshow('image', butt_image_gray_copy)
                # cv2.waitKey(5)

        if len(circles[0]) != 1 :
                print("ERROR in KEYHOLE DETECTION: len",len(circles[0]))
        DEBUG_KEYHOLE= False
        if DEBUG_KEYHOLE:
            # print(butt_image_gray.shape)
            # print("circels:",circles)
            for i in circles[0,:]:
                cv2.circle(ROI_key,(int(i[0]),int(i[1])),i[2],(255,255,255),1)
                cv2.circle(ROI_key,(int(i[0]),int(i[1])),2,(255,255,255),1)   
        
                # cv2.drawContours(orig, [contours_in[id_circle]], 0, (rng.uniform(0,255), rng.uniform(0, 255), rng.uniform(0, 255)), 2)
            cv2.imshow('lab_l_in', orig)
            cv2.imshow('ROI_key', ROI_key)
            # cv2.waitKey(0)

        circle_pos = np.array([circles[0,0] + shift])
    
    
        print("Circle pos: ")
        print(circle_pos[0][:2])
        print("Screen pos: ")
        print(ScreenPos)
        print("Distanza normale *************************")
        print(np.linalg.norm(circle_pos[0][:2]-ScreenPos))
        distance_from_screen = np.linalg.norm(circle_pos[0][:2]-ScreenPos)
        if distance_from_screen < 80:
            distance_from_screen_acceptable = True
        contours_in.pop(id_circle)    
    return(np.array([circles[0,0] + shift]),id_circle)

def getRoi(image, min_row, max_row, min_col,max_col):
    mask = np.zeros(image.shape[:2],np.uint8)
    mask[min_row:max_row, min_col:max_col] = 255
    new_img = cv2.bitwise_and(image,image,mask = mask)
    return new_img

def getAllColorSpaces(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    bw  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)    
    return hsv, lab, bw
    
def getBoardContour(image):
    contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = np.asarray(contours)
    #TODO controlla che estrai veramente la board
    contours = sorted(contours, key=lambda x: cv2.contourArea(x))
    contours_limited = contours[-8:]
    board_cnt = contours[-2]
    
    return board_cnt, contours_limited, contours
       
def main():

    rospy.init_node("offline_board_localization")

    # Retrieve nedded rosparam     
    # try:
    #     images_folder_path=rospy.get_param("~images_path")   
    # except KeyError:   
    #     rospy.logerr(RED + PARAM_NOT_DEFINED_ERROR.format("images_path") + END)
    #     return 0
    
    images_folder_path = "/home/samuele/projects/robothon_2022_ws/src/robothon2022_vision/file"
    realsense=RealSense()
    realsense.getCameraParam()  #For subscribe camera info usefull for Deprojection
    realsense.waitCameraInfo()
    # if True:
    #     single_image_name = "/frame_16.png"
    for single_image_name in os.listdir(images_folder_path):
        img = cv2.imread(os.path.join(images_folder_path,single_image_name))      
        # img = cv2.imread(images_folder_path+single_image_name)      
        # 

        crop_img      = img[144:669, 360:1150]
        crop_img_copy = img[144:669, 360:1150].copy()
        # from matplotlib import pyplot as plt
        # import numpy as np
        # plt.hist(crop_img_copy.ravel(),256,[0,256]); 
        # plt.show()
        print(img.shape)
        # cv2.imshow("img croppata",crop_img)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
        bw  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        hue,saturation,value = cv2.split(hsv)
        l_col,a_col,b_col    = cv2.split(lab)

        cv2.imshow("hue", hue)
        cv2.imshow("sat", saturation)
        cv2.imshow("value", value)
        cv2.imshow("l_col", l_col)
        cv2.imshow("a_col", a_col)
        cv2.imshow("b_col", b_col)    
            
        # Threshold and contours 
        ret,value_th = cv2.threshold(value,100,255,0)
        cv2.imshow('VALUE TH', value_th)

        board_cnt, contours_limited, contours = getBoardContour(value_th)
        
        
        print("board contour")
        #print(board_cnt)
        
        x,y,w,h = cv2.boundingRect(board_cnt)
        ROI_board = value[y:y+h, x:x+w]
 
        ################# MACRO ROI ##################
        new_img = getRoi(img,144,669,360,1150)
        cv2.imshow('board mask rgb NEWWWW', new_img)
        
        hsv, lab, bw = getAllColorSpaces(new_img)
        
        hue,saturation,value = cv2.split(hsv)
        ret,value_th = cv2.threshold(value,90,255,0)

        board_cnt, contours_limited, contours = getBoardContour(value_th)
        
        #Estimate board angle (test)
        (x,y),(MA,ma),angle = cv2.fitEllipse(board_cnt)
        rospy.loginfo(RED + "Board angle" + END)
        print(angle)
        cv2.drawContours(new_img, contours, -1, (0, 255, 0), 3)
        # cv2.imshow("CONTORNI",new_img)
        
        l_col,a_col,b_col    = cv2.split(lab)
        #################################################
        ScreenPos, idx_screen = getScreen(a_col,contours_limited)
        new_img = getRoi(img,144,669,360,1150)
        # new_img = getRoi(img,144,669,360,1150)
        
        # epsilon = 0.06 * cv2.arcLength(contours_limited[idx_screen], True)
        # approx = cv2.approxPolyDP(contours_limited[idx_screen], epsilon, True)
        # cv2.drawContours(new_img, approx, -1, (0, 255, 0), 3)
        cv2.drawContours(new_img, contours_limited[idx_screen], -1, (0, 255, 0), 3)
        cv2.imshow("SCREEEEN", new_img)
        contours_limited.pop(idx_screen)

        
        print("screeen poss****************************")
        print(ScreenPos)
        new_img = getRoi(img,144,669,360,1150)
        print(ScreenPos)
        
        ################
        new_img = getRoi(img,144,669,360,1150)          # TEMP FOR DEBUG TO SEE NEW CONTOUR
        RedBlueButPos, id_red_blue_contour  = getRedBlueButtons(saturation,b_col,contours_limited,new_img, ScreenPos)
        new_img = cv2.circle(new_img, (ScreenPos[0][0],ScreenPos[0][1]), 5, color = (255, 0, 0), thickness = 2)
        new_img = cv2.circle(new_img, (RedBlueButPos[0][0],RedBlueButPos[0][1]), 5, color = (255, 0, 0), thickness = 2)
        
        from math import cos, sin, pi
        starting_point = (RedBlueButPos[0][0],RedBlueButPos[0][1])
        
        # if angle>90:
        #     angle = angle - 90
        end_point = (round(RedBlueButPos[0][0] - 200* cos(angle*pi/180)),round(RedBlueButPos[0][1] - 200* sin(angle*pi/180)))
        print(starting_point)
        print(end_point)
        cv2.line(new_img, starting_point, end_point, color = (0, 0, 255), thickness = 3)
        
        new_img = getRoi(img,144,669,360,1150)
        contours_limited.pop(id_red_blue_contour)
        KeyLockPos , id_circle    = getKeyLock(l_col,contours_limited,crop_img,ScreenPos,new_img)
        
        cv2.line(new_img, (RedBlueButPos[0][0],RedBlueButPos[0][1]), (KeyLockPos[0][0],KeyLockPos[0][1]), color = (0, 255, 0), thickness = 3)
        # cv2.imshow("all id", new_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
        # Deprojection : Image frame -> Camera frame (camera_color_optical_frame)

        print(RedBlueButPos[0][0])
        print(RedBlueButPos[0][1])
        print(KeyLockPos[0][0])
        print(KeyLockPos[0][1])
        
        # deprojection_red_button = realsense.deproject(RedBlueButPos[0][0],RedBlueButPos[0][1],depth)
        
        # deprojection_key_lock = realsense.deproject(KeyLockPos[0][0],KeyLockPos[0][1],depth)
        
        # print(deprojection_red_button)
        # print(deprojection_key_lock)
        initial_depth = 550 # mm
        real_distance = 150 # mm
        tollerance = 0.2    # mm

        depth = 589         #Estimated
        deprojection_red_button = np.array(realsense.deproject(RedBlueButPos[0][0],RedBlueButPos[0][1],depth))
        deprojection_key_lock = np.array(realsense.deproject(KeyLockPos[0][0],KeyLockPos[0][1],depth))
        deprojection_screen = np.array(realsense.deproject(ScreenPos[0][0],ScreenPos[0][1],depth))
        
        x_axis = ( deprojection_key_lock - deprojection_red_button ) / np.linalg.norm( deprojection_key_lock - deprojection_red_button )
        
        y_axis_first_approach = ( deprojection_screen - deprojection_red_button )
        y_axis_norm = y_axis_first_approach - np.dot(y_axis_first_approach,x_axis)/(np.dot(x_axis,x_axis))*x_axis
        y_axis_norm = y_axis_norm / np.linalg.norm(y_axis_norm)
        
        z_axis = np.cross(x_axis,y_axis_norm)
        
        matR_camera_board=np.array([x_axis,y_axis_norm,z_axis]).T
        
        rotation_quat = tf.transformations.quaternion_from_matrix(matR_camera_board)
        
        
        # Estimation depth
        # computed_distance = np.inf
        # depth = initial_depth
        # while np.abs(computed_distance - real_distance) > tollerance:
        #     print(depth)
        #     deprojection_red_button = np.array(realsense.deproject(RedBlueButPos[0][0],RedBlueButPos[0][1],depth))
            
        #     deprojection_key_lock = np.array(realsense.deproject(KeyLockPos[0][0],KeyLockPos[0][1],depth))
        #     computed_distance = np.linalg.norm(deprojection_red_button[:2]- deprojection_key_lock[:2])
        #     print("computed distance")
        #     print(computed_distance)
        #     depth += 1
        #     # print("deprojection")
        #     # print(deprojection_red_button)
        
        # print(depth)
        # print(computed_distance)


        broadcaster = tf2_ros.StaticTransformBroadcaster()
        static_transformStamped = geometry_msgs.msg.TransformStamped()
        static_transformStamped.header.stamp = rospy.Time.now()
        static_transformStamped.header.frame_id = "camera_color_optical_frame"
        static_transformStamped.child_frame_id = "board"
        
        static_transformStamped.transform.translation.x = deprojection_red_button[0]/1000.0
        static_transformStamped.transform.translation.y = deprojection_red_button[1]/1000.0
        static_transformStamped.transform.translation.z = 589/1000.0
        # quat = tf.transformations.quaternion_from_euler(0.0,0.0,0.0)
        static_transformStamped.transform.rotation.x = rotation_quat[0]
        static_transformStamped.transform.rotation.y = rotation_quat[1]
        static_transformStamped.transform.rotation.z = rotation_quat[2]
        static_transformStamped.transform.rotation.w = rotation_quat[3]

        broadcaster.sendTransform(static_transformStamped)
        rospy.spin()
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        #return 0
        continue
        ################
        rospy.loginfo(GREEN + "Key lock pos" + END)
        print(KeyLockPos)
        #contours_limited.pop(id_circle)
        new_img = getRoi(img,144,669,360,1150)
        
        button_cell_identified = False
        try:
            ButtonCellPos , id_button_cell = getButtonCell(l_col, contours_limited,crop_img,ScreenPos,KeyLockPos,new_img)
            button_cell_identified = True
        except Exception:
            print("NO BUTTON CELL IDENTIFIED")    
        
        new_img = getRoi(img,144,669,360,1150)
        print(ScreenPos)
        new_img = cv2.circle(new_img, (ScreenPos[0][0],ScreenPos[0][1]), 5, color = (255, 0, 0), thickness = 2)
        new_img = cv2.circle(new_img, (RedBlueButPos[0][0],RedBlueButPos[0][1]), 5, color = (255, 0, 0), thickness = 2)
        new_img = cv2.circle(new_img, (KeyLockPos[0][0],KeyLockPos[0][1]), 5, color = (255, 0, 0), thickness = 2)
        if button_cell_identified:
            new_img = cv2.circle(new_img, (ButtonCellPos[0][0],ButtonCellPos[0][1]), 5, color = (255, 0, 0), thickness = 2)
            
        # Displaying the image
        cv2.imshow("Identificazione completata", new_img)
        #contours_limited.pop(id_button_cell)
        
        ################# APPROX BOARD ##################
        # new_img = new_img = getRoi(img,y,y+h,x,x+w)
        # cv2.imshow('board mask rgb NEWWWW', new_img)
        # hsv = cv2.cvtColor(new_img, cv2.COLOR_BGR2HSV)
        # hue,saturation,value = cv2.split(hsv)
        # ret,value_th = cv2.threshold(value,90,255,0)

        # contours_new, hierarchy = cv2.findContours(value_th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # cv2.drawContours(new_img, contours_new, -1, (0, 255, 0), 3)
        # cv2.imshow("CONTORNI",new_img)
        #################################################
        #non fatto il get opt threshold
        
        # #Get Screen position
        # cv2.imshow("A COL PER GET SCREEN", a_col)
        # ScreenPos = getScreen(a_col,contours_limited)

        ################# BOARD MASK ON REAL IMAGE ##################
        # board_mask = np.zeros(value.shape[:2],np.uint8)
        # print(value.shape)
        # cv2.drawContours(board_mask, [board_cnt], 0, (255,255,255), -1)
        # # board_mask = cv2.bitwise_and(a_col,a_col,mask = board_mask)
        # cv2.imshow("Board Mask",board_mask)
        
        # new_img = cv2.bitwise_and(img,img,mask = board_mask)
        # cv2.imshow('board mask rgb', new_img)
        

        # ##############################################
        
        # cv2.imshow("saturation", saturation)
        # cv2.imshow("b_col", b_col)
        # # cv2.imshow("saturation", saturation)
        # RedBlueButPos  = getRedBlueButtons(saturation,b_col,contours_limited)
            
        # cv2.drawContours(img, board_cnt, -1, (0,255,0), 3)
        # # cv2.imshow(single_image_name,img)
        # # cv2.imshow('HSV image', hsv)
        # # cv2.imshow('LAB image', lab)    
        # # cv2.imshow('BW image', bw)
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        
if __name__ == "__main__":
    main()
    
    
    