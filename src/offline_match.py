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

    images_folder_path = "/home/teamcari/projects/robothon2022_ws/src/robothon2022_vision/file"
    images_folder_path = "/home/samuele/projects/robothon_2022_ws/src/robothon2022_vision/file"
    realsense=RealSense()
    camera_parameters = {"height": 720,"width": 1280,"distortion_model": "plumb_bob", "D": [0.0, 0.0, 0.0, 0.0, 0.0],"K": [902.4752197265625, 0.0, 633.35498046875, 0.0, 901.7176513671875, 379.5818786621094, 0.0, 0.0, 1.0]}

    realsense.setcameraInfo(camera_parameters)
    # realsense.getCameraParam()  #For subscribe camera info usefull for Deprojection
    # realsense.waitCameraInfo()
    k=0
    if True:
        single_image_name = "/frame_3.png"
    # for single_image_name in os.listdir(images_folder_path):
    #     img = cv2.imread(os.path.join(images_folder_path,single_image_name))
        img1 = cv2.imread(images_folder_path+single_image_name,cv2.IMREAD_GRAYSCALE)
        img1 = getRoi(img1,144,669,360,1150)
        img2 = cv2.imread(images_folder_path+"/filefeature0.png",cv2.IMREAD_GRAYSCALE)
        # cv2.imshow('immagine', img)
        # ras=input("Riga alto sinistra")
        # cas=input("Colonna alto sinistra")
        # rbd=input("Riga basso destra")
        # cad=input("Colonna alto destra")
        # img = img[558:619,775:828]
        # k=+1
        # cv2.imwrite(images_folder_path + '/feature'+str(k)+".png", img)
        
        
        #-- Step 1: Detect the keypoints using SURF Detector, compute the descriptors
        minHessian = 400
        detector = cv2.SIFT_create()
        import matplotlib.pyplot as plt
        
        # find the keypoints and descriptors with SIFT
        kp1, des1 = detector.detectAndCompute(img1,None)
        kp2, des2 = detector.detectAndCompute(img2,None)

        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=100)   # or pass empty dictionary
        flann = cv2.FlannBasedMatcher(index_params,search_params)
        matches = flann.knnMatch(des1,des2,k=2)
        # Need to draw only good matches, so create a mask
        matchesMask = [[0,0] for i in range(len(matches))]
        good = []
        # ratio test as per Lowe's paper
        for i,(m,n) in enumerate(matches):
            if m.distance < 0.7*n.distance:
                matchesMask[i]=[1,0]
                good.append(m)
        # draw_params = dict(matchColor = (0,255,0),
        #                 singlePointColor = (255,0,0),
        #                 matchesMask = matchesMask,
        #                 flags = cv2.DrawMatchesFlags_DEFAULT)
        # img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
        
        MIN_MATCH_COUNT=10
        if len(good)>MIN_MATCH_COUNT:
            src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
            matchesMask = mask.ravel().tolist()

            h,w = img1.shape
            pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
            dst = cv2.perspectiveTransform(pts,M)

            img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

        else:
            print("Not enough matches are found -  {}/{}".format(len(good),MIN_MATCH_COUNT)) 
            matchesMask = None

        draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                    singlePointColor = None,
                    matchesMask = matchesMask, # draw only inliers
                    flags = 2)


        img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
        
        plt.imshow(img3, 'gray')
        plt.show()        
        
        
        # # detector = cv2.xfeatures2d_SURF.create(hessianThreshold=minHessian)
        # keypoints1, descriptors1 = detector.detectAndCompute(img1, None)
        # keypoints2, descriptors2 = detector.detectAndCompute(img2, None)
        # #-- Step 2: Matching descriptor vectors with a FLANN based matcher
        # # Since SURF is a floating-point descriptor NORM_L2 is used
        # matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
        # knn_matches = matcher.knnMatch(descriptors1, descriptors2, 2)
        # #-- Filter matches using the Lowe's ratio test
        # ratio_thresh = 0.58
        # good_matches = []
        # for m,n in knn_matches:
        #     if m.distance < ratio_thresh * n.distance:
        #         good_matches.append(m)
        # #-- Draw matches
        # img_matches = np.empty((max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1], 3), dtype=np.uint8)
        # cv2.drawMatches(img1, keypoints1, img2, keypoints2, good_matches, img_matches, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        # #-- Show detected matches
        # cv2.imshow('Good Matches', img_matches)


        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
