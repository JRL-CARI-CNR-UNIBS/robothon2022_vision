#! /usr/bin/env python3

import cv2
import numpy as np
from sklearn.cluster import KMeans
from math import cos,sin
import rospy

GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
BOLD = '\033[1m'
END = '\033[0m'

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

def getScreen(a_col_in,contours_in):
    passed_imgs = 0
    max_idx     = 0
    max_val     = 0
    set_butt    = False
    for idx,cnt in enumerate(contours_in):
        area = cv2.contourArea(cnt)
        print(area)
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
                # cv2.imshow('ROI', ROI)
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
            # cv2.imshow("BOTTONIIIIIII"+str(ind), butt_image_gray)
            contours_test, hierarchy_test = cv2.findContours(butt_image_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            img = new_img[y:y+h, x:x+w]
            cv2.drawContours(img, contours_test, -1, (0, 255, 0), 3)
            # cv2.imshow("CONTORNI Nuovo"+ str(ind),img)

            ret, thresh = cv2.threshold(butt_image_gray, 100, 255, 0)           # era 127
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(new_img, contours_in[butt_idx], -1, (0, 255, 0), 3)
            # cv2.imshow("CONTORNI Nuovo"+ str(ind),new_img)

            img_appoggio = new_img.copy()
            # cv2.imshow("appoggio",img_appoggio)

            # cv2.imshow("BOTTONI"+str(ind),thresh)
            #input("aspetta")
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
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

def getKeyLock(lab_l_in,contours_in,orig,ScreenPos,new_img):
    rospy.loginfo(RED + "Get key lock identification" + END)
    import traceback
    
    print("N contorni: {}".format(len(contours_in)))
    distance_from_screen_acceptable = False

    cv2.drawContours(new_img,contours_in,-1, (0, 255, 0), 3)
    # cv2.imshow("contorni chiaveeeeee",new_img)
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
            # cv2.imshow("prova chiave"+str(id),ROI_key)
        #### DEBUG

        x,y,w,h = cv2.boundingRect(contours_in[id_circle])
        extend_ = 10
        shift   = np.array([x-extend_,y-extend_,0])
        ROI_key = lab_l_in[y-extend_:y+h+extend_, x-extend_:x+w+extend_]

        param2  = 60
        circles = []

        while len(circles) == 0:
            circles = cv2.HoughCircles(ROI_key,cv2.HOUGH_GRADIENT,1.2,15, param1=150,param2=param2,minRadius=10,maxRadius=22)
            if circles is None:
                circles = []
            param2 = param2 - 1

        circles = np.uint16(np.around(circles))

        for i in circles[0,:]:
            cv2.circle(ROI_key,(int(i[0]),int(i[1])),i[2],(255,250,250),1)
            cv2.circle(ROI_key,(int(i[0]),int(i[1])),2,(250,250,250),1)
            # cv2.imshow("chiave",ROI_key)
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

                # cv2.imshow('image', butt_image_gray_copy)
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
            # cv2.imshow('lab_l_in', orig)
            # cv2.imshow('ROI_key', ROI_key)
            # cv2.waitKey(0)

        circle_pos = np.array([circles[0,0] + shift])


        print("Circle pos: {}".format(circle_pos[0][:2]))

        print("Screen pos: {}".format(ScreenPos))
        print(np.linalg.norm(circle_pos[0][:2]-ScreenPos))
        distance_from_screen = np.linalg.norm(circle_pos[0][:2]-ScreenPos)
        
        print("Distanza da screen: {}".format(distance_from_screen))
        if distance_from_screen < 80:
            distance_from_screen_acceptable = True
        if contours_in:
            contours_in.pop(id_circle)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return(np.array([circles[0,0] + shift]),id_circle)
