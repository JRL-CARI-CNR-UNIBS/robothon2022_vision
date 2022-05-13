#! /usr/bin/env python3

import rospy
from std_srvs.srv import SetBool,SetBoolResponse
from RealSense import RealSense

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

    rospy.init_node("save_images_node")

    # Retrieve nedded rosparam     
    try:
        images_path=rospy.get_param("~images_path")   
    except KeyError:   
        rospy.logerr(RED + PARAM_NOT_DEFINED_ERROR.format("images_path") + END)
        return 0
    
    realsense = RealSense()
    
    another_image = True
    while another_image:                                 # until user want another position
        realsense.saveImage(images_path)
        another_image_question = input(USER_QUESTION)                    # ask another position 
        if another_image_question == "n":
            another_image = False


    
    
if __name__ == "__main__":
    main()
    
    
    