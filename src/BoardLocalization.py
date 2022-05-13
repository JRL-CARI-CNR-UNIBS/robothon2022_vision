#! /usr/bin/env python3

import rospy
from std_srvs.srv import SetBool,SetBoolResponse

GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
BOLD = '\033[1m'
END = '\033[0m'


SERVICE_CALLBACK = GREEN + "Service call {} received" + END
PARAM_NOT_DEFINED_ERROR = "Parameter: {} not defined"
SUCCESSFUL = "Successfully executed"
NOT_SUCCESSFUL = "Not Successfully executed"

SERVICE_NAME = "/board_localization/..."

class BoardLocalization:
    
    def __init__(self):

        # Rosservice
        rospy.Service(SERVICE_NAME,SetBool,self.callbak)
    
    def callback(self,request):

        rospy.loginfo(SERVICE_CALLBACK.format(SERVICE_NAME))
        
        try:
            return SetBoolResponse(True,SUCCESSFUL)
        except :
            return SetBoolResponse(False,NOT_SUCCESSFUL)
        
def main():

    rospy.init_node("mongo_statistics")

    # Retrieve nedded rosparam     
    # try:
    #     param_name=rospy.get_param("~rosparam_name")   
    # except KeyError:   
    #     rospy.logerr(RED + PARAM_NOT_DEFINED_ERROR.format("mongo_database") + END)
    #     return 0

    
    try:
        board_localization = BoardLocalization()
    except:
        return 0     #Connection to db failed 
    
       
    
    rospy.spin()

if __name__ == "__main__":
    main()