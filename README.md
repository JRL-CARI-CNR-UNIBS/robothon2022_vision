# Board Localization Repository
<p align="center">
  <img width="600" src="https://github.com/JRL-CARI-CNR-UNIBS/robothon2022_report/blob/master/images/Vision_System.png">
</p>
This package is for real time robothon-board localization: the vision system is used to identify the position of the board relative to the robot base. Specifically, an rgb frame is acquired as the first task, then features such as the center of the red button, key lock, and screen are identified. 

The features are recognized by applying border detection, color clustering, canny detection, Hough transform and custom designed vision algorithms. Once the features position is detected in the image frame, we move on to the camera reference system and finally that of the robot base, thanks to the instrinsic and extrinsic parameters estimated in the offline part. 

## Requirements
- **realsense-ros**: you can find the necessary package [here](https://github.com/IntelRealSense/realsense-ros)
