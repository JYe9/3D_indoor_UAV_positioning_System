Using OpenCV computer vision library.

Calibrate stereo camera, because some cameras can pro duce distortions in the images they capture such as causing straight lines to appear curved, which will introduces error when doing computer vision.

### Calibration

This distortion can be computed and corrected by taking a load of photos with a chess board pattern in the frame, then pass those photos into its calibration API that gives back a matrix defining the camera's distortion which then can be used to correct for it.

### Stereos Depth Mapping

2 algorithms in OpenCV can achieve: **STEREOBM** & **STEREOSGBM**

Both algorithms rely on the input images to be rectified, which is taken the results of the calibration phase applied them and made sure that images line up nicely.

**STEREOBM** is the fastest of the two, which use black matching. It works by taking a block of pixels in the left hand frame scans across the corresponding x-axis in the right-hand frame then find the closest matching block between the two frames , the result of this is a disparity value per block which roughly is the distance in pixels between matching blocks. 

**STEREOSGBM** using semi-global block matching, for each block of pixels in the left-hand frame, can scan across multiple directions in the right-hand frame to find the closest match. This approach is a lot more intensive but tends to produce a much more accurate depth map. 

However, both algorithms don't give exactly the depth map straight away, to get the depth distance, the distance between two cameras to calculate the distance.

Intel REALSENSE D435

Reference: https://www.youtube.com/watch?v=_fH4SPhKpPM&list=WL&index=9&ab_channel=aka%3AMatchstic