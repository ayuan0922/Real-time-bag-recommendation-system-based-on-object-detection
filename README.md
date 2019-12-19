# Real-time-bag-recommendation-system-based-on-object-detection

## Introduction
This is an application of YOLOv3:real-time object detection. Input a video or turn on the camera to detect a bag, then the system will recommend the user six bags which are similar to the detected one.<br>
![image](https://github.com/ayuan0922/Real-time-bag-recommendation-system-based-on-object-detection/blob/master/%E6%B5%81%E7%A8%8B%E5%9C%96.png)<br>
 ## Requirement
 Python 3.6<br>
 Keras 2.1.5<br>
Tensorflow-gpu 1.4.0<br>

## Usage
1. Download weights from [here.](https://drive.google.com/drive/folders/1lgSpWjWevtSZ_2gob3swpvJw80U3NkQJ?usp=sharing)<br>
2. Run “system.py” to open the GUI.<br>
![image](https://github.com/ayuan0922/Real-time-bag-recommendation-system-based-on-object-detection/blob/master/GUI.JPG)<br>
Press”打開相機” to open the camera, then it will detect bags automatically.<br>
Press”顯示結果”to display the recommended bags.<br>
(After pressing “顯示結果”, pressing” pick me” below the picture can connect to the website where user can buy the bag.)<br>

## Demo result
![image](https://github.com/ayuan0922/Real-time-bag-recommendation-system-based-on-object-detection/blob/master/Demo.gif)<br>

## Reference
[keras-yolov3:https://github.com/qqwweee/keras-yolo3](https://github.com/qqwweee/keras-yolo3)
