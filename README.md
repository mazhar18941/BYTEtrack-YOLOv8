# Introduction
This repo contains code for Object tracking using BYTEtrack and Yolov8. BYTE is a simple tracker which use both high and 
low score detections for tracking.

# Requirements
Python =< 3.9
```
git clone https://github.com/mazhar18941/BYTEtrack-YOLOv8.git
```
```
pip install -r requirements.txt
```

# Code
Before running code download yolov8 weights from [here](https://docs.ultralytics.com/models/yolov8/#supported-tasks-and-modes).
```
python app.py --video video/video.mp4 --object_detector yolo/yolov8m.pt
```
Code only detects and track 'car' class. If you want to track another calss or number of classes, edit line 41 in app.py.

```
if result.names[box.cls[0].item()] == 'car':
```
