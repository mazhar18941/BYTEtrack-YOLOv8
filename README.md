# Introduction
This repo contains code for Object tracking using BYTEtrack and Yolov8. BYTE is a simple tracker which use both high and 
low score detections for tracking.

# Requirements
Python =< 3.9
```
pip install -r requirements.txt
```

# Code
Before running code download yolov8 weights from [here](https://docs.ultralytics.com/models/yolov8/#supported-tasks-and-modes).
```
python app.py --video video/video.mp4 --object_detector yolo/yolov8m.pt
```
