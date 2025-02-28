# basketball-video-tracking
Simple tracking for video input

#How to use
Execute main.py (to use another video change video_path in main.py)
Select a person to track in pop up window
After calculating bounding boxes for each frame a code will generate a video tracked_video with the selected person marked

#Libraries used
CV2, YOLO

#Used method info
Main models/methods used: YOLOv8 for person detection, TrackerCSRT for tracking (during tracking YOLO periodically recalculates player poxsitions)
Also deepsort was considered, but TrackerCSRT prooved to be easier to use for one object tracking
