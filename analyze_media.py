import cv2
from ultralytics import YOLO
import numpy as np
import copy
import torch
from video import Video
import time
import json
import os

# Load the YOLOv8 model
model = YOLO('weights/yolov8n.pt')

# Open the video file
video_path = "media/test.mp4"
video = Video(video_path)

metadata = {
    'path':  video_path,
    'name': '',
    'width_px': 0,
    'height_px': 0,     
    'frame_position': 0,
    'frame_timestamp_ms': 0,
    
    "analytics": {
        "detection": {
            "objects": {
                "classes": [],
                "confidences": [],
                "bboxes": [],
            },
            "faces": {
                "names": [],
                "ages": [],
                "genders": [],
                "embeddings": [],
                "landmarks": [],
                "confidences": [],
                "bboxes": [],
            },
        },
        "classification": {},
        "segmentation": {},
        "captioning": {},
    }
}


DISPLAY = False
output = []

start = time.time()
# Loop through the video frames
while video.get_current_frame_position() < video.frame_number:
    # Read a frame from the video
    frame = video.get_frame()

    if frame is not None:
        # Run YOLOv8 inference on the frame
        results = model(frame, imgsz=640, conf=0.5, iou=0.7, half=False, device='cpu')

        # Process results
        classes = [results[0].names[int(i)] for i in results[0].boxes.cls]
        confidences = [np.round(i, decimals=2) for i in results[0].boxes.conf.tolist()]
        bboxes = results[0].boxes.xywh.type(torch.int).tolist()

        # Fill metadata with detections
        current_metadata = copy.deepcopy(metadata)
        current_metadata['name'] = current_metadata['path'].split('/')[-1].split('.')[0] + '_' + str(video.get_current_frame_position())
        current_metadata['frame_position'] = video.get_current_frame_position()
        current_metadata['frame_timestamp_ms'] = video.get_current_frame_timestamp()
        current_metadata['analytics']['detection']['objects']['bboxes'] = bboxes
        current_metadata['analytics']['detection']['objects']['classes'] = classes
        current_metadata['analytics']['detection']['objects']['confidences'] = confidences

        output.append(current_metadata)

        if DISPLAY:
            # Visualize the results on the frame
            annotated_frame = results[0].plot()
            # Display the annotated frame
            cv2.imshow("YOLOv8 Inference", annotated_frame)
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    else:
        # Break the loop if the end of the video is reached
        break

print('computing time : ' + str(np.round(time.time() - start, 2)) + ' s')

with open(os.path.join('database', video.name + '.json'), 'w') as f:
    json.dump(output, f)

