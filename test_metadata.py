import json
import cv2
import os
from video import Video
import time
import numpy as np


with open('database/test.json') as f:
    database = json.load(f)

start = time.time()
video_path = "media/test.mp4"
video = Video(video_path)

for item in database:
    indexes = [i for i,val in enumerate(item['analytics']['detection']['objects']['classes']) if val in ["truck"]]
    if len(indexes) > 0:
        frame = video.get_frame_position_by_index(item["frame_position"])
        for index in indexes:
            bbox = item['analytics']['detection']['objects']['bboxes'][index]
            cv2.imwrite(os.path.join('extracted_data', item["name"]+'_'+str(index)+'.jpg'), frame[bbox[1]-int(bbox[3]/2):bbox[1]+int(bbox[3]/2), bbox[0]-int(bbox[2]/2):bbox[0]+int(bbox[2]/2), :])

print('computing time : ' + str(np.round(time.time() - start, 2)) + ' s')
