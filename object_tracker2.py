import time
import numpy as np
import cv2
import os
import datetime
import matplotlib.pyplot as plt

import tensorflow as tf
from yolov3_tf2.models import YoloV3
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils import convert_boxes

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from utils.load_config import load_config
from utils.counting import count

from _collections import deque
from shapely.geometry import Point, Polygon


MAX_COSINE_DISTANCE = 0.5
NN_BUDGET = None
NMS_MAX_OVERLAP = 0.8
    
    
def detect_with_YOLOv3(img):
    img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_in = tf.expand_dims(img_in, 0)
    img_in = transform_images(img_in, 416)

    t1 = time.time()

    boxes, scores, classes, nums = yolo.predict(img_in)
    boxes = boxes[0]
    scores = scores[0]
    classes = classes[0]
    return boxes, scores, classes

def load_detection_output(detection_path, video_path, frame_id):
    frame_base_name = os.path.basename(video_path)
    frame_base_name = os.path.splitext(frame_base_name)[0] + "_{:05d}.txt"
    frame_name = frame_base_name.format(frame_id)
    frame_path = os.path.join(detection_path, frame_name)
    
    boxes = []
    scores = []
    classes = []

    with open(frame_path, 'r') as fi:
        lines = fi.read().splitlines()
        for line in lines:
            content = line.split()
            classes.append(int(content[0]))
            boxes.append([float(x) for x in content[1:5]])
            scores.append(float(content[5]))
            
    boxes = np.array(boxes)
    scores = np.array(scores)
    classes = np.array(classes)

    return boxes, scores, classes

# Leave OUTPUT_PATH=None if you don't want to output visualize video

def tracking(VIDEO_PATH, OUTPUT_PATH, DETECTION_PATH, config):
    
    class_names = [c.strip() for c in open('./data/labels/obj.names').readlines()]
    # yolo = YoloV3(classes=len(class_names))
    # yolo.load_weights('./weights/yolov3.tf')

    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric('cosine', MAX_COSINE_DISTANCE, NN_BUDGET)
    tracker = Tracker(metric)

    vid = cv2.VideoCapture(VIDEO_PATH)
    vid_fps = vid.get(cv2.CAP_PROP_FPS)
    codec = cv2.VideoWriter_fourcc(*'mp4v')
    vid_fps =int(vid.get(cv2.CAP_PROP_FPS))
    vid_width,vid_height = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if OUTPUT_PATH is not None:
        out = cv2.VideoWriter(OUTPUT_PATH, codec, vid_fps, (vid_width, vid_height))

    pts = [deque(maxlen=2000) for _ in range(500000)]
    track_history = [deque(maxlen=2000) for _ in range(500000)]

    frame_count = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_id = 0

    cmap = plt.get_cmap('tab20b')
    colors = [cmap(i)[:3] for i in np.linspace(0,1,20)]
    
    while True:
        _, img = vid.read()
        if img is None:
            print('Completed')
            break
        frame_id = frame_id + 1
        t1 = time.time()

        # boxes, scores, classes = detect_with_YOLOv3(img)
        boxes, scores, classes = load_detection_output(DETECTION_PATH, VIDEO_PATH, frame_id)

        names = []
        for i in range(len(classes)):
            names.append(class_names[int(classes[i])])
        names = np.array(names)
        converted_boxes = np.array(convert_boxes(img, boxes))
        features = encoder(img, converted_boxes)

        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in
                    zip(converted_boxes, scores, names, features)]

        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, NMS_MAX_OVERLAP, scores)
        detections = [detections[i] for i in indices]

        tracker.predict()
        # for track in tracker.tracks:
        #     bbox = track.to_tlbr()
        #     cv2.rectangle(img, (int(bbox[0]),int(bbox[1])), (int(bbox[2]),int(bbox[3])), (0,255,0), 1)
        tracker.update(detections)

        height, width, _ = img.shape
        
        if OUTPUT_PATH is not None:
            cv2.polylines(img, np.array([config["roi"]]), isClosed=True, color=(200,255,0), thickness=3)
            for check_region in config["check_regions"]:
                cv2.polylines(img, np.array([check_region]), isClosed=True, color=(227,211,232), thickness=3)
            for bbox in converted_boxes:
                cv2.rectangle(img, (int(bbox[0]),int(bbox[1])), (int(bbox[0]+bbox[2]),int(bbox[1]+bbox[3])), (0,0,255), 1)
            
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            class_name= track.get_class()
            color = colors[int(track.track_id) % len(colors)]
            color = [i * 255 for i in color]


            coord2 = [[int(bbox[0]),int(bbox[1])], 
                [int(bbox[2]),int(bbox[1])], 
                [int(bbox[2]),int(bbox[3])], 
                [int(bbox[0]),int(bbox[3])]]
            bounding_box = Polygon(coord2)
            
            if OUTPUT_PATH is not None:
                cv2.rectangle(img, (int(bbox[0]),int(bbox[1])), (int(bbox[2]),int(bbox[3])), color, 1)
                cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[0])+(len(class_name)
                        +len(str(track.track_id)))*12, int(bbox[1]+20)), color, -1)
                cv2.putText(img, class_name+"."+str(track.track_id), (int(bbox[0]), int(bbox[1]+10)), 0, 0.5,
                        (255,255,255), 2)
            
            if config['roi_poly'].intersects(bounding_box):
                
                # Chỉ vẽ các phương tiện cần đếm
                if int(class_name) > 0:
                    center = (int(((bbox[0]) + (bbox[2]))/2), int(((bbox[1])+(bbox[3]))/2))
                    pts[track.track_id].append(center)
                    track_history[track.track_id].append([center,frame_id,class_name, track.track_id])
                    
                    if OUTPUT_PATH is not None:
                        for j in range(1, len(pts[track.track_id])):
                            if pts[track.track_id][j-1] is None or pts[track.track_id][j] is None:
                                continue
                            thickness = 2 # int(np.sqrt(64/float(j+1))*2)
                            cv2.line(img, (pts[track.track_id][j-1]), (pts[track.track_id][j]), color, thickness)
                
        fps = 1./(time.time()-t1)
        
        if OUTPUT_PATH is not None:
            out.write(img)

        running = (frame_id * 1.0/frame_count)*100
        print("running: {:.2f}%".format(running), "    fps: {:.2f} ".format(fps), os.path.basename(os.path.normpath(VIDEO_PATH)))

    vid.release()
    if OUTPUT_PATH is not None:
        out.release()
        
    return track_history, frame_count

def run_video(VIDEO_NAME):
    
    t = time.time()
    CONFIG_PATH = 'zone_config/{}.txt'.format(VIDEO_NAME)
    VIDEO_PATH = '/dataset/Students/Team1/25_video/{}.mp4'.format(VIDEO_NAME)
    OUTPUT_PATH = '/dataset/Students/Team2/tracking/ss_{}.mp4'.format(VIDEO_NAME)
    DETECTION_PATH = '/storage/detection_result/test_set_a/{}/'.format(VIDEO_NAME)
    SUBMISSION_FILE = 'data/video/submission_{}.txt'.format(VIDEO_NAME)
    config = load_config(CONFIG_PATH)
    print(VIDEO_PATH)
    print(OUTPUT_PATH)
    print(DETECTION_PATH)
    print(SUBMISSION_FILE)
    
    track_history, frame_count = tracking(VIDEO_PATH, OUTPUT_PATH, DETECTION_PATH, config)
    
    count(track_history, frame_count, SUBMISSION_FILE, VIDEO_NAME, config)
    t = time.time()-t
    t = datetime.datetime.fromtimestamp(t).strftime('%H:%M:%S')
    print('video processing:', t)
    

def main():
    for i in [10]:
        run_video("cam_{:02d}".format(i))
    
if __name__ == "__main__":
    main()