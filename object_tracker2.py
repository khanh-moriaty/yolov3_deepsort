import time
import numpy as np
import cv2
import shutil
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
from utils.post_proc_visualize import visualize

from _collections import deque
from shapely.geometry import Point, MultiPoint, Polygon
from shapely.ops import nearest_points


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

# Discards all objects that are too far from ROI.
def inROI(detection, roi_poly,
          ROI_PROXIMITY=0):
    x, y = detection.to_xyah()[:2]
    center = Point(x, y)
    dist = roi_poly.distance(center)
    # print(detection.class_name, dist)
    return dist <= ROI_PROXIMITY
    
MOI_COLOR = [
             (204, 51, 153, ),  # pink
             (75, 0, 130, ),  # violet
             (139, 69, 19, ),  # brown
             (255, 153, 51, ),  # orange
             (65, 105, 225, ),  # light blue
             (0, 206, 209, ),  # cyan
             (50, 205, 50, ),  # light green
             (128, 128, 0, ),  # dark yellow
             (220, 20, 60, ),  # red
             (255, 215, 0, ),  # light yellow
             (0, 100, 0, ),  # dark green
             (25, 25, 112, ),  # dark blue
             ]

# Leave OUTPUT_PATH=None if you don't want to output visualize video
def tracking(VIDEO_PATH, OUTPUT_PATH, DETECTION_PATH, config,
             CROP_PADDING=0.15):
    
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
    track_img = [[10**9, None] for _ in range(500000)]

    frame_count = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_id = 0

    cmap = plt.get_cmap('tab20b')
    colors = [cmap(i)[:3] for i in np.linspace(0,1,20)]
    
    while True:
        _, img = vid.read()
        if img is None:
            print('Completed')
            break
        img_copy = img.copy()
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
        
        # [print(x.wkt, end=' ') for x in detection_space]
        # print(detection_multipoint.wkt)
        
        # detections = [detection for detection in detections if inROI(detection, config['roi_poly'])]
        # [print(d.class_name) for d in detections]
        # break

        tracker.predict()
        # for track in tracker.tracks:
        #     bbox = track.to_tlbr()
        #     cv2.rectangle(img, (int(bbox[0]),int(bbox[1])), (int(bbox[2]),int(bbox[3])), (0,255,0), 1)
        tracker.update(detections)

        detection_space = []
        for d in detections:
            tmp = tuple(d.to_xyah()[:2])
            detection_space.append(Point(tmp))
        detection_multipoint = MultiPoint(detection_space)
        
        height, width, _ = img.shape
        
        if OUTPUT_PATH is not None:
            cv2.polylines(img, np.array([config["roi"]]), isClosed=True, color=(200,255,0), thickness=3)
            for check_region in config["check_regions"]:
                cv2.polylines(img, np.array([check_region]), isClosed=True, color=(227,211,232), thickness=3)
            for moi_id, moi in enumerate(config["mois"]):
                for [index_head, index_tail] in moi:
                    cv2.arrowedLine(img, tuple(config["mois_head"][index_head]), tuple(config["mois_tail"][index_tail]), 
                                    color=MOI_COLOR[moi_id][::-1], thickness=2, tipLength=0.01)
            for index, bbox in enumerate(converted_boxes):
                cv2.rectangle(img, (int(bbox[0]),int(bbox[1])), (int(bbox[0]+bbox[2]),int(bbox[1]+bbox[3])), (0,0,255), 1)
                # cv2.rectangle(img, (int(bbox[0]+bbox[2])+(len(classes[index])+1)*12,int(bbox[1]+bbox[3])), 
                #               (int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3])+20), (0,0,0), -1)
                # cv2.putText(img, classes[index], (int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3])+10), 0, 0.5,
                #         (255,255,255), 2)
            
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            bbox = [int(x) for x in bbox]
            class_name= track.get_class()
            color = colors[int(track.track_id) % len(colors)]
            color = [i * 255 for i in color]

            center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
            roi_centroid = config['roi_poly'].centroid
            dist = Point(center).distance(roi_centroid)
            if dist < track_img[track.track_id][0]:
                track_img[track.track_id][0] = dist
                xmin = np.clip(int(bbox[0] - (bbox[2] - bbox[0]) * CROP_PADDING), 0, width)
                ymin = np.clip(int(bbox[1] - (bbox[3] - bbox[1]) * CROP_PADDING), 0, height)
                xmax = np.clip(int(bbox[2] + (bbox[2] - bbox[0]) * CROP_PADDING), 0, width)
                ymax = np.clip(int(bbox[3] + (bbox[3] - bbox[1]) * CROP_PADDING), 0, height)
                track_img[track.track_id][1] = img_copy[ymin:ymax, xmin:xmax].copy()
                
                
            # Finds nearest bbox to track and determines its class
            center_pts = Point(center)
            
            # nearest_bbox = nearest_points(detection_multipoint, center_pts)
            # print(nearest_bbox[0].wkt, detection_space.index(nearest_bbox[0]), 
            #       classes[detection_space.index(nearest_bbox[0])])
            # detection_class_name = classes[detection_space.index(nearest_bbox[0])]
            
            min_dist = 10**9
            min_index = 0
            for index, point in enumerate(detection_space):
                dist = Point(point).distance(center_pts)
                if dist < min_dist:
                    min_dist = dist
                    min_index = index
            if len(detections) == 0: 
                detection_class_name = '0'
            else: 
                detection_class_name = detections[min_index].class_name
            

            coord2 = np.array([[bbox[0],bbox[1]], 
                               [bbox[2],bbox[1]], 
                               [bbox[2],bbox[3]], 
                               [bbox[0],bbox[3]]])
            bounding_box = Polygon(coord2)
            
            if OUTPUT_PATH is not None:
                cv2.rectangle(img, (bbox[0],bbox[1]), (bbox[2],bbox[3]), color, 1)
                cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0]+(len(detection_class_name)
                        +len(str(track.track_id)))*12, bbox[1]+20), color, -1)
                cv2.putText(img, detection_class_name+"."+str(track.track_id), (bbox[0], bbox[1]+10), 0, 0.5,
                        (255,255,255), 2)
            
            if config['roi_poly'].intersects(bounding_box):
                
                # Chỉ vẽ các phương tiện cần đếm
                if True: # int(class_name) > 0:
                    pts[track.track_id].append(center)
                    track_history[track.track_id].append([center, frame_id, detection_class_name, track.track_id, coord2])
                    # print(frame_id, track.track_id, class_name, detection_class_name)
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
        
    return track_history, track_img, frame_count

def run_video(VIDEO_NAME):
    
    t = time.time()
    CONFIG_PATH = 'zone_config/sub26/{}.txt'.format(VIDEO_NAME)
    VIDEO_PATH = '/dataset/Students/Team1/25_video/{}.mp4'.format(VIDEO_NAME)
    VIDEO_PATH = '/storage/video_cut/5p/{}.mp4'.format(VIDEO_NAME)
    OUTPUT_PATH = '/dataset/Students/Team2/tracking/sub28/{}.mp4'.format(VIDEO_NAME)
    REID_OUTPUT_PATH = '/dataset/Students/Team2/reid/sub28/{}.mp4'.format(VIDEO_NAME)
    DETECTION_PATH = '/storage/detection_result/test_set_a/sub15/{}/'.format(VIDEO_NAME)
    CLASS_CROP_PATH = '/dataset/Students/Team2/crops/sub28/{}/'.format(VIDEO_NAME)
    SUBMISSION_FILE = '/storage/submissions/sub28/submission_{}.txt'.format(VIDEO_NAME)
    config = load_config(CONFIG_PATH)
    print(VIDEO_PATH)
    print(OUTPUT_PATH)
    print(DETECTION_PATH)
    print(SUBMISSION_FILE)
    
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(REID_OUTPUT_PATH), exist_ok=True)
    if os.path.exists(CLASS_CROP_PATH):
        shutil.rmtree(CLASS_CROP_PATH)
    for x in ['0','1','2','3','4']:
        os.makedirs(os.path.join(CLASS_CROP_PATH, x), exist_ok=True)
    os.makedirs(os.path.dirname(SUBMISSION_FILE), exist_ok=True)
    
    track_history, track_img, frame_count = tracking(VIDEO_PATH, OUTPUT_PATH, DETECTION_PATH, config)
    t = time.time()-t
    t = datetime.datetime.fromtimestamp(t).strftime('%H:%M:%S')
    print('video processing:', t)
    t = time.time()
    
    track_history = count(track_history, track_img, frame_count, SUBMISSION_FILE, VIDEO_NAME, CLASS_CROP_PATH, config)
    print('counting: {:.2f}'.format(time.time() - t))
    
    t = time.time()
    visualize(track_history, OUTPUT_PATH, REID_OUTPUT_PATH)
    print('reid: {:.2f}'.format(time.time() - t))
    

def main():
    video_list = [1,2,3] # track1
    video_list = [6,7,8] # track2
    video_list = [11,12,13] # track3
    video_list = [16,17] # track4
    video_list = [21,22,23,24,18] # track5
    video_list = [10,5] # track6
    video_list = [9] # track7
    video_list = [4,20,19] # track8
    video_list = [15,25,14] # track9
    
    video_list = [14] # track9
    video_list = [15] # track9
    
    # video_list = [1,2,3,15] # track1
    # video_list = [6,7,8,19] # track2
    # video_list = [11,12,13,14] # track3
    # video_list = [16,17,20] # track4
    # video_list = [21,22,23,24,18] # track5
    # video_list = [10,5,25] # track6
    # video_list = [9,4] # track7
    
    # video_list = [22]
    
    for i in video_list:
        run_video("cam_{:02d}".format(i))
    
if __name__ == "__main__":
    main()