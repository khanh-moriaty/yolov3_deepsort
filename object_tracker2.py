from absl import flags
import sys
FLAGS = flags.FLAGS
FLAGS(sys.argv)

import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob

import tensorflow as tf
from yolov3_tf2.models import YoloV3
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils import convert_boxes

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet

from shapely.geometry import Point, Polygon

class_names = [c.strip() for c in open('./data/labels/obj.names').readlines()]
yolo = YoloV3(classes=len(class_names))
yolo.load_weights('./weights/yolov3.tf')

max_cosine_distance = 0.5
nn_budget = None
nms_max_overlap = 0.8

model_filename = 'model_data/mars-small128.pb'
encoder = gdet.create_box_encoder(model_filename, batch_size=1)
metric = nn_matching.NearestNeighborDistanceMetric('cosine', max_cosine_distance, nn_budget)
tracker = Tracker(metric)

vid = cv2.VideoCapture('./data/video/cam_18_2p.mp4')
vid_fps = vid.get(cv2.CAP_PROP_FPS)
codec = cv2.VideoWriter_fourcc(*'XVID')
vid_fps =int(vid.get(cv2.CAP_PROP_FPS))
vid_width,vid_height = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('./data/video/cam_18_2p_test1.avi', codec, vid_fps, (vid_width, vid_height))

from _collections import deque
pts = [deque(maxlen=200) for _ in range(5000)]
track_history = [deque(maxlen=20000) for _ in range(1000000)]

counter = []
Xe_may = []
Xe_hoi = []
Xe_Khach = []
Xe_tai = []

frame_count = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
loong = frame_count/vid_fps
sec = 0.0
frame_id = 0
coord = [
            [132, 307],
            [605, 148],
            [941, 249],
            [885, 691]
            ]

check1 = [
            [150, 270],
            [510, 445],
            [160, 700],
            [0, 700],
            [0, 307]
            ]
check2 = [
            [680+10, 210],
            [810, 130],
            [950, 170],
            [935, 280]
            ]

poly = Polygon(coord)
checker1 = Polygon(check1)
checker2 = Polygon(check2)

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

while True:
    _, img = vid.read()
    if img is None:
        print('Completed')
        break

    boxes, scores, classes = detect_with_YOLOv3(img)

    names = []
    for i in range(len(classes)):
        names.append(class_names[int(classes[i])])
    names = np.array(names)
    converted_boxes = convert_boxes(img, boxes)
    features = encoder(img, converted_boxes)

    detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in
                  zip(converted_boxes, scores, names, features)]

    print('detections',detections)
    break

    boxs = np.array([d.tlwh for d in detections])
    scores = np.array([d.confidence for d in detections])
    classes = np.array([d.class_name for d in detections])
    indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
    detections = [detections[i] for i in indices]

    tracker.predict()
    tracker.update(detections)

    cmap = plt.get_cmap('tab20b')
    colors = [cmap(i)[:3] for i in np.linspace(0,1,20)]

    current_count = int(0)

    height, width, _ = img.shape
    
    

    cv2.line(img, (731, 168), (205, 424), (214,228,201), thickness=2)
    cv2.line(img, (526,566), (842,199), (227,211,232), thickness=2)
    #cv2.putText(img, "h1", (340,200), 0, 0.75, (0, 0, 0), 2)
    #cv2.putText(img, "t1", (250,400), 0, 0.75, (0, 0, 0), 2)
    #cv2.putText(img, "h2", (560,400), 0, 0.75, (0, 0, 0), 2)
    #cv2.putText(img, "t2", (475,200), 0, 0.75, (0, 0, 0), 2)

    
    
    
    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update >1:
            continue
        bbox = track.to_tlbr()
        class_name= track.get_class()
        color = colors[int(track.track_id) % len(colors)]
        color = [i * 255 for i in color]

        cv2.polylines(img, np.array([coord]), isClosed=True, color=(200,255,0), thickness=3)
        #cv2.polylines(img, np.array([check1]), isClosed=True, color=(214,228,201), thickness=3)
        #cv2.polylines(img, np.array([check2]), isClosed=True, color=(227,211,232), thickness=3)

        coord2 = [[int(bbox[0]),int(bbox[1])], 
              [int(bbox[2]),int(bbox[1])], 
              [int(bbox[2]),int(bbox[3])], 
              [int(bbox[0]),int(bbox[3])]]
        bounding_box = Polygon(coord2)
        
        if poly.intersects(bounding_box):
            
            cv2.rectangle(img, (int(bbox[0]),int(bbox[1])), (int(bbox[2]),int(bbox[3])), color, 1)
            cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[0])+(len(class_name)
                    +len(str(track.track_id)))*12, int(bbox[1]+20)), color, -1)
            cv2.putText(img, class_name+"."+str(track.track_id), (int(bbox[0]), int(bbox[1]+10)), 0, 0.5,
                    (255,255,255), 2)
            if class_name == '1' or class_name == '2' or class_name == '3' or class_name == '4':
                if class_name == '1':
                  Xe_may.append(int(track.track_id))
                elif class_name == '2':
                  Xe_hoi.append(int(track.track_id))
                elif class_name == '3':
                  Xe_Khach.append(int(track.track_id))
                elif class_name == '4':
                  Xe_tai.append(int(track.track_id))
                counter.append(int(track.track_id))
                current_count += 1
                center = (int(((bbox[0]) + (bbox[2]))/2), int(((bbox[1])+(bbox[3]))/2))
                pts[track.track_id].append(center)
                track_history[track.track_id].append([center,frame_id,class_name, track.track_id])
                
                

                for j in range(1, len(pts[track.track_id])):
                  if pts[track.track_id][j-1] is None or pts[track.track_id][j] is None:
                    continue
                  thickness = int(np.sqrt(64/float(j+1))*2)
                  cv2.line(img, (pts[track.track_id][j-1]), (pts[track.track_id][j]), color, thickness)
             
    #print("frame_id: ", frame_id, "\n", set(Xe_may))
    
    total_count = len(set(counter))
    Count_Xe_may = len(set(Xe_may))
    Count_Xe_hoi = len(set(Xe_hoi))
    Count_Xe_Khach = len(set(Xe_Khach))
    Count_Xe_tai = len(set(Xe_tai))
    #cv2.rectangle(img,(0, 50), (125, 250), (211,209,153), -1)

    #cv2.putText(img, "ROI: " + str(current_count), (0, 80), 0, 0.75, (0, 0, 0), 2)

    #cv2.putText(img, "Sum: " + str(total_count), (0,110), 0, 0.75, (0, 0, 0), 2)

    #cv2.putText(img, "x.may: " + str(Count_Xe_may), (0,142), 0, 0.75, (0, 0, 0), 2)

    #cv2.putText(img, "x.hoi: " + str(Count_Xe_hoi), (0,174), 0, 0.75, (0, 0, 0), 2)

    #cv2.putText(img, "x.khach: " + str(Count_Xe_Khach), (0,206), 0, 0.75, (0, 0, 0), 2)

    #cv2.putText(img, "x.tai: " + str(Count_Xe_tai), (0,238), 0, 0.75, (0, 0, 0), 2)

    fps = 1./(time.time()-t1)
    
    cv2.putText(img, "FPS: {:.2f}".format(fps), (0,30), 0, 1, (0,0,255), 2)
    out.write(img)

    frame_id = frame_id + 1
    if(frame_id%(vid_fps) == 0):
      sec = sec + 1
    running = (sec/loong)*100
    print("running: {:.2f}%".format(running), "    fps: {:.2f} ".format(fps))



def find_moi(a, b):

  mois = [[731, 168], [526,566], [205, 424], [842,199]]  
  index_a = 0
  index_b = 2
  min_a = Point(tuple(a[0])).distance(Point(tuple(mois[0])))
  min_b = Point(tuple(b[0])).distance(Point(tuple(mois[int(len(mois)/2)])))
  for i in range(0, int(len(mois)/2)):
    if min_a > Point(tuple(a[0])).distance(Point(tuple(mois[i]))):
      index_a = i
      min_a = Point(tuple(a[0])).distance(Point(tuple(mois[i])))
  for j in range(int(len(mois)/2), len(mois)):
    if min_b > Point(tuple(b[0])).distance(Point(tuple(mois[j]))):
      index_b = j
      min_b = Point(tuple(b[0])).distance(Point(tuple(mois[j])))
  return (index_a + 1), (index_b + 1), tuple(b[0])

def confirm_moi(index_a, index_b, center):
  if (index_a == 1 and index_b == 3):
    if((checker1.contains(Point(center)))):
      print("index_a =", index_a, "and index_b =", index_b, "center available, moi = 1")
      return 1
    elif not(checker1.contains(Point(center))):
      print("center not available!, moi = -1")
      return -1
  elif (index_a == 2 and index_b == 4 and (checker2.contains(Point(center)))):
    if ((checker2.contains(Point(center)))):
      print("index_a =", index_a, "and index_b =", index_b, "center available, moi = 2")
      return 2
    elif not((checker2.contains(Point(center)))):
      print("center not available!, moi = -1")
      return -1
  print("index_a =", index_a, "and index_b =", index_b, "invalid moi, moi = -1")
  return -1


file = open("data/video/submission.txt", "w")
file.close()

for i in range(len(track_history)):
  if (list(track_history[i]) == []):
    continue
  print("track_id: ", list(track_history[i])[-1][3])
  #print("len: ", len(list(track_history[i])))
  if (len(list(track_history[i])) < 5):
    continue
  head, tail, out_point = find_moi(list(track_history[i])[0], list(track_history[i])[-1])
  moi = confirm_moi(head, tail, out_point)
  kq = "BVUB" + " " + str(list(track_history[i])[-1][1]) + " " + str(moi) + " " + str(list(track_history[i])[-1][2]) + " " + str(list(track_history[i])[-1][0][0]) + " " + str(list(track_history[i])[-1][0][1])

  #str(list(track_history[i])[-1][3])
  #str(list(track_history[i])[-1][0][0]) + " " + str(list(track_history[i])[-1][0][1])
  if (moi == -1):
    continue
  else:
    #print(kq)
    file = open("data/video/submission.txt", "a") 
    file.write("".join(kq))
    file.write("\n")
    file.close()

vid.release()
out.release()
cv2.destroyAllWindows()