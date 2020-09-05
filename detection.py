# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random, time

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances

from shapely.geometry import Polygon
from multiprocessing import Process

register_coco_instances("team2", {}, 
                        "/content/team2_dataset/augment/labels.json", 
                        "/content/team2_dataset/augment/")
MetadataCatalog.get("team2").set(thing_classes=["0","1","2","3","4"])

# mode: mỗi mode tương ứng với 1 model detect
# mode = 0: trời sáng
# mode = 1: trời sáng mưa
# mode = 2: trời tối
# mode = 3: trời tối trắng đen

MODEL_BASE_URL = "/mmlabstorage/workingspace/khanhmmlab/aicity/weights/FasterRCNN/"
MODEL_MODES = ["R_101_FPN_Baseline_Day.pth",
               "R_101_FPN_Baseline_Rain.pth",
               "R_101_FPN_Baseline_Night.pth",
               "R_101_FPN_Baseline_BW.pth",
               ]

def getConfig(mode):
    BASELINE = "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(BASELINE))
    # Restore from checkpoint
    cfg.MODEL.WEIGHTS = os.path.join(MODEL_BASE_URL, MODEL_MODES[mode])
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256   # số lượng ROI được đề xuất, mặc định là 512
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5  # 5 class theo format của BTC
    cfg.MODEL.RETINANET.NUM_CLASSES = cfg.MODEL.ROI_HEADS.NUM_CLASSES
    
    THRESHOLD = 0.5 # set a custom testing threshold for this model
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = THRESHOLD
    return cfg
    
# Kiểm tra xem hai object có giao nhau quá nhiều hay không
# Nếu giao nhau vượt qua iou_threshold thì sẽ cân nhắc giữ lại 
# object có confidence cao hơn.
def check_intersect(box1, box2, iou_threshold = 0.85):
    box1 = [int(box) for box in box1]
    box2 = [int(box) for box in box2]
    box1 = [[box1[0], box1[1]],
            [box1[2], box1[1]],
            [box1[2], box1[3]],
            [box1[0], box1[3]]]
    box2 = [[box2[0], box2[1]],
            [box2[2], box2[1]],
            [box2[2], box2[3]],
            [box2[0], box2[3]]]
    poly1 = Polygon(box1)
    poly2 = Polygon(box2)
    intersection = poly1.intersection(poly2).area
    union = poly1.union(poly2).area
    iou = intersection / union
    return (iou >= iou_threshold)
    
def post_processing(pred_bboxes, pred_classes, pred_scores):
    out_list = [[] for _ in range(5)]
    out_labels = [0, 1, 2, 3, 4]
    for index, pred_class in enumerate(pred_classes):
        if pred_classes[index] == -1: continue
        if pred_classes[index] >= 2:
            # Kiểm tra những bbox giao nhau với bbox hiện tại
            # (Chỉ kiểm tra xe hơi, xe khách và xe tải)
            for index2, pred_class2 in enumerate(pred_classes[index+1:]):
                # (Chỉ kiểm tra xe hơi, xe khách và xe tải)
                if pred_class2 < 2: continue
                # Kiểm tra xem 2 bbox có giao nhau quá nhiều hay không
                if not check_intersect(pred_bboxes[index],
                                       pred_bboxes[index+index2+1]): continue
                # Nếu có thì chỉ giữ lại cái nào có confidence cao hơn
                if pred_scores[index] <= pred_scores[index+index2+1]:
                    print('del1', index, pred_classes[index], pred_scores[index])
                    pred_classes[index] = -1
                    break
                print('del2', index+index2+1, pred_classes[index], pred_scores[index+index2+1])
                pred_classes[index+index2+1] = -1
            if pred_classes[index] == -1: continue
        if pred_classes[index] >= 0:
            out_list[out_labels[pred_classes[index]]].append((pred_bboxes[index], pred_scores[index]))
    return out_list

def test_video(VIDEO_PATH, OUTPUT_PATH, mode=0,
               SUBDIVISION=20):
    cfg = getConfig(mode)
    predictor = DefaultPredictor(cfg)
    vid = cv2.VideoCapture(VIDEO_PATH)
    FRAME_BASE_NAME = os.path.basename(VIDEO_PATH)
    FRAME_BASE_NAME = os.path.splitext(FRAME_BASE_NAME)[0] + "_{:05d}.txt"
    
    VIDEO_NAME = os.path.basename(VIDEO_PATH)
    VIDEO_NAME = os.path.splitext(VIDEO_NAME)[0]
    OUTPUT_PATH = os.path.join(OUTPUT_PATH, VIDEO_NAME)
    print(OUTPUT_PATH)
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    start_time = time.time()
    frame_id = 0
    t = time.time()
    while True:
        _, im = vid.read()
        if im is None:
            print('video ended')
            break

        frame_id += 1
        frame_name = FRAME_BASE_NAME.format(frame_id)
        out_img_name = os.path.splitext(frame_name)[0] + '.jpg'
        out_path = os.path.join(OUTPUT_PATH, frame_name)
        out_img_path = os.path.join(OUTPUT_PATH, out_img_name)
        fo = open(out_path, 'w')
        height, width = im.shape[:2]
        # im = cv2.resize(im, (600, 350))
        outputs = predictor(im)
        pred_bboxes = outputs['instances'].pred_boxes.tensor.cpu().numpy().tolist()
        pred_classes = outputs['instances'].pred_classes.cpu().numpy().tolist()
        pred_scores = outputs['instances'].scores.cpu().numpy().tolist()
        # print('pred_bboxes', pred_bboxes)
        # print('pred_classes', pred_classes)
        # print('pred_scores', pred_scores)
        # break
        out_list = post_processing(pred_bboxes, pred_classes, pred_scores)
        for class_id, pred_class in enumerate(out_list):
            for (pred_bbox, pred_score) in pred_class:

                out_str = [str(class_id),
                        str(pred_bbox[0] / width),
                        str(pred_bbox[1] / height),
                        str(pred_bbox[2] / width),
                        str(pred_bbox[3] / height),
                        str(pred_score)]
                out_str = ' '.join(out_str) + '\n'
                fo.write(out_str)
        fo.close()
        if frame_id % SUBDIVISION == 0:
            print(SUBDIVISION, 'frames processed. Time:', time.time() - t, frame_name)
            t = time.time()


        # print(outputs['instances'].pred_classes.tensor.cpu().numpy())
        # We can use `Visualizer` to draw the predictions on the image.
        # v = Visualizer(im[:, :, ::-1], MetadataCatalog.get("team1"), scale=1.2)
        # out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        # cv2.imwrite(out_img_path, out.get_image()[:, :, ::-1])

    print(time.time() - start_time)    

def main():
    # Video cần xử lý
    # VIDEO_PATH_10 = [
    #               "/dataset/Students/Team1/25_video/cam_02.mp4",
    # ]
    # VIDEO_PATH_11 = [
    #               "/dataset/Students/Team1/25_video/cam_07.mp4",
    # ]
    # VIDEO_PATH_12 = [
    #               "/dataset/Students/Team1/25_video/cam_21.mp4",
    # ]
    # VIDEO_PATH_13 = [
    #               "/dataset/Students/Team1/25_video/cam_06.mp4",
    # ]
    # VIDEO_PATH_14 = [
    #               "/dataset/Students/Team1/25_video/cam_11.mp4",
    # ]
    # VIDEO_PATH_15 = [
    #               "/dataset/Students/Team1/25_video/cam_23.mp4",
    # ]
    # mode 0
    VIDEO_PATH_1 = [
                  "/dataset/Students/Team1/25_video/cam_01.mp4",
                  "/dataset/Students/Team1/25_video/cam_04.mp4",
                  "/dataset/Students/Team1/25_video/cam_09.mp4",
    ]
    # mode 1
    VIDEO_PATH_2 = [
                  "/dataset/Students/Team1/25_video/cam_02.mp4",
                  "/dataset/Students/Team1/25_video/cam_06.mp4",
                  "/dataset/Students/Team1/25_video/cam_07.mp4",
    ]
    # mode 2
    VIDEO_PATH_3 = [
                  "/dataset/Students/Team1/25_video/cam_03.mp4",
                  "/dataset/Students/Team1/25_video/cam_05.mp4",
                  "/dataset/Students/Team1/25_video/cam_08.mp4",
                  "/dataset/Students/Team1/25_video/cam_13.mp4",
    ]
    # mode 3
    VIDEO_PATH_4 = [
                  "/dataset/Students/Team1/25_video/cam_20.mp4",
                  "/dataset/Students/Team1/25_video/cam_22.mp4",
                  "/dataset/Students/Team1/25_video/cam_25.mp4",
    ]
    # mode 0
    VIDEO_PATH_5 = [
                  "/dataset/Students/Team1/25_video/cam_14.mp4",
                  "/dataset/Students/Team1/25_video/cam_16.mp4",
                  "/dataset/Students/Team1/25_video/cam_18.mp4",
    ]
    # mode 1
    VIDEO_PATH_6 = [
                  "/dataset/Students/Team1/25_video/cam_11.mp4",
                  "/dataset/Students/Team1/25_video/cam_21.mp4",
                  "/dataset/Students/Team1/25_video/cam_23.mp4",
    ]
    # mode 2
    VIDEO_PATH_7 = [
                  "/dataset/Students/Team1/25_video/cam_15.mp4",
                  "/dataset/Students/Team1/25_video/cam_17.mp4",
                  "/dataset/Students/Team1/25_video/cam_19.mp4",
    ]
    # mode 0
    VIDEO_PATH_8 = [
                  "/dataset/Students/Team1/25_video/cam_24.mp4",
                  "/dataset/Students/Team1/25_video/cam_10.mp4",
                  "/dataset/Students/Team1/25_video/cam_12.mp4",
    ]
    
    def test_dir(VIDEO_DIR, OUTPUT_PATH, mode):
        for VIDEO_PATH in VIDEO_DIR:
            test_video(VIDEO_PATH, OUTPUT_PATH, mode)
                  
    # Thư mục chứa các thư mục label
    OUTPUT_PATH = "/storage/detection_result/test_set_a/sub17/"
    p1 = Process(target=test_dir, args=(VIDEO_PATH_1, OUTPUT_PATH, 0))
    p2 = Process(target=test_dir, args=(VIDEO_PATH_2, OUTPUT_PATH, 1))
    p3 = Process(target=test_dir, args=(VIDEO_PATH_3, OUTPUT_PATH, 2))
    p4 = Process(target=test_dir, args=(VIDEO_PATH_4, OUTPUT_PATH, 3))
    p5 = Process(target=test_dir, args=(VIDEO_PATH_5, OUTPUT_PATH, 0))
    p6 = Process(target=test_dir, args=(VIDEO_PATH_6, OUTPUT_PATH, 1))
    p7 = Process(target=test_dir, args=(VIDEO_PATH_7, OUTPUT_PATH, 2))
    p8 = Process(target=test_dir, args=(VIDEO_PATH_8, OUTPUT_PATH, 0))
    p1.start()
    p2.start()
    p3.start()
    p4.start()
    p5.start()
    p6.start()
    p7.start()
    p8.start()
    p1.join()
    p2.join()
    p3.join()
    p4.join()
    p5.join()
    p6.join()
    p7.join()
    p8.join()
    # test_video(VIDEO_PATH, OUTPUT_PATH, mode=1)

if __name__ == "__main__":
    main()