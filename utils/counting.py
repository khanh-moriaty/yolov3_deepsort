import numpy as np
import sys
from shapely.geometry import Point
import os
import cv2
import scipy.stats
# from utils.classify_resnet50v2 import predict_class

# Sắp xếp các head moi theo độ dài tới head
# Sắp xếp các tail moi theo độ dài tới tail
def find_moi_nearest(head, tail, config):

    mois_head = config['mois_head']
    mois_tail = config['mois_tail']
    
    index_head, index_tail = [], []
    for index, moi_head in enumerate(mois_head):
        dist = Point(tuple(head)).distance(Point(tuple(moi_head)))
        index_head.append((index, dist))
    index_head.sort(key=lambda a: a[1])
    index_head = [x[0] for x in index_head]
    for index, moi_tail in enumerate(mois_tail):
        dist = Point(tuple(tail)).distance(Point(tuple(moi_tail)))
        index_tail.append((index, dist))
    index_tail.sort(key=lambda a: a[1])
    index_tail = [x[0] for x in index_tail]
    
    return index_head, index_tail, tail
    
# Tìm xem vector nào tạo góc nhỏ nhất so với vector (tail - head)
def find_moi_cosine(head, tail, config):
    pass
    
def find_moi(head, tail, config):
    # Chỉ quan tâm đếm tọa độ tâm của object (head[0] và tail[0])
    head, tail = head[0], tail[0]
    index_head, index_tail, tail = find_moi_nearest(head, tail, config)
    # Chỉ quan tâm đến head và tail gần nhất
    return index_head[0], index_tail[0], tail
    
def confirm_moi(index_head, index_tail, center, config):
    
    print("index_head =", index_head, "index_tail =", index_tail, end=' ')
    mois = config['mois']
    
    center = Point(center)
    for index, moi in enumerate(mois):
        # Tìm xem head và tail tìm được có phải MOI đang xét hay không.
        if [index_head, index_tail] in moi:
            # Confirmation process: Kiểm tra tail có nằm trong check_poly hay không.
            if config['check_poly'][index_tail].contains(center):
                # Nếu có thì MOI đang xét chính là MOI của object.
                print("center valid, moi =", index)
                return index
            else:
                # Nếu điều kiện trên không thỏa mãn thì object chưa ra khỏi ROI.
                # Có 2 trường hợp có thể xảy ra:
                # - Track bị mất dấu trước khi object ra khỏi ROI.
                # - Thuật toán find_moi trả về sai (head, tail).
                print("center invalid, moi = -1")
                return -1
            
    # Trường hợp vẫn chưa tìm được MOI khớp với (head, tail)
    print("MOI invalid, moi = -1")
    return -1
    
def count(track_history, track_img, frame_count, 
          SUBMISSION_FILE, VIDEO_NAME, CLASS_CROP_PATH, config,
          MINIMUM_DISTANCE=20):

    file = open(SUBMISSION_FILE, "w")
    track_history = [list(x) for x in track_history]

    log_file = open(SUBMISSION_FILE[:-4] + "_log.txt", 'w')
    stdout = sys.stdout
    sys.stdout = log_file
    
    for track in track_history:
        # print(track)
        if (len(track) < 5):
            continue
        track_id = track[-1][3]
        print("track_id: ", track_id)
        pt_head = Point(track[0][0])
        pt_tail = Point(track[-1][0])
        if pt_head.distance(pt_tail) <= MINIMUM_DISTANCE: 
            continue
        head, tail, out_point = find_moi(track[0], track[-1], config)
        moi = confirm_moi(head, tail, out_point, config)
        if moi == -1: 
            continue
        
        frame_id = track[-1][1] + config['mois_shift'][moi]
        if frame_id <= 0: frame_id = 1
        if frame_id > frame_count: 
            continue
        
        img_crop = track_img[track_id][1]
        # pred_class = predict_class(img_crop)
        pred_class = track[-1][2]
        
        # Majority Voting:
        class_votes = [history[2] for history in track]
        pred_class = int(scipy.stats.mode(class_votes)[0])

        # Ad-hoc solution:
        # if VIDEO_NAME[-2:] in ['11']: # cam_11 không có class 3 và 4
        #     if pred_class in [3, 4]: pred_class = 1
        # if VIDEO_NAME[-2:] in ['21', '22']: # cam_21, cam_22 không có class 3, 4
        #     if pred_class in [3, 4]: pred_class = 2
        
        if pred_class == -1: 
            continue
        crop_path = "{}_{:05d}_{}.jpg".format(VIDEO_NAME, frame_id, track_id)
        crop_path = os.path.join(CLASS_CROP_PATH, str(pred_class), crop_path)
        cv2.imwrite(crop_path, img_crop)
        
        moi_vector = np.array(track[-1][0]) - np.array(track[-2][0])
        center = (track[-1][0] + moi_vector * config['mois_shift'][moi]).astype(int)
        
        kq = VIDEO_NAME + " " + str(frame_id) + " " + str(moi + 1) + " " \
            + str(pred_class) + " " + str(center[0]) + " " + str(center[1])
        
        print(kq)
        file.write("".join(kq))
        file.write("\n")
            
    file.flush()
    file.close()
    sys.stdout = stdout
    log_file.flush()
    log_file.close()