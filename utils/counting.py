import numpy as np
import sys
import os
import cv2
import math
import time
import scipy.stats
from shapely.geometry import Point, Polygon
from utils.motion_proposal import find_moi, confirm_moi
from utils.prediction import predict_track_history
from utils.track_merging import merge_tracks
    
def valid_track(track, MINIMUM_DISTANCE=20):
    if (len(track) < 4):
        return False
    pt_head = Point(track[0][0])
    pt_tail = Point(track[-1][0])
    if pt_head.distance(pt_tail) <= MINIMUM_DISTANCE: 
        return False
    return True
    
def count(track_history, track_img, frame_count, 
          SUBMISSION_FILE, VIDEO_NAME, CLASS_CROP_PATH, config):

    file = open(SUBMISSION_FILE, "w")
    track_history = [list(x) for x in track_history]
    track_history = [track for track in track_history if valid_track(track)]
    print('track history len =', len(track_history))

    log_file = open(SUBMISSION_FILE[:-4] + "_log.txt", 'w')
    stdout = sys.stdout
    sys.stdout = log_file
    
    t = time.time()
    roi_poly = config['roi_btc_poly']
    track_enter, track_enter_taylor, track_escape, track_escape_taylor = predict_track_history(track_history, roi_poly)
    
    track_history, track_enter, track_enter_taylor, track_escape, track_escape_taylor = \
        merge_tracks(track_history, track_enter, track_enter_taylor, track_escape, track_escape_taylor)
        
    sys.stdout = stdout
    print('merge time: ', time.time() - t)
    sys.stdout = log_file
    
    print(track_history)
    counted_track = []
    
    for track, enter, escape in zip(track_history, track_enter, track_escape):
        # print(track)
        if escape is None:
            x, y, frame_shift = None, None, None
        else:
            x, y, frame_shift = escape
        track_id = track[-1][3]
        print("track_id: ", track_id)
        if enter is None or escape is None or enter[0] is None or escape[0] is None:
            head, tail, out_point = find_moi(track[0][0], track[-1][0], config)
        else:
            head, tail, out_point = find_moi(enter[:2], escape[:2], config)
        if head == -1 or tail == -1:
            continue
        moi = confirm_moi(head, tail, track[-1][0], config)
        moi = confirm_moi(head, tail, out_point, config)
        if moi == -1: 
            continue
        if moi >= config['k']:
            continue
        
        if frame_shift is None or frame_shift < 0: # Dời frame theo config nếu không thể predict
            frame_id = track[-1][1] + config['mois_shift'][moi]
            moi_vector = np.array(track[-1][0]) - np.array(track[-2][0])
            center = (track[-1][0] + moi_vector * config['mois_shift'][moi]).astype(int)
        else:
            frame_id = track[-1][1] + frame_shift
            center = x, y
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
        if pred_class == 0:
            continue
        
        kq = VIDEO_NAME + " " + str(frame_id) + " " + str(moi + 1) + " " \
            + str(pred_class) + " " + str(center[0]) + " " + str(center[1]) + " " + str(track_id)
        
        if enter is not None and enter[0] is not None:
            track[0][0] = (enter[0], enter[1])
        if escape is not None and escape[0] is not None:
            track[-1][0] = (escape[0], escape[1])
            track[-1][1] = frame_id
        counted_track.append(track)
        print(kq)
        file.write("".join(kq))
        file.write("\n")
            
    file.flush()
    file.close()
    sys.stdout = stdout
    log_file.flush()
    log_file.close()
    return counted_track