import numpy as np
import math
from shapely.geometry import Point, Polygon

def inROI(bbox_poly, roi_poly, OVERLAP_THRESHOLD=0.05):
    return roi_poly.intersection(bbox_poly).area >= bbox_poly.area * OVERLAP_THRESHOLD

def taylor_series(history, roi_poly, predict_tail):
        
    v = [math.sqrt((history[i][0][0] - history[i-1][0][0])**2 + 
                   (history[i][0][1] - history[i-1][0][1])**2) / (history[i][1] - history[i-1][1])
         for i in range(1, len(history))]
    
    a = [(v[i] - v[i-1]) / (history[i+1][1] - history[i][1]) for i in range(1, len(v))]
    b = [(a[i] - a[i-1]) / (history[i+2][1] - history[i+1][1]) for i in range(1, len(a))]
    c = [(b[i] - b[i-1]) / (history[i+3][1] - history[i+2][1]) for i in range(1, len(b))]
    
    v = np.median(v)
    a = np.median(a)
    b = np.median(b)
    c = np.median(c)
    
    w = [(history[i][4][2] - history[i][4][0]) for i in range(len(history))]
    v_w = [(w[i] - w[i-1]) / (history[i][1] - history[i-1][1]) for i in range(1, len(w))]
    a_w = [(v_w[i] - v_w[i-1]) / (history[i+1][1] - history[i][1]) for i in range(1, len(v_w))]
    
    w = np.median(w)
    v_w = np.median(v_w)
    a_w = np.median(a_w)
    
    h = [(history[i][4][3] - history[i][4][1]) for i in range(len(history))]
    v_h = [(h[i] - h[i-1]) / (history[i][1] - history[i-1][1]) for i in range(1, len(h))]
    a_h = [(v_h[i] - v_h[i-1]) / (history[i+1][1] - history[i][1]) for i in range(1, len(v_h))]
    
    h = np.median(h)
    v_h = np.median(v_h)
    a_h = np.median(a_h)
    
    return (v, a, b, c), (w, v_w, a_w), (h, v_h, a_h)

def normalize_vector(move_x, move_y):
    move_magnitude = np.linalg.norm((move_x, move_y))
    if move_magnitude == 0: return None, None, None
    move_x /= move_magnitude
    move_y /= move_magnitude
    return move_x, move_y, move_magnitude

def calc_tail_vector(history, MAX_HISTORY_LENGTH=5):
    history = history[-MAX_HISTORY_LENGTH:]
    move_x = history[-1][0][0] - history[0][0][0]
    move_y = history[-1][0][1] - history[0][0][1]
    return normalize_vector(move_x, move_y)

def calc_head_vector(history, MAX_HISTORY_LENGTH=5):
    history = history[:MAX_HISTORY_LENGTH]
    # move_x = history[-1][0][0] - history[0][0][0]
    # move_y = history[-1][0][1] - history[0][0][1]
    move_x = history[0][0][0] - history[-1][0][0]
    move_y = history[0][0][1] - history[-1][0][1]
    return normalize_vector(move_x, move_y)

def predict_with_taylor(taylor, time, predict_tail):
    direction, displacement, bbox_width, bbox_height = taylor
    move_x, move_y, move_magnitude = direction
    x0, y0, v, a, b, c = displacement
    w, v_w, a_w = bbox_width
    h, v_h, a_h = bbox_height
    if (time < 0):
        move_x, move_y = -move_x, -move_y
    x = x0 + move_x * abs(v * time + 1/2 * a * time * time)
    y = y0 + move_y * abs(v * time + 1/2 * a * time * time)
    w = w
    h = h
    return x, y, w, h

def predict_escape_core(taylor, roi_poly, predict_tail):
    
    res = None, None, None
    x, y, w, h = predict_with_taylor(taylor, 0, predict_tail)
    bbox = [[x - w/2, y - h/2],
            [x + w/2, y - h/2],
            [x + w/2, y + h/2],
            [x - w/2, y + h/2]]
    try:
        bbox_poly = Polygon(bbox)
    except:
        return res
    forward = inROI(bbox_poly, roi_poly)
    
    if forward: # If in ROI: searches forward
        search_range = range(1000)
    else:
        if not predict_tail: 
            return taylor[1][0], taylor[1][1], 0
        search_range = range(0, -1000, -1)
        
    
    for t in search_range:
        x, y, w, h = predict_with_taylor(taylor, t, predict_tail)
        bbox = [[x - w/2, y - h/2],
                [x + w/2, y - h/2],
                [x + w/2, y + h/2],
                [x - w/2, y + h/2]]
        try:
            bbox_poly = Polygon(bbox)
        except:
            continue
        
        if forward != inROI(bbox_poly, roi_poly):
            res = int(x), int(y), t
            break
    
    print(res)
    return res

# Dự đoán thời điểm phương tiện thoát khỏi ROI
def predict_escape(history, roi_poly, predict_tail=True, 
                   MAX_TAYLOR_HISTORY=10):
    
    res = None, None
    if predict_tail:
        direction = calc_tail_vector(history)
        x0, y0 = history[-1][0]
        taylor_history = history[-MAX_TAYLOR_HISTORY:]
    else:
        direction = calc_head_vector(history)
        x0, y0 = history[0][0]
        taylor_history = history[:MAX_TAYLOR_HISTORY]
    if direction[0] is None:
        return res
    
    movement, bbox_width, bbox_height = taylor_series(taylor_history, roi_poly, predict_tail)
    displacement = (x0, y0) + movement
    # direction = move_x, move_y
    taylor = direction, displacement, bbox_width, bbox_height
    print(taylor)
    
    return predict_escape_core(taylor, roi_poly, predict_tail), taylor

def predict_track_history(track_history, roi_poly):
    track_enter = []
    track_enter_taylor = []
    track_escape = []
    track_escape_taylor = []
    for track in track_history:
        print('prediction:', track[-1][3])
        prediction, taylor = predict_escape(track, roi_poly, predict_tail=False)
        if prediction is None or taylor is None:
            track_enter.append(None)
            track_enter_taylor.append(None)
        else:
            track_enter.append(prediction[:3])
            track_enter_taylor.append(taylor)
        
        prediction, taylor = predict_escape(track, roi_poly, predict_tail=True)
        if prediction is None or taylor is None:
            track_escape.append(None)
            track_escape_taylor.append(None)
        else:
            track_escape.append(prediction[:3])
            track_escape_taylor.append(taylor)
    
    return track_enter, track_enter_taylor, track_escape, track_escape_taylor