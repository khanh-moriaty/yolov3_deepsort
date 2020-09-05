import numpy as np
from utils.maximum_matching import MaximumMatching
from utils.prediction import predict_with_taylor
from utils.motion_proposal import calc_angle_vectors
from shapely.geometry import Point
import math

def potential_match(track1, track2,
                    MAX_MATCH_DISTANCE=80, MAX_MATCH_VECTOR=math.pi/4, 
                    MIN_VALID_VECTOR_DISTANCE=20):
    track1, enter1, enter_taylor1, escape1, escape_taylor1 = track1
    track2, enter2, enter_taylor2, escape2, escape_taylor2 = track2
    time = track2[0][1] - track1[-1][1]
    if escape1 is None:
        return False
    track1_predict = Point(predict_with_taylor(escape_taylor1, time)[:2])
    # track2_predict = Point(predict_with_taylor(enter_taylor2, time)[:2])
    print(track1[-1][3], track2[-1][3], time, predict_with_taylor(escape_taylor1, time), track2[0][0])
    
    track1 = Point(track1[-1][0])
    track2 = Point(track2[0][0])
    res = (enter2 is None or enter_taylor2[0][2] <= MIN_VALID_VECTOR_DISTANCE \
        or calc_angle_vectors(escape_taylor1[0][:2], -np.array(enter_taylor2[0][:2])) <= MAX_MATCH_VECTOR) \
            and track1_predict.distance(track2) <= MAX_MATCH_DISTANCE
    print(res)
    # print(type(escape_taylor1[0]), escape_taylor1[0], enter_taylor2[0])
    return res

def merge_tracks(track_history, track_enter, track_enter_taylor, track_escape, track_escape_taylor,
                 MAX_WAIT_FRAME=120):
    # return track_history, track_enter, track_enter_taylor, track_escape, track_escape_taylor

    assert len(track_history) == len(track_escape), "history and escape is not the same length."
    
    new_history = []
    new_enter = []
    new_enter_taylor = []
    new_escape = []
    new_escape_taylor = []
    
    graph = np.zeros(shape=(len(track_history), len(track_history)), dtype=np.uint8)

    for index1, (track1, enter1, enter_taylor1, escape1, escape_taylor1) in \
        enumerate(zip(track_history, track_enter, track_enter_taylor, track_escape, track_escape_taylor)):
        for index2, (track2, enter2, enter_taylor2, escape2, escape_taylor2) in \
            enumerate(zip(track_history, track_enter, track_enter_taylor, track_escape, track_escape_taylor)):
            
            if index1 == index2:
                continue
            time = track2[0][1] - track1[-1][1]
            if time < 0 or time > MAX_WAIT_FRAME:
                continue
            
            if potential_match((track1, enter1, enter_taylor1, escape1, escape_taylor1), 
                               (track2, enter2, enter_taylor2, escape2, escape_taylor2)):
                graph[index1][index2] = 1
                
    print(graph)
    print(np.where(graph > 0))
    mm = MaximumMatching(graph)
    result, matchR = mm.maxBPM()
    trace = np.full(shape=(len(track_history),), fill_value=-1, dtype=int)
    for index, match in enumerate(np.asarray(matchR)):
        if match != -1:
            trace[match] = index
    # print(matchR)
    print(trace)
    visited = np.zeros(shape=(len(track_history),), dtype=np.uint8)
    for index, tr in enumerate(trace):
        if visited[index]:
            continue
        track = []
        enter = track_enter[index]
        enter_taylor = track_enter_taylor[index]
        escape = []
        escape_taylor = []
        print('tracing maximum matching:', index)
        while index != -1:
            print(track_history[index][-1][3], end=' ')
            visited[index] = True
            track += track_history[index]
            escape = track_escape[index]
            escape_taylor = track_escape_taylor[index]
            index = trace[index]
        print('')
        new_history.append(track)
        new_enter.append(enter)
        new_enter_taylor.append(enter_taylor)
        new_escape.append(escape)
        new_escape_taylor.append(escape_taylor)
        
    return new_history, new_enter, new_enter_taylor, new_escape, new_escape_taylor