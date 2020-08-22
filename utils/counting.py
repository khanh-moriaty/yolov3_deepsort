import numpy as np
import sys
from shapely.geometry import Point

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
    
    # for index, moi_head in enumerate(mois_head):
    #     dist = Point(tuple(head)).distance(Point(tuple(mois_head[index_head])))
    #     dist_new = Point(tuple(head)).distance(Point(tuple(moi_head)))
    #     if dist_new < dist: index_head = index
    # for index, moi_tail in enumerate(mois_tail):
    #     dist = Point(tuple(tail)).distance(Point(tuple(mois_tail[index_tail])))
    #     dist_new = Point(tuple(tail)).distance(Point(tuple(moi_tail)))
    #     if dist_new < dist: index_tail = index
    
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
    
def count(track_history, frame_count, SUBMISSION_FILE, VIDEO_NAME, config,
          MINIMUM_DISTANCE=25):

    file = open(SUBMISSION_FILE, "w")
    track_history = [list(x) for x in track_history]

    log_file = open(SUBMISSION_FILE[:-4] + "_log.txt", 'w')
    stdout = sys.stdout
    sys.stdout = log_file
    
    for track in track_history:
        if (len(track) < 5):
            continue
        print("track_id: ", track[-1][3])
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
        
        moi_vector = np.array(track[-1][0]) - np.array(track[-2][0])
        center = (track[-1][0] + moi_vector * config['mois_shift'][moi] * 1.25).astype(int)
        
        kq = VIDEO_NAME + " " + str(frame_id) + " " + str(moi + 1) + " " \
            + str(track[-1][2]) + " " + str(center[0]) + " " + str(center[1])
        
        print(kq)
        file.write("".join(kq))
        file.write("\n")
            
    file.flush()
    file.close()
    sys.stdout = stdout
    log_file.flush()
    log_file.close()