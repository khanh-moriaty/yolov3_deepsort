import numpy as np
from shapely.geometry import Point, Polygon
import math

# Tính khoảng cách từ 1 điểm tới 1 đường thẳng
def calc_dist_point2line(pts, line_head, line_tail):
    pts = np.asarray(pts)
    line_head = np.asarray(line_head)
    line_tail = np.asarray(line_tail)
    return np.abs(np.cross(line_tail - line_head, line_head - pts) / np.linalg.norm(line_tail - line_head))

def calc_dist_point2point(p1, p2):
    p1 = Point(p1)
    p2 = Point(p2)
    return p1.distance(p2)

# Sắp xếp các head moi theo độ dài tới head
# Sắp xếp các tail moi theo độ dài tới tail
def find_moi_nearest(head, tail, mois_head, mois_tail, mois):

    # mois_head = config['mois_head']
    # mois_tail = config['mois_tail']
    
    head_list, tail_list = [], []
    
    for moi in mois:
        for [index_head, index_tail] in moi:
            moi_head = mois_head[index_head]
            moi_tail = mois_tail[index_tail]
            if moi_head is None or moi_tail is None: continue
            # dist = calc_dist_point2line(head, moi_head, moi_tail)
            dist = calc_dist_point2point(head, moi_head)
            head_list.append((index_head, dist))
            # dist = calc_dist_point2line(tail, moi_head, moi_tail)
            dist = calc_dist_point2point(tail, moi_tail)
            tail_list.append((index_tail, dist))
    
    # for index, moi_head in enumerate(mois_head):
    #     if moi_head is None: continue
    #     dist = Point(tuple(head)).distance(Point(tuple(moi_head)))
    #     head_list.append((index, dist))
    # for index, moi_tail in enumerate(mois_tail):
    #     if moi_tail is None: continue
    #     dist = Point(tuple(tail)).distance(Point(tuple(moi_tail)))
    #     tail_list.append((index, dist))
        
    head_list.sort(key=lambda a: a[1])
    print(head_list)
    head_list = [x[0] for x in head_list]    
    tail_list.sort(key=lambda a: a[1])
    print(tail_list)
    tail_list = [x[0] for x in tail_list]
    
    return head_list, tail_list, tail
    
def calc_angle_vectors(u, v):
    if np.linalg.norm(u) == 0 or np.linalg.norm(v) == 0: return 0
    dot_product = (u[0]*v[0] + u[1]*v[1]) / (np.linalg.norm(u) * np.linalg.norm(v))
    if dot_product > 1: dot_product=1
    if dot_product < -1: dot_product=-1
    return math.acos(dot_product)
    
# Loại bỏ những mois_head và mois_tail tạo góc quá lớn so với đường đi của phương tiện
def find_moi_cosine(head, tail, mois_head, mois_tail, mois,
                    ANGLE_THRESHOLD = math.pi / 4):
    
    mois_head = mois_head[:]
    mois_tail = mois_tail[:]
    check_mois_head = [False for _ in mois_head]
    check_mois_tail = [False for _ in mois_tail]
    
    movement_vector = tail - head
    # for index_head, moi_head in enumerate(mois_head):
    #     for index_tail, moi_tail in enumerate(mois_tail):
            
    for moi in mois:
        for [index_head, index_tail] in moi:
            moi_head = mois_head[index_head]
            moi_tail = mois_tail[index_tail]
            moi_vector = moi_tail - moi_head
            # print('cosine',movement_vector, moi_vector, calc_angle_vectors(moi_vector, movement_vector))
            if calc_angle_vectors(moi_vector, movement_vector) <= ANGLE_THRESHOLD:
                check_mois_head[index_head] = True
                check_mois_tail[index_tail] = True
                
    # for index_head, moi_head in enumerate(mois_head):
    #     if not check_mois_head[index_head]: mois_head[index_head] = None
    # for index_tail, moi_tail in enumerate(mois_tail):
    #     if not check_mois_tail[index_tail]: mois_tail[index_tail] = None
    
    mois_head = [moi_head if check_mois_head[index_head] else None for index_head, moi_head in enumerate(mois_head)]
    mois_tail = [moi_tail if check_mois_tail[index_tail] else None for index_tail, moi_tail in enumerate(mois_tail)]
    
    print(mois_head)
    print(mois_tail)
                
    return mois_head, mois_tail
    
def find_moi(head, tail, config):
    head, tail = np.array(head), np.array(tail)
    mois_head = config['mois_head']
    mois_tail = config['mois_tail']
    mois = config['mois']
    
    mois_head, mois_tail = find_moi_cosine(head, tail, mois_head, mois_tail, mois)
    index_head, index_tail, tail = find_moi_nearest(head, tail, mois_head, mois_tail, mois)
    # Chỉ quan tâm đến head và tail gần nhất
    if len(index_head) == 0 or len(index_tail) == 0:
        return -1, -1, -1
    return index_head[0], index_tail[0], tail
    
def confirm_moi_check(index_head, index_tail, moi):
    chk_head = False
    chk_tail = False
    for arr in moi:
        if arr[0] == index_head:
            chk_head = True
        if arr[1] == index_tail:
            chk_tail = True
    return chk_head and chk_tail
    
def confirm_moi(index_head, index_tail, center, config,
                MAX_CONFIRM_DISTANCE=15):
    
    print("index_head =", index_head, "index_tail =", index_tail, end=' ')
    mois = config['mois']
    
    center = Point(center)
    for index, moi in enumerate(mois):
        # Tìm xem head và tail tìm được có phải MOI đang xét hay không.
        if confirm_moi_check(index_head, index_tail, moi):
            # Confirmation process: Kiểm tra tail có nằm trong check_poly hay không.
            if config['check_poly'][index_tail].distance(center) <= MAX_CONFIRM_DISTANCE:
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