from shapely.geometry import Polygon
import numpy as np

# Định dạng của file config:
################################################################################
# Dòng đầu tiên chứa 3 số n, m, k lần lượt là
# <n: số lượng head> <m: số lượng tail/check region> <k: số lượng MOI theo BTC>
################################################################################
# N dòng tiếp theo, mỗi dòng chứa hai số nguyên là tọa độ của các head
# <a[0]> <b[0]>
# <a[1]> <b[1]>
# ...
# <a[N-1]> <b[N-1]>
################################################################################
# M dòng tiếp theo, mỗi dòng chứa hai số nguyên là tọa độ của các tail
# <c[0]> <d[0]>
# <c[1]> <d[1]>
# ...
# <c[M-1]> <d[M-1]>
################################################################################
# K dòng tiếp theo, dòng thứ i chứa các cặp số nguyên <x[i]> <y[i]>
# với ý nghĩa head[x[i]] nối với tail[x[i]], đồng thời cũng là
# MOI thứ i theo quy ước của BTC.
# Số cuối cùng trên mỗi dòng là số frame "trừ hao" của mỗi MOI.
# Tức là khi đếm object thì frame_id sẽ được cộng thêm giá trị "trừ hao" này.
# Lưu ý: có thể có nhiều cặp head/tail tương ứng với cùng 1 MOI của BTC.
# <x[0]> <y[0]> <x[0]> <y[0]> <z[0]>
# <x[1]> <y[1]> <z[1]>
# <x[2]> <y[2]> <x[2]> <y[2]> <x[2]> <y[2]> <z[2]>
# ...
# <x[K]> <y[K]> <z[K]>
################################################################################
# M dòng tiếp theo, dòng thứ i chứa các cặp số nguyên là tọa độ 
# các đỉnh của check region cho MOI thứ i của BTC.
# Lưu ý: có thể có nhiều hoặc ít hơn 4 cặp đỉnh.
# <x[0]> <y[0]> <x[0]> <y[0]> <x[0]> <y[0]> <x[0]> <y[0]> <x[0]> <y[0]>
# <x[1]> <y[1]> <x[1]> <y[1]> <x[1]> <y[1]>
# <x[2]> <y[2]> <x[2]> <y[2]> <x[2]> <y[2]> <x[2]> <y[2]>
# ...
# <x[M]> <y[M]> <x[M]> <y[M]> <x[M]> <y[M]> <x[M]> <y[M]> <x[M]> <y[M]>
################################################################################
# Dòng cuối cùng của config chứa các cặp số nguyên là tọa độ 
# các đỉnh của ROI.
# Lưu ý: có thể có nhiều hoặc ít hơn 4 cặp đỉnh.
# <x[0]> <y[0]> <x[1]> <y[1]> <x[2]> <y[2]> <x[3]> <y[3]>
################################################################################
def load_config(config_path):
    config_file = open(config_path, 'r')
    config = {}
    
    # Đọc từng dòng của file config và xóa đi những dòng trống
    lines = config_file.read().splitlines()
    lines = [x.strip() for x in lines]
    lines = [x for x in lines if x]
    
    print(lines)
    
    n, m, k = [int(x) for x in lines[0].split()[:3]]
    config['n'] = n
    config['m'] = m
    config['k'] = k
    lines = lines[1:]
    
    config['mois_head'] = []
    for i in range(n):
        coord = np.array([int(x) for x in lines[i].split()[:2]])
        config['mois_head'].append(coord)
    lines = lines[n:]
    
    config['mois_tail'] = []
    for i in range(m):
        coord = np.array([int(x) for x in lines[i].split()[:2]])
        config['mois_tail'].append(coord)
    lines = lines[m:]
    
    config['mois'] = [[] for _ in range(k+1)]
    config['mois_shift'] = [0 for _ in range(k+1)]
    for i in range(k+1):
        id_list = [int(x) for x in lines[i].split()]
        for index in range(0, len(id_list)-1, 2):
            config['mois'][i].append([id_list[index], id_list[index+1]])
        config['mois_shift'][i] = id_list[-1]
    lines = lines[k+1:]
    
    config['check_regions'] = [[] for _ in range(m)]
    config['check_poly'] = [None for _ in range(m)]
    for i in range(m):
        coord_list = [int(x) for x in lines[i].split()]
        for index in range(0, len(coord_list), 2):
            config['check_regions'][i].append([coord_list[index], coord_list[index+1]])
        config['check_poly'][i] = Polygon(config['check_regions'][i])
    lines = lines[m:]
    
    config['roi'] = []
    coord_list = [int(x) for x in lines[0].split()]
    for index in range(0, len(coord_list), 2):
        config['roi'].append([coord_list[index], coord_list[index+1]])
    config['roi_poly'] = Polygon(config['roi'])
    
    config['roi_btc'] = []
    coord_list = [int(x) for x in lines[1].split()]
    for index in range(0, len(coord_list), 2):
        config['roi_btc'].append([coord_list[index], coord_list[index+1]])
    config['roi_btc_poly'] = Polygon(config['roi_btc'])
    
    return config