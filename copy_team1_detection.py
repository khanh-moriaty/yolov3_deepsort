import os
import shutil
from multiprocessing import Pool
from itertools import repeat

INP_PATH = '/dataset/Students/Team1/25_video/extract_frames/'
# INP_PATH = '/storage/detection_result/test_set_a/sub1/'
OUT_PATH = '/storage/detection_result/test_set_a/sub2/'

INP_DIRS = [
            # 'cam_01/det_txt/',
            # 'cam_04/det_txt/',
            # 'cam_09/det_txt/',
            # 'cam_10/det_txt/',
            'cam_12/det_txt/',
            'cam_14/det_txt/',
            'cam_16/det_txt/',
            'cam_18/det_txt/',
            'cam_24/det_txt/',
]

OUT_DIRS = [
            # 'cam_01/',
            # 'cam_04/',
            # 'cam_09/',
            # 'cam_10/',
            'cam_12/',
            'cam_14/',
            'cam_16/',
            'cam_18/',
            'cam_24/',
]

# INP_DIRS = [
#             'cam_02',
#             'cam_03',
#             'cam_05',
#             'cam_06',
#             'cam_07',
#             'cam_08',
#             'cam_11',
#             'cam_13',
#             'cam_15',
#             'cam_17',
#             'cam_19',
#             'cam_20',
#             'cam_21',
#             'cam_22',
#             'cam_23',
#             'cam_25',
# ]

# OUT_DIRS = [
#             'cam_02',
#             'cam_03',
#             'cam_05',
#             'cam_06',
#             'cam_07',
#             'cam_08',
#             'cam_11',
#             'cam_13',
#             'cam_15',
#             'cam_17',
#             'cam_19',
#             'cam_20',
#             'cam_21',
#             'cam_22',
#             'cam_23',
#             'cam_25',
# ]

IMG_WIDTH = 1280
IMG_HEIGHT = 720

def copy_file(fn, inp_dir, out_dir):
    print(fn)
    fi = os.path.join(inp_dir, fn)
    fi = open(fi, 'r')
    lines = fi.read().splitlines()
    
    out_fn = os.path.splitext(fn)[0].split('_')
    out_fn = '{}_{}_{:05d}.txt'.format(out_fn[0], out_fn[1], int(out_fn[2])+1)
    fo = os.path.join(out_dir, out_fn)
    
    # shutil.copyfile(os.path.join(inp_dir, fn), fo)
    # return
    
    fo = open(fo, 'w')
    
    for line in lines:
        content = line.split()
        content[-1] = int(content[-1]) + 1
        if content[-1] > 4: continue
        content[0] = int(content[0]) / IMG_WIDTH
        content[1] = int(content[1]) / IMG_HEIGHT
        content[2] = int(content[2]) / IMG_WIDTH + content[0]
        content[3] = int(content[3]) / IMG_HEIGHT + content[1]
        content = [str(x) for x in content]
        res = [content[-1]]
        [res.append(x) for x in content[:4]]
        res.append(content[4])
        res = ' '.join(res)
        fo.write(res + '\n')
        
    fo.close()
    
def copy_dir(inp_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    dir = os.listdir(inp_dir)
    pool = Pool(500)
    pool.starmap(copy_file, zip(dir, repeat(inp_dir), repeat(out_dir)))
    

for inp_dir, out_dir in zip(INP_DIRS, OUT_DIRS):
    copy_dir(os.path.join(INP_PATH, inp_dir),
             os.path.join(OUT_PATH, out_dir))