import os

def merge(SUBMISSION_DIR):
    dir = os.listdir(SUBMISSION_DIR)
    dir = [fi for fi in dir if os.path.splitext(fi)[1] == '.txt']
    dir = [fi for fi in dir if os.path.splitext(fi)[0].startswith('submission_cam_')]
    dir = [fi for fi in dir if not os.path.splitext(fi)[0].endswith('log')]
    dir.sort()
    
    sub_path = os.path.join(SUBMISSION_DIR, 'submission.txt')
    fo = open(sub_path, 'w')
    
    for fn in dir:
        inp_path = os.path.join(SUBMISSION_DIR, fn)
        fi = open(inp_path, 'r')
        lines = fi.read().splitlines()
        for line in lines:
            content = line.split()[:-2]
            content = ' '.join(content)
            fo.write(content + '\n')
        fi.close()
        
    fo.close()
            
if __name__ == '__main__':
    merge('data/video/')
