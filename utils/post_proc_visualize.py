import cv2
import numpy as np

def visualize(track_history, VIDEO_PATH, OUTPUT_PATH):
    vid = cv2.VideoCapture(VIDEO_PATH)
    vid_fps = vid.get(cv2.CAP_PROP_FPS)
    codec = cv2.VideoWriter_fourcc(*'mp4v')
    vid_fps =int(vid.get(cv2.CAP_PROP_FPS))
    vid_width,vid_height = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(OUTPUT_PATH, codec, vid_fps, (vid_width, vid_height))
    
    frame_id = 0
    while True:
        _, img = vid.read()
        if img is None:
            print('Completed')
            break
        frame_id += 1
        
        for track in track_history:
            if 0 <= frame_id - track[-1][1] <= 3:
                pts = np.asarray([x[0] for x in track])
                # print(pts)
                cv2.polylines(img, [pts], isClosed=False, color=(255,255,255), thickness=2)
        
        out.write(img)
        
    vid.release()
    out.release()