#!/usr/bin/env python
import os
import time
import cv2
import numpy as np
from multiprocessing.dummy import Pool as ThreadPool


def downsize(array):
    e = array[1::2, ::2]
    return e


# 1.定义操作
def getfra_store(path_video, path_store):
    v_cap = cv2.VideoCapture(path_video)
    v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # frame_list = np.linspace(0, v_len-1, n_frames, dtype=np.int16)
    # f_b = int((v_len*3)/20)
    # f_f = int((v_len*19 + 19)/20)
    # f_use = int((v_len*16 + 19)/20)
    # n_f = int(((v_len*16 + 19)/20)/n_frames)
    for fn in range(v_len):
        # if fn < f_b:
        # success, trash = v_cap.read()
        # if success is False:
        # 	continue
        # elif f_b <= fn <= (f_b+n_frames*n_f):
        success, frame = v_cap.read()
        if success is False:
            continue
        # 	if (fn-f_b)%n_f == 0:
        # 		avg_frame = frame/n_f
        # 	else:
        # 		avg_frame = avg_frame + frame/n_f
        # 	if (fn+1-f_b)%n_f == 0:
        # if fn in frame_list:
        # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if frame.shape[0] + frame.shape[1] - 2000 > 0:  # - 1280 - 720
            frame = downsize(frame)
        # if frame.shape[0] + frame.shape[1] - 2000 <= 0:  # - 1280 - 720
        #     print('skip:', path_video)
        #     break
        cv2.imwrite(os.path.join(path_store, str(fn+1).zfill(5) + ".jpg"), frame)
        #print(path2img)
    # if fn > (f_b+n_frames*n_f):
    # 	print(fn)
    v_cap.release()
    return v_len


# 2.循环读取视频，获取帧，并将它们存储为jpg文件:
video_folder = "../data/Aff-Wild2/video_cleaned"
frame_folder = "../data/Aff-Wild2/Frames"
n = 1
t0 = time.time()
# for filename in os.listdir(video_folder)
#     # if i <= 0:
#     #    print(i, filename)
#     #    i += 1
#     #    continue
#     # filename = '6-30-1920x1080_left.mp4'
#     name = filename.split('.')[0]
#     # establish store address
#     path2store = os.path.join(jpg_folder, name)
#     os.makedirs(path2store, exist_ok=True)
#     # save
#     path2vid = os.path.join(video_folder, filename)
#     vlen = getfra_store(path2vid, path2store)
#     # cal speed
#     speed = (time.time() - t0)/i
#     print(i, path2store, vlen, speed)
#     n = n + 1


def process(name):
    filename = name.split('.')[0]
    # establish store address
    path2store = os.path.join(frame_folder, filename)
    os.makedirs(path2store, exist_ok=True)
    # save
    path2vid = os.path.join(video_folder, name)
    vlen = getfra_store(path2vid, path2store)
    # cal speed
    print(path2store, vlen)


# VAfile = "../data/Aff-Wild2/5th_ABAW_Annotations/VA_Estimation_Challenge/VA_csv"
# AUfile = "../data/Aff-Wild2/5th_ABAW_Annotations/AU_Detection_Challenge/AU_csv"
# EXPRfile = "../data/Aff-Wild2/5th_ABAW_Annotations/EXPR_Classification_Challenge/EXPR_csv"
# VA_list = os.listdir(VAfile)
# AU_list = os.listdir(AUfile)
# EXPR_list = os.listdir(EXPRfile)
txtpath = '../data/Aff-Wild2/test/names_of_videos_in_each_test_set/Expression_Classification_Challenge_test_set_release.txt'
EXPR_list = []
with open(txtpath, 'r', encoding='UTF-8') as f:
    for line in f.readlines():
        l = line.split('\n')[0]
        EXPR_list.append(l)
handle = []
video_names = os.listdir(video_folder)
for video_name in video_names:
    filename = video_name.split('.')[0]
    if filename in EXPR_list:
        handle.append(video_name)
print(len(handle))

# 3.根据处理目录读取视频，获取特定帧，并将它们存储为jpg文件:
# pool = ThreadPool()
# pool.map(process, handle)
# pool.close()
# pool.join()
