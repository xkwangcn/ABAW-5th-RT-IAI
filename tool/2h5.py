#!/usr/bin/env python
import os
import time
import pandas
import numpy as np
import cv2
import h5py
import pandas as pd

# -*- coding: utf-8 -*-
# @Time    : 2019/6/23 11:40
# @Author  : YYLin
# @Email   : 854280599@qq.com
# @File    : save_and_load_h5py.py
import os
import pickle
import time
import pandas
import numpy as np
import cv2
import h5py
import pandas as pd
from multiprocessing.dummy import Pool as ThreadPool


# save_image_to_h5py(r'E:\BaiduNetdiskDownload\AffectNet\Manually_Annotated\tt', r"E:\output")

# img_np = np.array(img_list)
# label_np = np.array(label_list)
# print('数据集中原始的标签顺序是:\n', label_np)
# img_pkl = pickle.dumps(img_np)


# f['labels'] = label_np

def save_image_to_h5py_1(path, output, handlelist):
    img_list = []
    # label_list = []
    dir_counter = 0
    num_for_test = 0
    # strattime=time.time()
    strattime = time.time()

    for child_dir in handlelist:
        # print(child_dir):104 105 1000 1031...

        # new_dir = os.path.join(output,child_dir)
        # os.makedirs(new_dir,exist_ok=True)
        # os.makedirs()

        child_path = os.path.join(path, child_dir)
        print('文件中的子文件名是:\n', child_path)
        if os.path.exists(os.path.join(output, f'{child_dir}.h5')):
            continue
        f = h5py.File(os.path.join(output, f'{child_dir}.h5'), 'w')  # 创建一个文件夹
        # 总共有9个文件夹 第一个文件夹加载10文件 其他文件夹中加载1个文件
        for dir_image in os.listdir(child_path):
            # print('dir_image中图像的名称是:\n', dir_image):0efbeiqoeuwo2g45v.jpg
            if not dir_image.endswith(".jpg"):
                continue
            img = cv2.imread(os.path.join(child_path, dir_image))
            # img =cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)#单通道，分辨率会下降
            img_np = np.array(img)
            # f.create_dataset("image", img_np)
            # label_list.append(dir_counter)
            f[dir_image] = img_np
        f.close()
        dir_counter += 1

    endtime = time.time() - strattime
    print("cost time:", endtime)


def file2h5(csv_path, output_file):
    train = pd.read_csv(csv_path,
                        header=None).to_numpy()
    # print(type(train))--><class 'numpy.ndarray'>
    with h5py.File(output_file, 'w') as f:
        for i in train[1:]:
            try:
                img = cv2.imread(os.path.join(r"E:/Aff2/Faces/", i[0]))
            except:
                print("img error----", i[0])
                continue
            # if f.get(i[0]):
            #     continue
            f[i[0]] = img
            print(i[0])


# 'key = 135-24-1920x1080_right/01998.jpg'
def load_h5py_to_np(path, key):
    h5_path = os.path.join(path, key.split("/")[0] + ".h5")  # E:/output3/135-24-1920x1080_right.h5
    # E:/OUTPUT2/105.h5
    key_img = key.split("/")[1]
    with h5py.File(h5_path, 'r') as hf:
        data = np.array(hf.get(key_img))
    return data


def process(child_dir):
    # print(child_dir):104 105 1000 1031...
    # new_dir = os.path.join(output,child_dir)
    # os.makedirs(new_dir,exist_ok=True)
    # os.makedirs()

    child_path = os.path.join(imgpath, child_dir)
    # print('文件中的子文件名是:\n', child_path)
    f = h5py.File(os.path.join(output, f'{child_dir}.h5'), 'w')  # 创建一个文件夹
    # 总共有9个文件夹 第一个文件夹加载10文件 其他文件夹中加载1个文件
    print(child_path, len(os.listdir(child_path)))
    for dir_image in os.listdir(child_path):
        # print('dir_image中图像的名称是:\n', dir_image):0efbeiqoeuwo2g45v.jpg
        if not dir_image.endswith(".jpg"):
            continue
        img = cv2.imread(os.path.join(child_path, dir_image))
        # img =cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)#单通道，分辨率会下降
        img_np = np.array(img)
        # f.create_dataset("image", img_np)
        # label_list.append(dir_counter)
        f[dir_image] = img_np
    f.close()
    print('down:', child_path)


if __name__ == '__main__':
    # 挑出需要转的文件夹
    imgpath = "../data/Aff-Wild2/Faces/"
    VAfile = "../data/Aff-Wild2/5th_ABAW_Annotations/VA_Estimation_Challenge/VA_csv/"
    AUfile = "../data/Aff-Wild2/5th_ABAW_Annotations/AU_Detection_Challenge/AU_csv/"
    EXPRfile = "../data/Aff-Wild2/5th_ABAW_Annotations/EXPR_Classification_Challenge/EXPR_csv/"
    output = "../data/Aff-Wild2/h5file/"
    os.makedirs(output, exist_ok=True)
    VA_list = os.listdir(VAfile)
    AU_list = os.listdir(AUfile)
    EXPR_list = os.listdir(EXPRfile)
    handle = []
    filenames = os.listdir(imgpath)
    for filename in filenames:
        filename = filename + '.csv'
        if filename in VA_list or filename in AU_list or filename in EXPR_list:
            handle.append(filename.split('.')[0])
    print(len(handle))

    # save_image_to_h5py_1(imgpath, output, handle)
    # data = load_h5py_to_np("E:/output3/",'1-30-1280x720.h5')
    # f = h5py.File('E:/output3/1-30-1280x720.h5', 'r')
    # print([key for key in f.keys()])

    # 多CPU并行处理
    pool = ThreadPool()
    pool.map(process, handle)
    pool.close()
    pool.join()

# file2h5(r"E:\Aff2\train.csv",os.path.join(r"E:\output3",f'train.h5'))
# file2h5(r"E:\BaiduNetdiskDownload\AffectNet\Manually_Annotated_file_lists/validation.csv",os.path.join(r"E:\output2",f'validation.h5'))
