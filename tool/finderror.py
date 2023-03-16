#!/usr/bin/env python
import os.path
import cv2
from multiprocessing.dummy import Pool as ThreadPool


def process(filename):
    filename = filename.split('.')[0]
    filepath = os.path.join(imgpath, filename)
    imglist = os.listdir(filepath)
    imglist = sorted(imglist, key=lambda x: int(x.split('.')[0]))
    firstimgpath = os.path.join(filepath, '00001.jpg')
    firstimg = cv2.imread(firstimgpath)
    shap = firstimg.shape
    print(shap)
    for imgname in imglist:
        imgpath = os.path.join(filepath, imgname)
        img = cv2.imread(imgpath)
        if img.shape != shap:
            print(filename, imgname)


imgpath = "./data/Aff-Wild2/Faces"
VAfile = "./data/Aff-Wild2/5th_ABAW_Annotations/VA_Estimation_Challenge/VA_txt"
AUfile = "./data/Aff-Wild2/5th_ABAW_Annotations/AU_Detection_Challenge/AU_txt"
EXPRfile = "./data/Aff-Wild2/5th_ABAW_Annotations/EXPR_Classification_Challenge/EXPR_txt"

VA_list = os.listdir(VAfile)
AU_list = os.listdir(AUfile)
EXPR_list = os.listdir(EXPRfile)
handle = []
filenames = os.listdir(imgpath)
for filename in filenames:
    filename = filename + '.txt'
    if filename in VA_list or filename in AU_list or filename in EXPR_list:
        handle.append(filename)
print(len(handle))

pool = ThreadPool()
pool.map(process, handle)
pool.close()
pool.join()
