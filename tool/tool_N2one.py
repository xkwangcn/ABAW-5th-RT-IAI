import os.path
import random
import pandas as pd
import csv


labelpath = '../../DATA/Aff-Wild2/5th_ABAW_Annotations/EXPR_Classification_Challenge/EXPR_csv'
storepath = '../../DATA/Aff-Wild2/5th_ABAW_Annotations/EXPR_Classification_Challenge'
filenames = os.listdir(labelpath)
random.shuffle(filenames)
csvfile = open(os.path.join(storepath, 'EXPR_all.csv'), 'w', newline='', encoding='gbk')
writer = csv.writer(csvfile)
writer.writerow(['filename', 'frame',
                 'EXPR',
                 'left seq', 'right seq', 'min seq'])

i = 0
for filename in filenames:
    csvpath = os.path.join(labelpath, filename)
    filename = filename.split('.')[0]
    with open(csvpath, 'r', encoding='UTF-8') as f:
        for line in f.readlines():
            if 'frame' in line:
                continue
            writer.writerow([filename, line.split(',')[0],
                             line.split(',')[1],
                             line.split(',')[2], line.split(',')[3], line.split('\n')[0].split(',')[4]])
    i += 1
    print(i, filename)
