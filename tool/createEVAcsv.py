#!/usr/bin/env python
import os.path
import csv


def cal_sequence(num, labels):
    leftlen = 0
    rightlen = 0
    a = num-1
    b = num+1
    while a >= 1:
        if '-1' in labels[a] or '-5' in labels[a]:
            break
        else:
            leftlen += 1
            a -= 1
    while b <= len(labels) - 1:
        if '-1' in labels[b] or '-5' in labels[b]:
            break
        else:
            rightlen += 1
            b += 1
    min_len = min(leftlen, rightlen) * 2 + 1
    return leftlen, rightlen, min_len


EXPRlabel_folder = "../data/Aff-Wild2/5th_ABAW_Annotations/EXPR_Classification_Challenge/EXPR_csv"
VAlabel_folder = "../data/Aff-Wild2/5th_ABAW_Annotations/VA_Estimation_Challenge/VA_csv"
EVAlabel_folder = "../data/Aff-Wild2/5th_ABAW_Annotations/EVA/EVA_csv"

EXPRfilenames = os.listdir(EXPRlabel_folder)
VAfilenames = os.listdir(VAlabel_folder)
n = 0
for filename in EXPRfilenames:
    if filename in VAfilenames:
        EXPRcsvpath = os.path.join(EXPRlabel_folder, filename)
        VAcsvpath = os.path.join(VAlabel_folder, filename)
        # txt to list
        labelarray = []
        f = open(EXPRcsvpath, 'r', encoding='UTF-8')
        g = open(VAcsvpath, 'r', encoding='UTF-8')
        for line1, line2 in zip(f.readlines(), g.readlines()):
            labelarray.append([line1.split(',')[0], line1.split(',')[1], line2.split(',')[1], line2.split(',')[2]])
        f.close()
        g.close()
        # creat new label array
        newlabelpath = os.path.join(EVAlabel_folder, filename)
        newcsvfile = open(newlabelpath, 'w', newline='', encoding='gbk')
        writer = csv.writer(newcsvfile)
        writer.writerow(['frame', 'EXPR', 'valence', 'arousal', 'useful', 'left seq', 'right seq', 'min seq'])
        for num in range(1, len(labelarray)):
            label = labelarray[num]
            framename = label[0]
            EXPR = label[1]
            V = label[2]
            A = label[3]
            if '-5' in label or '-1' in label:
                left_seq, right_seq, min_seq = 0, 0, 0
                writer.writerow([framename, EXPR, V, A, -1, left_seq, right_seq, min_seq])
            else:
                left_seq, right_seq, min_seq = cal_sequence(num, labelarray)
                writer.writerow([framename, EXPR, V, A, 1, left_seq, right_seq, min_seq])
        newcsvfile.close()
        n += 1
        print(n, filename)