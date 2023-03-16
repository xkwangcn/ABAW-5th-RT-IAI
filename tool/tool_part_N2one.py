import os
import csv
import random
from shutil import copy

prop = 0.8
task = 'EXPR'
if task == 'AU':
    labelpath = '../data/Aff-Wild2/5th_ABAW_Annotations/AU_Detection_Challenge/AU_csv'
    storepath = '../data/Aff-Wild2/5th_ABAW_Annotations/AU_Detection_Challenge'
    traincsvfile = open(os.path.join(storepath, 'AU_train.csv'), 'w', newline='', encoding='gbk')
    valcsvfile = open(os.path.join(storepath, 'AU_valid.csv'), 'w', newline='', encoding='gbk')
    writer1 = csv.writer(traincsvfile)
    writer1.writerow(['filename', 'frame',
                      'AU1', 'AU2', 'AU3', 'AU4', 'AU5', 'AU6', 'AU7', 'AU8', 'AU9', 'AU10', 'AU11', 'AU12',
                      'left seq', 'right seq', 'min seq'])
    writer2 = csv.writer(valcsvfile)
    writer2.writerow(['filename', 'frame',
                      'AU1', 'AU2', 'AU3', 'AU4', 'AU5', 'AU6', 'AU7', 'AU8', 'AU9', 'AU10', 'AU11', 'AU12',
                      'left seq', 'right seq', 'min seq'])

    filenames = os.listdir(labelpath)
    random.shuffle(filenames)
    i = 0
    for filename in filenames:
        csvpath = os.path.join(labelpath, filename)
        filename = filename.split('.')[0]
        if i <= len(filenames) * prop:
            with open(csvpath, 'r', encoding='UTF-8') as f:
                for line in f.readlines():
                    if 'frame' in line:
                        continue
                    writer1.writerow([filename, line.split(',')[0],
                                      line.split(',')[1], line.split(',')[2], line.split(',')[3], line.split(',')[4],
                                      line.split(',')[5], line.split(',')[6], line.split(',')[7], line.split(',')[8],
                                      line.split(',')[9], line.split(',')[10], line.split(',')[11], line.split(',')[12],
                                      line.split(',')[13], line.split(',')[14], line.split('\n')[0].split(',')[15]])
        else:
            with open(csvpath, 'r', encoding='UTF-8') as f:
                for line in f.readlines():
                    if 'frame' in line:
                        continue
                    writer2.writerow([filename, line.split(',')[0],
                                      line.split(',')[1], line.split(',')[2], line.split(',')[3], line.split(',')[4],
                                      line.split(',')[5], line.split(',')[6], line.split(',')[7], line.split(',')[8],
                                      line.split(',')[9], line.split(',')[10], line.split(',')[11], line.split(',')[12],
                                      line.split(',')[13], line.split(',')[14], line.split('\n')[0].split(',')[15]])
        i += 1
        print(i, filename)
    traincsvfile.close()
    valcsvfile.close()

elif task == 'EXPR':
    labelpath = '../data/Aff-Wild2/5th_ABAW_Annotations/EXPR_Classification_Challenge/EXPR_csv'
    storepath = '../data/Aff-Wild2/5th_ABAW_Annotations/EXPR_Classification_Challenge'
    splitpath = os.path.join(storepath, 'official')
    os.makedirs(splitpath, exist_ok=True)
    train_dest_dir = os.path.join(splitpath, 'train_csv')
    val_dest_dir = os.path.join(splitpath, 'val_csv')
    os.makedirs(train_dest_dir, exist_ok=True)
    os.makedirs(val_dest_dir, exist_ok=True)

    traincsvfile = open(os.path.join(splitpath, 'EXPR_train.csv'), 'w', newline='', encoding='gbk')
    valcsvfile = open(os.path.join(splitpath, 'EXPR_valid.csv'), 'w', newline='', encoding='gbk')
    writer1 = csv.writer(traincsvfile)
    writer1.writerow(['filename', 'frame',
                      'EXPR',
                      'left seq', 'right seq', 'min seq'])
    writer2 = csv.writer(valcsvfile)
    writer2.writerow(['filename', 'frame',
                      'EXPR',
                      'left seq', 'right seq', 'min seq'])
    trainnames = os.listdir('../data/Aff-Wild2/5th_ABAW_Annotations_origin/EXPR_Classification_Challenge/Train_Set')
    validnames = os.listdir('../data/Aff-Wild2/5th_ABAW_Annotations_origin/EXPR_Classification_Challenge/Validation_Set')
    i = 0
    for filename in trainnames:
        filename = filename.split('.')[0]
        csvpath = os.path.join(labelpath, filename + '.csv')
        copy(csvpath, train_dest_dir)
        filename = filename.split('.')[0]
        with open(csvpath, 'r', encoding='UTF-8') as f:
            for line in f.readlines():
                if 'frame' in line:
                    continue
                writer1.writerow([filename, line.split(',')[0],
                                  line.split(',')[1],
                                  line.split(',')[2], line.split(',')[3], line.split('\n')[0].split(',')[4]])
        i += 1
        print(i, filename)
    for filename in validnames:
        filename = filename.split('.')[0]
        csvpath = os.path.join(labelpath, filename + '.csv')
        copy(csvpath, val_dest_dir)
        filename = filename.split('.')[0]
        with open(csvpath, 'r', encoding='UTF-8') as f:
            for line in f.readlines():
                if 'frame' in line:
                    continue
                writer2.writerow([filename, line.split(',')[0],
                                  line.split(',')[1],
                                  line.split(',')[2], line.split(',')[3], line.split('\n')[0].split(',')[4]])
        i += 1
        print(i, filename)
    traincsvfile.close()
    valcsvfile.close()
    # filenames = os.listdir(labelpath)
    # random.shuffle(filenames)
    # parts = [filenames[int(len(filenames)*0.2*i):int(len(filenames)*0.2*(i+1))] for i in range(5)]
    # for split in range(5):
    #     if split == 0:
    #         trainnames = parts[1] + parts[2] + parts[3] + parts[4]
    #     elif split == 1:
    #         trainnames = parts[0] + parts[2] + parts[3] + parts[4]
    #     elif split == 2:
    #         trainnames = parts[0] + parts[1] + parts[3] + parts[4]
    #     elif split == 3:
    #         trainnames = parts[0] + parts[1] + parts[2] + parts[4]
    #     elif split == 4:
    #         trainnames = parts[0] + parts[1] + parts[2] + parts[3]
    #     validnames = parts[split]
    #     splitpath = os.path.join(storepath, 'Split{}'.format(split))
    #     os.makedirs(splitpath, exist_ok=True)
    #     train_dest_dir = os.path.join(splitpath, 'Splited_train_csv')
    #     val_dest_dir = os.path.join(splitpath, 'Splited_val_csv')
    #     os.makedirs(train_dest_dir, exist_ok=True)
    #     os.makedirs(val_dest_dir, exist_ok=True)
    #
    #     traincsvfile = open(os.path.join(splitpath, 'EXPR_train.csv'), 'w', newline='', encoding='gbk')
    #     valcsvfile = open(os.path.join(splitpath, 'EXPR_valid.csv'), 'w', newline='', encoding='gbk')
    #     writer1 = csv.writer(traincsvfile)
    #     writer1.writerow(['filename', 'frame',
    #                       'EXPR',
    #                       'left seq', 'right seq', 'min seq'])
    #     writer2 = csv.writer(valcsvfile)
    #     writer2.writerow(['filename', 'frame',
    #                       'EXPR',
    #                       'left seq', 'right seq', 'min seq'])
    #
    #     i = 0
    #     for filename in trainnames:
    #         csvpath = os.path.join(labelpath, filename)
    #         copy(csvpath, train_dest_dir)
    #         filename = filename.split('.')[0]
    #         with open(csvpath, 'r', encoding='UTF-8') as f:
    #             for line in f.readlines():
    #                 if 'frame' in line:
    #                     continue
    #                 writer1.writerow([filename, line.split(',')[0],
    #                                   line.split(',')[1],
    #                                   line.split(',')[2], line.split(',')[3], line.split('\n')[0].split(',')[4]])
    #         i += 1
    #         print(i, filename)
    #     for filename in validnames:
    #         csvpath = os.path.join(labelpath, filename)
    #         copy(csvpath, val_dest_dir)
    #         filename = filename.split('.')[0]
    #         with open(csvpath, 'r', encoding='UTF-8') as f:
    #             for line in f.readlines():
    #                 if 'frame' in line:
    #                     continue
    #                 writer2.writerow([filename, line.split(',')[0],
    #                                   line.split(',')[1],
    #                                   line.split(',')[2], line.split(',')[3], line.split('\n')[0].split(',')[4]])
    #         i += 1
    #         print(i, filename)
    #     traincsvfile.close()
    #     valcsvfile.close()

elif task == 'VA':
    labelpath = '../data/Aff-Wild2/5th_ABAW_Annotations/VA_Estimation_Challenge/VA_csv'
    storepath = '../data/Aff-Wild2/5th_ABAW_Annotations/VA_Estimation_Challenge'
    traincsvfile = open(os.path.join(storepath, 'VA_train.csv'), 'w', newline='', encoding='gbk')
    valcsvfile = open(os.path.join(storepath, 'VA_valid.csv'), 'w', newline='', encoding='gbk')
    writer1 = csv.writer(traincsvfile)
    writer1.writerow(['filename', 'frame',
                      'valence', 'arousal',
                      'left seq', 'right seq', 'min seq'])
    writer2 = csv.writer(valcsvfile)
    writer2.writerow(['filename', 'frame',
                      'valence', 'arousal',
                      'left seq', 'right seq', 'min seq'])

    filenames = os.listdir(labelpath)
    random.shuffle(filenames)
    i = 0
    for filename in filenames:
        csvpath = os.path.join(labelpath, filename)
        filename = filename.split('.')[0]
        if i <= len(filenames) * prop:
            with open(csvpath, 'r', encoding='UTF-8') as f:
                for line in f.readlines():
                    if 'frame' in line:
                        continue
                    writer1.writerow([filename, line.split(',')[0],
                                      line.split(',')[1], line.split(',')[2],
                                      line.split(',')[3], line.split(',')[4], line.split('\n')[0].split(',')[5]])
        else:
            with open(csvpath, 'r', encoding='UTF-8') as f:
                for line in f.readlines():
                    if 'frame' in line:
                        continue
                    writer2.writerow([filename, line.split(',')[0],
                                      line.split(',')[1], line.split(',')[2],
                                      line.split(',')[3], line.split(',')[4], line.split('\n')[0].split(',')[5]])
        i += 1
        print(i, filename)
    traincsvfile.close()
    valcsvfile.close()

elif task == 'EVA':
    labelpath = '../data/Aff-Wild2/5th_ABAW_Annotations/EXPR_VA/EVA_csv'
    storepath = '../data/Aff-Wild2/5th_ABAW_Annotations/EXPR_VA'
    traincsvfile = open(os.path.join(storepath, 'EVA_train.csv'), 'w', newline='', encoding='gbk')
    valcsvfile = open(os.path.join(storepath, 'EVA_valid.csv'), 'w', newline='', encoding='gbk')
    writer1 = csv.writer(traincsvfile)
    writer1.writerow(['filename', 'frame',
                      'EXPR', 'valence', 'arousal',
                      'useful', 'left seq', 'right seq', 'min seq'])
    writer2 = csv.writer(valcsvfile)
    writer2.writerow(['filename', 'frame',
                      'EXPR', 'valence', 'arousal',
                      'useful', 'left seq', 'right seq', 'min seq'])

    filenames = os.listdir(labelpath)
    random.shuffle(filenames)
    i = 0
    for filename in filenames:
        csvpath = os.path.join(labelpath, filename)
        filename = filename.split('.')[0]
        if i <= len(filenames)*prop:
            with open(csvpath, 'r', encoding='UTF-8') as f:
                for line in f.readlines():
                    if 'frame' in line:
                        continue
                    writer1.writerow([filename, line.split(',')[0],
                                      line.split(',')[1], line.split(',')[2], line.split(',')[3],
                                      line.split(',')[4], line.split(',')[5], line.split(',')[6], line.split('\n')[0].split(',')[7]])
        else:
            with open(csvpath, 'r', encoding='UTF-8') as f:
                for line in f.readlines():
                    if 'frame' in line:
                        continue
                    writer2.writerow([filename, line.split(',')[0],
                                      line.split(',')[1], line.split(',')[2], line.split(',')[3],
                                      line.split(',')[4], line.split(',')[5], line.split(',')[6], line.split('\n')[0].split(',')[7]])
        i += 1
        print(i, filename)
    traincsvfile.close()
    valcsvfile.close()
