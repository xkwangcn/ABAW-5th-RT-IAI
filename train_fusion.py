#!/usr/bin/env python
import argparse
import os
import sys
import time

import numpy
import torch
import random
from datetime import datetime
from dateutil import tz
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader

import config
from train import train_one_epoch
from loss import CCCLoss
from eval import cal_ccc_pcc, cal_acc_f1, evaluate
from utils import Logger, seed_worker
from fusion_dataset import TrainDataset, ValDataset
from models.model_fusion import Model


def parse_args():
    parser = argparse.ArgumentParser(description='ABAW')
    parser.add_argument('--task', type=str, required=False, default='EXPR',
                        choices=['VA','Valence','Arousal',
                                 'EXPR','Neutral','Anger','Disgust', 'Fear','Happiness','Sadness','Surprise','Other',
                                 'AU','AU1','AU2','AU4','AU6','AU7','AU10','AU12','AU15','AU23','AU24','AU25','AU26',
                                 'ERI','Adoration','Amusement','Anxiety','Disgust','Empathic Pain','Fear','Surprise'],
                        help='Specify the Task.')
    parser.add_argument('--model', required=False, default='FANLSTM')
    parser.add_argument('--feature', required=False, default='Face', choices=['Video', 'Face', 'Gaze', 'Action', 'Audio'],
                        help='Specify the features used (only one).')
    parser.add_argument('--sequence', required=False, default=180,
                        help='Specify the frame len.')
    parser.add_argument('--data_path', required=False, default="/mnt/wd0/home_back/shutao/ABAW/data/Aff-Wild2/",
                        help='Specify the features used (only one).')
    parser.add_argument('--optimizer', required=False, default='AdamW',
                        help='Specify the optimizer')
    parser.add_argument('--linear_dropout', type=float, default=0)
    parser.add_argument('--epochs', type=int, default=50,
                        help='Specify the number of epochs (default: 100).')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Specify the batch size (default: 256).')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Specify initial learning rate (default: 0.0001).')
    parser.add_argument('--backbone_lr', type=float, default=0,
                        help='Specify initial learning rate (default: 0.00001).')
    parser.add_argument('--seed', type=int, default=101,
                        help='Specify the initial random seed (default: 101).')
    parser.add_argument('--n_seeds', type=int, default=1,
                        help='Specify number of random seeds to try (default: 5).')
    parser.add_argument('--result_csv', default=None, help='Append the results to this csv (or create it, if it '
                                                           'does not exist yet). Incompatible with --predict')
    parser.add_argument('--early_stopping_patience', type=int, default=10, help='Patience for early stopping')
    parser.add_argument('--reduce_lr_patience', type=int, default=2, help='Patience for reduction of learning rate')
    parser.add_argument('--use_gpu', action='store_true', default=True,
                        help='Specify whether to use gpu for training (default: False).')
    parser.add_argument('--cache', action='store_true',
                        help='Specify whether to cache data as pickle file (default: False).')
    parser.add_argument('--save_path', type=str, default='preds',
                        help='Specify path where to save the predictions (default: preds).')
    parser.add_argument('--eval', action='store_true', default=False,
                        help='Specify when want to eval one model ')
    parser.add_argument('--eval_model', type=str, default=None,
                        help='Specify model which is to be evaluated; no training with this option (default: False).')
    parser.add_argument('--test', action='store_true', default=False,
                        help='Specify when want to test')
    parser.add_argument('--predict', action='store_true', default=False,
                        help='Specify when no test labels are available; test predictions will be saved '
                             '(default: False). Incompatible with result_csv')
    parser.add_argument('--regularization', type=float, required=False, default=0.0,
                        help='L2-Penalty')
    parser.add_argument('--r', type=int, default=10,
                        help='Specify random seed r')
    args = parser.parse_args()
    if not (args.result_csv is None) and args.predict:
        print("--result_csv is not compatible with --predict")
        sys.exit(-1)
    return args


def get_loss_fn(device):  # LOSS
    if args.task == 'VA':
        return CCCLoss(), 'CCC Loss'
    elif args.task == 'EXPR':
        #weights = torch.Tensor([0.0182, 0.2056, 0.2935, 0.2721, 0.037, 0.045, 0.1113, 0.0172]).to(device)#all
        #weights = torch.Tensor([0.0186, 0.1774, 0.3409, 0.2590, 0.0384, 0.0452, 0.1035, 0.0170]).to(device)#Split0
        #weights = torch.Tensor([0.0202, 0.1971, 0.2690, 0.2440, 0.0409, 0.0435, 0.1629, 0.0224]).to(device)#Split1
        #weights = torch.Tensor([0.0183, 0.2010, 0.3495, 0.2410, 0.0398, 0.0366, 0.0976, 0.0163]).to(device)#Split2
        #weights = torch.Tensor([0.0163, 0.2194, 0.2508, 0.3285, 0.0316, 0.0443, 0.0947, 0.0144]).to(device)#Split3
        #weights = torch.Tensor([0.0162, 0.2277, 0.2558, 0.2923, 0.0319, 0.0569, 0.1038, 0.0154]).to(device)#Split4
        weights = torch.Tensor([0.0427, 0.2249, 0.2360, 0.2764, 0.0486, 0.0430, 0.1117, 0.0167]).to(device)
        return nn.CrossEntropyLoss(weight=weights), 'CE Loss'
    elif args.task == 'AU':
        return nn.CrossEntropyLoss(), 'CE Loss'


def get_eval_fn():
    if args.task == 'VA':
        return cal_ccc_pcc, "Vccc_Accc_Vpcc_Apcc"
    elif args.task == 'EXPR':
        return cal_acc_f1, 'ACC_F1'
    elif args.task == 'AU':
        return cal_acc_f1, 'F1'


def save_model(model, model_folder, current_seed):
    model_file_name = f'model_{current_seed}.pth'
    model_file = os.path.join(model_folder, model_file_name)
    torch.save(model, model_file)
    return model_file


def main(args):
    # ensure reproducibility
    random.seed(args.r)
    numpy.random.seed(args.r)

    # test_set = TestDataset(path=args.data_path, Task=args.task, Feature=args.feature, Sequence=args.sequence)
    # test_loader = DataLoader(dataset=test_set, num_workers=12, batch_size=args.batch_size,
    #                          shuffle=False, worker_init_fn=seed_worker)

    device = torch.device("cuda" if args.use_gpu else "cpu")
    print(device)
    loss_fn, loss_str = get_loss_fn(device)
    eval_fn, eval_str = get_eval_fn()

    if args.eval is False:  # Train and validate for each seed
        seeds = range(args.seed, args.seed + args.n_seeds)
        val_losses, val_scores, best_model_files, test_scores = [], [], [], []

        for seed in seeds:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

            # model = Model().to(device)
            model = Model() #wxk debug
            # print(model)

            # 确定优化器
            backbone_params = list(map(id, model.faceencoder.parameters()))
            downstream_params = filter(lambda p: id(p) not in backbone_params, model.parameters())
            params = [{'params': model.faceencoder.parameters(), 'lr': args.backbone_lr},
                      {'params': downstream_params, 'lr': args.lr}]
            if args.optimizer == 'SGD':
                optimizer = optim.SGD(params, lr=args.lr, momentum=0.9, nesterov=True, weight_decay=0.05)
            elif args.optimizer == 'Adam':
                optimizer = optim.Adam(params, lr=args.lr, weight_decay=0.05)
            elif args.optimizer == 'AdamW':
                optimizer = optim.AdamW(params, lr=args.lr, betas=(0.9, 0.999), weight_decay=0.05)
            lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min',
                                                                patience=args.reduce_lr_patience,
                                                                factor=0.5, min_lr=1e-6, verbose=True)
            epochs = args.epochs
            best_val_loss = float('inf')
            best_epoch = 0
            best_val_score = -1
            best_model_file_path = ''
            early_stop = 0
            model_path = args.paths['model']
            current_seed = seed
            early_stopping_patience = args.early_stopping_patience
            print('=' * 50)
            print(f'Training model... [seed {seed}] for at most {args.epochs} epochs')
            t0 = time.time()
            for epoch in range(1, epochs + 1):
                # 训练步骤
                train_set = TrainDataset(path=args.data_path, Task=args.task, Feature=args.feature, Sequence=args.sequence)
                train_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=args.batch_size,
                                          shuffle=True, worker_init_fn=seed_worker, pin_memory=True)  # shuffle训练时打乱样本
                print(f'Training for Epoch {epoch}...')
                train_loss = 0
                #model, train_loss = train_one_epoch(args, model, device, train_loader, optimizer, loss_fn)
                t1 = time.time()
                print(t1 - t0)
                print(f'Epoch:{epoch:>3} / {epochs} | [Train] | Loss: {train_loss:>.4f}')

                # 测试步骤
                val_set = ValDataset(path=args.data_path, Task=args.task, Feature=args.feature, Sequence=args.sequence)
                val_loader = DataLoader(dataset=val_set, num_workers=4, batch_size=args.batch_size,
                                        shuffle=False, worker_init_fn=seed_worker, pin_memory=True)
                print('begin to evaluate')
                val_loss, val_score = 0, 0
                val_loss, acc, f1 = evaluate(args, model, device, val_loader, loss_fn, eval_fn)
                val_score = (0.33*acc+0.67*f1)/1
                t0 = time.time()
                print(t0 - t1)
                print(
                    f'Epoch:{epoch:>3}/{epochs}|[Val]|Loss:{val_loss:>.4f}|[{eval_str}]:{acc:7.4f};{f1:7.4f}')
                print('-' * 50)

                if val_score >= best_val_score:
                    early_stop = 0
                    best_epoch = epoch
                    best_val_loss = val_loss
                    best_val_score = val_score
                    best_model_file_path = save_model(model, model_path, current_seed)
                else:
                    early_stop += 1
                    if early_stop >= early_stopping_patience:
                        print(f'Note: target can not be optimized for {early_stopping_patience} consecutive epochs, '
                              f'early stop the training process!')
                        print('-' * 50)
                        break
                lr_scheduler.step(1 - numpy.mean(val_score))

            print(f'Seed {seed} | '
                  f'Best [Val {eval_str}]:{best_val_score:>7.4f} | Loss: {best_val_loss:>.4f} | epoch: {best_epoch}')
            val_losses.append(best_val_loss)
            val_scores.append(best_val_score)
            best_model_files.append(best_model_file_path)

        best_idx = val_scores.index(max(val_scores))  # find best performing seed
        model_file = best_model_files[best_idx]  # best model of all of the seeds
        print('=' * 50)
        print(f'Best {eval_str} on [Val] for seed {seeds[best_idx]}: '
              f'[Val {eval_str}]: {val_scores[best_idx]:7.4f}')
        print('=' * 50)

        # if args.test:
        #     model = torch.load(model_file)
        #     val_loss, val_score = evaluate(args, model, test_loader, loss_fn, eval_fn)

        # if args.predict:
        #     model = torch.load(model_file)
        #     pred = model(img, feat)
        #
        #     if not args.result_csv is None:
        #         log_results(args.result_csv, params=args, seeds=list(seeds), metric_name=eval_str,
        #                     model_files=best_model_files, test_results=test_scores, val_results=val_score,
        #                     best_idx=best_idx)


if __name__ == '__main__':
    args = parse_args()

    args.log_file_name = '{}_{}_{}_[{}_{}]'.format(datetime.now(tz=tz.gettz()).strftime("%Y%m%d-%H-%M"),
                                                   args.task.replace(os.path.sep, "-"),
                                                   args.model.replace(os.path.sep, "-"),
                                                   args.lr, args.backbone_lr)

    # adjust your paths in config.py
    args.paths = {'log': os.path.join(config.LOG_FOLDER, args.task),
                  'data': os.path.join(config.DATA_FOLDER, args.task),
                  'model': os.path.join(config.MODEL_FOLDER, args.task, args.log_file_name)}
    if args.predict:
        args.paths['predict'] = os.path.join(config.PREDICTION_FOLDER, args.task, args.log_file_name)
    for folder in args.paths.values():
        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)

    # add more path
    args.paths.update({'features': config.PATH_TO_FEATURES[args.task],
                       'labels': config.PATH_TO_LABELS[args.task]})

    # establish log.txt
    sys.stdout = Logger(os.path.join(args.paths['log'], args.log_file_name + '.txt'))
    print(' '.join(sys.argv))
    print('model:', args.model,
          ' ,optimizer:', args.optimizer,
          ' ,lr:', args.lr,
          ' ,backbone_lr:', args.backbone_lr,
          ' ,epochs:', args.epochs, ' batch_size:', args.batch_size,
          ' ,early_stopping_patience:', args.early_stopping_patience,
          ' ,reduce_lr_patience:', args.reduce_lr_patience,
          ' ,eval_model:', args.eval_model,
          ' ,predict:', args.predict,
          )
    main(args)
