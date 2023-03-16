import numpy as np
import os
import time
import pandas as pd
import sklearn.metrics
from sklearn.metrics import f1_score, classification_report
import torch
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import roc_curve, auc
from scipy import stats
from config import LABELS


def calc_ccc(preds, labels):
    """
    Concordance Correlation Coefficient
    :param preds: 1D np array
    :param labels: 1D np array
    :return:
    """
    preds_mean, labels_mean = np.mean(preds), np.mean(labels)
    cov_mat = np.cov(preds, labels)
    covariance = cov_mat[0, 1]
    preds_var, labels_var = cov_mat[0, 0], cov_mat[1, 1]

    ccc = 2.0 * covariance / (preds_var + labels_var + (preds_mean - labels_mean) ** 2)
    return ccc


def mean_ccc(preds, labels):
    """
    :param preds: list of list of lists (num batches, batch_size, num_classes)
    :param labels: same
    :return: scalar
    """
    preds = np.row_stack([np.array(p) for p in preds])
    labels = np.row_stack([np.array(l) for l in labels])
    num_classes = preds.shape[1]
    class_wise_cccs = np.array([calc_ccc(preds[:, i], labels[:, i]) for i in range(num_classes)])
    meanccc = np.mean(class_wise_cccs)  # mean valence and arousal
    return meanccc


def calc_pearsons(preds, labels):
    r = stats.pearsonr(preds, labels)
    return r[0]


def mean_pearsons(preds, labels):
    preds = np.row_stack([np.array(p) for p in preds])
    labels = np.row_stack([np.array(l) for l in labels])
    num_classes = preds.shape[1]
    class_wise_r = np.array([calc_pearsons(preds[:, i], labels[:, i]) for i in range(num_classes)])
    mean_r = np.mean(class_wise_r)
    return mean_r


def calc_auc(preds, labels):
    # preds = np.concatenate(preds)
    # labels = np.concatenate(labels)
    fpr, tpr, thresholds = roc_curve(labels, preds)
    return auc(fpr, tpr)


def cal_auc(preds, labels):
    # preds = np.concatenate(preds)
    # preds = np.argmax(preds, axis=1)
    # preds = torch.Tensor(preds)
    # labels = np.concatenate(labels)
    # labels = np.argmax(labels, axis=1)
    # labels = torch.Tensor(labels)
    # print(labels)
    # print(preds)
    sum = 0
    true = 0
    for i in range(preds.shape[0]):
        if preds[i] == labels[i]:
            true = true + 1
        sum = sum + 1
    return true / sum


def cal_f1(labels, preds):
    labels = np.concatenate(labels)
    labels = np.argmax(labels, axis=1)
    preds = np.concatenate(preds)
    preds = np.argmax(preds, axis=1)
    f1 = f1_score(labels, preds, labels=[0, 1, 2, 3, 4, 5, 6, 7], average='macro', zero_division="warn")
    return f1


def cal_ccc_pcc(preds, labels):
    preds = np.concatenate(preds)
    labels = np.concatenate(labels)
    pred0 = preds[:, 0]
    label0 = labels[:, 0]
    # pred1 = preds[:, 1]
    # label1 = labels[:, 1]
    # print('pred0',pred0,pred0.shape, type(pred0))
    # print('label0',label0,label0.shape, type(label0))
    Vccc = calc_ccc(pred0, label0)
    Accc = 0  # calc_ccc(pred1, label1)
    Vpcc = calc_pearsons(pred0, label0)
    Apcc = 0  # calc_pearsons(pred1, label1)
    return Vccc, Accc, Vpcc, Apcc


def cal_acc_f1(preds, labels):
    preds = np.concatenate(preds)
    labels = np.concatenate(labels)
    # print('preds',preds,preds.shape, type(preds))
    # print('labels',labels,labels.shape, type(labels))
    # pred1 = preds#[:, 0:8]
    # label1 = labels#[:, 0:8]
    # print('pred1',pred1,pred1.shape, type(pred1))
    # print('pred1',pred1,pred1.shape, type(pred1))
    pred1 = np.argmax(preds, axis=1)
    pred1 = torch.Tensor(pred1)
    label1 = np.argmax(labels, axis=1)
    label1 = torch.Tensor(label1)
    # print('pred1',pred1,pred1.shape, type(pred1))
    # print('label1',label1,label1.shape, type(label1))

    acc = cal_auc(pred1, label1)
    # listf1 = f1_score(label1, pred1, labels=[0, 1, 2, 3, 4, 5, 6, 7], average=None, zero_division="warn")
    f1 = f1_score(label1, pred1, labels=[0, 1, 2, 3, 4, 5, 6, 7], average='macro', zero_division="warn")
    print('模型报告', classification_report(label1, pred1, labels=[0, 1, 2, 3, 4, 5, 6, 7]))
    return acc, f1


def write_reaction_predictions(full_preds, csv_dir, filename):
    # meta_arr = np.row_stack(full_metas)
    # meta_arr = meta_arr.squeeze()
    preds_arr = np.row_stack(full_preds)
    pred_df = pd.DataFrame(columns=['File_ID'] + LABELS)
    pred_df['File_ID'] = '[00000]'
    pred_df[REACTION_LABELS] = preds_arr
    pred_df.to_csv(os.path.join(csv_dir, filename), index=False)
    return None


def write_predictions(task, full_preds, prediction_path, filename):
    assert prediction_path != ''

    if not os.path.exists(prediction_path):
        os.makedirs(prediction_path)

    if task == 'reaction':
        return write_reaction_predictions(full_preds, prediction_path, filename)

    metas_flat = []
    # for meta in full_metas:
    #     metas_flat.extend(meta)
    preds_flat = []
    for pred in full_preds:
        preds_flat.extend(pred if isinstance(pred, list) else (pred.squeeze() if pred.ndim > 1 else pred))

    if isinstance(metas_flat[0], list):
        num_meta_cols = len(metas_flat[0])
    else:
        # np array
        num_meta_cols = metas_flat[0].shape[0]
    prediction_df = pd.DataFrame(columns=[f'meta_col_{i}' for i in range(num_meta_cols)])
    for i in range(num_meta_cols):
        prediction_df[f'meta_col_{i}'] = [m[i] for m in metas_flat]
    prediction_df['prediction'] = preds_flat
    # prediction_df['label'] = labels_flat
    prediction_df.to_csv(os.path.join(prediction_path, filename), index=False)


def flatten_stress_for_ccc(lst):
    """
    Brings full_preds and full_labels of stress into the right format for the CCC function
    :param lst: list of lists of different lengths
    :return: flattened numpy array
    """
    return np.concatenate([np.array(l) for l in lst])


def evaluate(args, model, device, data_loader, loss_fn, eval_fn):
    model = model.to(device)
    total_loss, total_size = 0, 0
    full_preds = []
    full_labels = []
    i = 0
    model.eval()
    with torch.no_grad():
        for seqimgs, vlen, poses, labels in data_loader:  # , imgseqs, audios
            # if torch.any(torch.isnan(labels)):
            #    print('No labels available, no evaluation')
            #    return np.nan, np.nan
            batch_size = labels.shape[0]
            cutoff = batch_size

            seqimgs = seqimgs.to(device)
            vlen = vlen.to(device)
            poses = poses.to(device)
            labels = labels.to(device)  # 数据使用GPU

            seqimgs = seqimgs.to(torch.float32)
            poses = poses.to(torch.float32)
            labels = labels.to(torch.float32)

            preds = model(seqimgs, vlen, poses)  # , imgseqs, audios
            # print('pred:',preds)
            # print('label',labels)
            loss = loss_fn(preds, labels)
            # print(loss)

            total_loss += loss.item() * batch_size
            total_size += batch_size
            full_preds.append(preds.cpu().detach().numpy().tolist()[:cutoff])
            full_labels.append(labels.cpu().detach().numpy().tolist()[:cutoff])
            if (i + 1) % 1000 == 0:
                print('Step [{}] done'.format(total_size), (time.time() - tbe) / total_size)
            i = i + 1
            # plt.plot(full_preds[:200],
            #          color='red',  # 线颜色
            #          linewidth=1.0,  # 线宽
            #          )
            # plt.plot(full_labels[:200],
            #          color='blue',  # 线颜色
            #          linewidth=1.0,  # 线宽
            #          )
            # plt.title("eval")
            # plt.show()
        eval_loss = total_loss / total_size
        # print('full_preds:',full_preds, type(full_preds))
        # print('full_labels',full_labels)
        acc, f1 = eval_fn(full_preds, full_labels)
        return eval_loss, acc, f1

