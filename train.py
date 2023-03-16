import time
import torch


def train_one_epoch(args, model, device, train_loader, optimizer, loss_fn):
    model = model.to(device)
    total_loss, total_size = 0, 0
    i = 0
    model.train(mode=True)
    tbe = time.time()
    for seqimgs, vlen, poses, labels in train_loader:
        batch_size = labels.shape[0]
        seqimgs = seqimgs.to(device)
        vlen = vlen.to(device)
        poses = poses.to(device)
        labels = labels.to(device)  # 数据使用GPU

        seqimgs = seqimgs.to(torch.float32)
        poses = poses.to(torch.float32)
        labels = labels.to(torch.float32)

        optimizer.zero_grad()  # 优化，注意梯度清零
        preds = model(seqimgs, vlen, poses)  #
        # print('pred:', preds)
        # print('label:', labels)
        loss = loss_fn(preds, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_size
        total_size += batch_size
        if (i + 1) % 1000 == 0:
            loss_report = total_loss / total_size
            print('Step [{}] done, Loss_[]: {:.4f}'.format(total_size, loss_report), (time.time() - tbe) / total_size)
        i = i + 1
    train_loss = total_loss / total_size
    return model, train_loss
