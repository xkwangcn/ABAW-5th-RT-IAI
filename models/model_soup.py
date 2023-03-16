#!/usr/bin/env python
import torch
import torch.nn as nn
# import torch.nn.functional as F
# from emonet_split import EmoNet, EmoNet_
from models.model_fan import Model_fan
# nn.InstanceNorm2d = nn.BatchNorm2d
from collections import OrderedDict


def avg_modelsoup():
    model = Model_fan()
    model1 = torch.load("/mnt/wd0/home_back/shutao/ABAW/models/model_3799.pth")
    model2 = torch.load("/mnt/wd0/home_back/shutao/ABAW/models/model_3685.pth")
    # net3 = torch.load('models/model_3523.pth')
    # checkpoint1 = model1.state_dict()
    # checkpoint2 = model2.state_dict()
    # checkpoint3 = (checkpoint1 + checkpoint2)/2
    # print(checkpoint1)
    # print(checkpoint2)
    soup_models = [model1, model2]
    worker_state_dict = [x.state_dict() for x in soup_models]
    weight_keys = list(worker_state_dict[0].keys())
    fed_state_dict = OrderedDict()
    for key in weight_keys:
        key_sum = 0
        for i in range(len(soup_models)):
            key_sum = key_sum + worker_state_dict[i][key]
        fed_state_dict[key] = key_sum / len(soup_models)
    print(fed_state_dict)
    #### update fed weights to fl model
    model.load_state_dict(fed_state_dict)
    print('model soup down')
    return model



if __name__ == '__main__':
    model = Model_fan()
    model1 = torch.load("model_3799.pth")
    model2 = torch.load("model_3685.pth")
    # net3 = torch.load('models/model_3523.pth')
    # checkpoint1 = model1.state_dict()
    # checkpoint2 = model2.state_dict()
    # checkpoint3 = (checkpoint1 + checkpoint2)/2
    # print(checkpoint1)
    # print(checkpoint2)

    soup_models = [model1, model2]
    worker_state_dict = [x.state_dict() for x in soup_models]
    weight_keys = list(worker_state_dict[0].keys())
    fed_state_dict = OrderedDict()
    for key in weight_keys:
        key_sum = 0
        for i in range(len(soup_models)):
            key_sum = key_sum + worker_state_dict[i][key]
        fed_state_dict[key] = key_sum / len(soup_models)
    print(fed_state_dict)
    #### update fed weights to fl model
    model.load_state_dict(fed_state_dict)
    print('model soup down')