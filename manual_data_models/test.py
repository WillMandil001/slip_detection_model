#!/usr/bin/env python
# coding: utf-8

# In[1]:


# -*- coding: utf-8 -*-
import csv
import tqdm
import click
import logging
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

from string import digits

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torch.nn.functional as F

seed = 42
epochs = 50
batch_size = 32
learning_rate = 1e-3
context_frames = 10
sequence_length = 16
lookback = sequence_length

context_epochs = 20
context_batch_size = 1
context_learning_rate = 1e-3
context_data_length = 20

valid_train_split = 0.8  # precentage of train data from total
test_train_split = 0.9  # precentage of train data from total

torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")#  use gpu if available


# In[2]:


class BatchGenerator:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        data_map = []
        with open(data_dir + 'map.csv', 'r') as f:  # rb
            reader = csv.reader(f)
            for row in reader:
                data_map.append(row)

        if len(data_map) <= 1: # empty or only header
            print("No file map found")
            exit()

        self.data_map = data_map

    def load_full_data(self):
        dataset_train = FullDataSet(self.data_dir, self.data_map, type_="train")
        dataset_valid = FullDataSet(self.data_dir, self.data_map, type_="valid")
        dataset_test = FullDataSet(self.data_dir, self.data_map, type_="test")
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False)
        return train_loader, valid_loader, test_loader


class FullDataSet():
    def __init__(self, data_dir, data_map, type_="train"):
        dataset_full = []
        for index, value in enumerate(data_map[1:]):  # ignore header
            robot = np.load(data_dir + value[0])
            xela1 = np.load(data_dir + value[1])
            xela2 = np.load(data_dir + value[2])
            experiment = np.load(data_dir + value[3])
            time_step  = np.load(data_dir + value[4])
            for i in range(len(robot)):
                dataset_full.append([robot[i].astype(np.float32),
                                     xela1[i].astype(np.float32),
                                     xela2[i].astype(np.float32),
                                     experiment[i],
                                     time_step[i]])
        if type_ == "train":
            self.samples = dataset_full[0:int(len(dataset_full)*test_train_split)]
        elif type_ == "valid":
            self.samples = dataset_full[int(len(dataset_full)*(valid_train_split)):int(len(dataset_full)*test_train_split)]
        elif type_ == "test":
            self.samples = dataset_full[int(len(dataset_full)*test_train_split):-1]

        data_map = None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self,idx):
        return(self.samples[idx])

 
data_dir = '/home/user/Robotics/Data_sets/slip_detection/manual_slip_detection/'
BG = BatchGenerator(data_dir)
print("done")


prev_exp = -1
train_full_loader, valid_full_loader, test_full_loader = BG.load_full_data()

experiment = []
with torch.no_grad():
    for index__, batch_features in enumerate(test_full_loader):
        for time_step in range(len(batch_features[0])):
            if batch_features[3][time_step] == 106:
                experiment.append([batch_features[0][time_step], batch_features[1][time_step], batch_features[2][time_step], batch_features[3][time_step], batch_features[4][time_step]])

time_step_to_test_t1 = 0
time_step_to_test_t9 = 5
predicted_data_t1 = []
predicted_data_t9 = []
groundtruth_data = []
time_step_series = []
for index, time_step_values in enumerate(experiment):
    time_step_series.append(time_step_values[4][time_step_to_test_t1])
    groundtruth_data.append(time_step_values[1][time_step_to_test_t1])

index = 0
titles = ["sheerx", "sheery", "normal"]
for j in range(3):
    for i in range(16):
        groundtruth_taxle = []
        predicted_taxel = []
        predicted_taxel_t1 = []
        predicted_taxel_t9 = []
        for k in range(len(groundtruth_data)):  # add in length of context data
            groundtruth_taxle.append(groundtruth_data[k][2])#[j+i])

        index += 1
        plt.title("Simple_LSTM")
        plt.plot(groundtruth_taxle, alpha=0.5, c="r", label="gt")
        plt.ylim([0, 1])
        plt.grid()
        plt.legend(loc="upper right")
        plt.show()

        break
    break









# action = batch_features[0].permute(1,0,2).to(device)
# tactile = batch_features[1].permute(1,0,2).to(device)

# model = MT.full_model
# data_dir = MT.data_dir

# criterion1 = nn.L1Loss()
# criterion2 = nn.MSELoss()
# tactile_predictions = []
# tactile_groundtruth = []
# test_lossesMAE = 0.0
# test_lossesMSE = 0.0
# with torch.no_grad():
#     for index__, batch_features in enumerate(MT.test_full_loader):
#         # 2. Reshape data and send to device:
#         action = batch_features[0].permute(1,0,2).to(device)
#         tactile = batch_features[1].permute(1,0,2).to(device)

#         tp = model.forward(tactiles=tactile, actions=action)
#         tactile_predictions.append(tp)  # Step 3. Run our forward pass.
#         tactile_groundtruth.append(tactile[context_frames:])
#         # calculate losses
#         test_lossMAE = criterion1(tp.to(device), tactile[context_frames:])
#         test_lossesMAE += test_lossMAE.item()
#         test_lossMSE = criterion2(tp.to(device), tactile[context_frames:])
#         test_lossesMSE += test_lossMSE.item()

# print("test loss MAE(L1): ", str(test_lossesMAE / index__))
# print("test loss MSE: ", str(test_lossesMSE / index__))



# print("test loss MAE(L1): ", str(test_lossesMAE / index__))
# print("test loss MSE: ", str(test_lossesMSE / index__))

# # calculate tactile values for full sample:
# time_step_to_test_t1 = 0    # [batch_set, prediction frames(t1->tx)(6), batch_size, features(48)]
# time_step_to_test_t9 = 5
# predicted_data_t1 = []
# predicted_data_t9 = []
# groundtruth_data = []
# for index, batch_set in enumerate(tactile_predictions):
#     for batch in range(0, len(batch_set[0])):
#         prediction_values = batch_set[time_step_to_test_t1][batch]
#         predicted_data_t1.append(prediction_values)
#         prediction_values = batch_set[time_step_to_test_t9][batch]
#         predicted_data_t9.append(prediction_values)
#         gt_values = tactile_groundtruth[index][time_step_to_test_t1][batch]
#         groundtruth_data.append(gt_values)  
# print("done")


# # In[50]:


# # calculate tactile values for full sample:
# time_step_to_test_t1 = 0    # [batch_set, prediction frames(t1->tx)(6), batch_size, features(48)]
# time_step_to_test_t9 = 5
# predicted_data_t1 = []
# predicted_data_t9 = []
# groundtruth_data = []
# for index, batch_set in enumerate(tactile_predictions):
#     for batch in range(0, len(batch_set[0])):
#         prediction_values = batch_set[time_step_to_test_t1][batch]
#         predicted_data_t1.append(prediction_values)
#         prediction_values = batch_set[time_step_to_test_t9][batch]
#         predicted_data_t9.append(prediction_values)
#         gt_values = tactile_groundtruth[index][time_step_to_test_t1][batch]
#         groundtruth_data.append(gt_values)  

# # test data
# index = 0
# titles = ["sheerx", "sheery", "normal"]
# for j in range(3):
#     for i in range(16):
#         groundtruth_taxle = []
#         predicted_taxel = []
#         predicted_taxel_t1 = []
#         predicted_taxel_t9 = []
#         # good = 140, 145 (lifting up the )
#         for k in range(450, 600):#len(predicted_data_t1)):#310, 325):#len(predicted_data_t1)):  # add in length of context data
#             predicted_taxel_t1.append(predicted_data_t1[k][j+i].cpu().detach().numpy())
#             predicted_taxel_t9.append(predicted_data_t9[k][j+i].cpu().detach().numpy())
#             groundtruth_taxle.append(groundtruth_data[k][j+i].cpu().detach().numpy())

#         index += 1
#         plt.title("Simple_LSTM")
#         plt.plot(predicted_taxel_t1, alpha=0.5, c="b", label="t5")
#         plt.plot(predicted_taxel_t9, alpha=0.5, c="g", label="t0")
#         plt.plot(groundtruth_taxle, alpha=0.5, c="r", label="gt")
#         plt.ylim([0, 1])
#         plt.grid()
#         plt.legend(loc="upper right")
# #         plt.savefig('/home/user/Robotics/slip_detection_model/manual_images/Simple_LSTM_new_data/simple_model_test_sample_' + str(index) + '.png')
#         plt.show()


# # In[ ]:



