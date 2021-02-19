import csv
import tqdm
import click
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from string import digits

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torch.nn.functional as F

seed = 42
epochs = 20
batch_size = 32
learning_rate = 1e-3

lookback = 20
context_frames = lookback
sequence_length = lookback

context_epochs = 30
context_batch_size = 1
context_learning_rate = 1e-3

test_train_split = 0.9  # precentage of train data from total
DATA_SPLIT_PCT = 0.2
SEED = 123 #used to help randomly select the data points

torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")#  use gpu if available


class DataFormatter():
    def __init__(self):
        print("initialised data formatter")

    def get_datasets(self):
        #=======================================================================================================================
        # load dataset
        tactile_data = np.load('/home/user/Robotics/slip_detection_franka/python scripts/dataFiles/xela_numeric.npy')

        robot_data = np.load('/home/user/Robotics/slip_detection_franka/python scripts/dataFiles/robot_data_26993.npy')

        Y = pd.read_csv('/home/user/Robotics/slip_detection_franka/python scripts/dataFiles/labels_data_26993.csv')
        Y = Y['slip']

        print(tactile_data.shape)
        print(Y.shape)
        #========================================================================================================================
        # function for making suitable input for lstm
        def temporalize(X, y, lookback):
            output_X = []
            output_y = []
            for i in range(len(X)-lookback-1):
                t = []
                for j in range(1,lookback+1):
                    # Gather past records upto the lookback period
                    t.append(X[[(i+j+1)], :])
                output_X.append(t)
                output_y.append(y[i+lookback+1])
            return output_X, output_y


        # Temporalize the data
        X_tactile, y = temporalize(X = tactile_data[:-lookback], y = Y, lookback = lookback)
        X_robot_trajectory, zq = temporalize(X = robot_data[:], y = Y, lookback = lookback*2)
        label, rr = temporalize(X = tactile_data[lookback:], y = Y, lookback = lookback)

        #split data into train, validation and test set
        X_tactile_train, X_tactile_test, y_train, y_test = train_test_split(np.array(X_tactile), np.array(y), test_size=DATA_SPLIT_PCT, random_state=SEED)
        label_train, label_test, rr_train, rr_test = train_test_split(np.array(label), np.array(rr), test_size=DATA_SPLIT_PCT, random_state=SEED)
        X_robot_trajectory_train, X_robot_trajectory_test, zq_train, zq_test = train_test_split(np.array(X_robot_trajectory), np.array(zq), test_size=DATA_SPLIT_PCT, random_state=SEED)

        X_tactile_train, X_tactile_valid, y_train, y_valid = train_test_split(X_tactile_train, y_train, test_size=DATA_SPLIT_PCT, random_state=SEED)
        X_robot_trajectory_train, X_robot_trajectory_valid, zq_train, zq_valid = train_test_split(X_robot_trajectory_train, zq_train, test_size=DATA_SPLIT_PCT, random_state=SEED)
        label_train, label_valid, rr_train, rr_valid = train_test_split(np.array(label_train), np.array(rr_train), test_size=DATA_SPLIT_PCT, random_state=SEED)


        print('tactile train set input data shape:', X_tactile_train.shape)
        print('robot train set input data shape:', X_robot_trajectory_train.shape)
        #=============================================================================================================================
        #Reshaping the data
        #The tensors we have here are 4-dimensional. We will reshape them into the desired 3-dimensions corresponding to sample x lookback x features.
        n_features_tactile = 96
        n_features_robot = 4

        X_tactile_train = X_tactile_train.reshape(X_tactile_train.shape[0], lookback, n_features_tactile)
        X_tactile_valid = X_tactile_valid.reshape(X_tactile_valid.shape[0], lookback, n_features_tactile)
        X_tactile_test = X_tactile_test.reshape(X_tactile_test.shape[0], lookback, n_features_tactile)

        X_robot_trajectory_train = X_robot_trajectory_train.reshape(X_robot_trajectory_train.shape[0], lookback*2, n_features_robot)
        X_robot_trajectory_valid = X_robot_trajectory_valid.reshape(X_robot_trajectory_valid.shape[0], lookback*2, n_features_robot)
        X_robot_trajectory_test = X_robot_trajectory_test.reshape(X_robot_trajectory_test.shape[0], lookback*2, n_features_robot)

        label_train = label_train.reshape(label_train.shape[0], lookback, n_features_tactile)
        label_valid = label_valid.reshape(label_valid.shape[0], lookback, n_features_tactile)
        label_test = label_test.reshape(label_test.shape[0], lookback, n_features_tactile)
        #=============================================================================================================================
        #Standardize the data
        #It is usually better to use a standardized data (transformed to Gaussian, mean 0 and sd 1) for autoencoders.

        def flatten(X):
            '''
            Flatten a 3D array.
            Input
            X            A 3D array for lstm, where the array is sample x timesteps x features.
            Output
            flattened_X  A 2D array, sample x features.
            '''
            flattened_X = np.empty((X.shape[0], X.shape[2]))  # sample x features array.
            for i in range(X.shape[0]):
                flattened_X[i] = X[i, (X.shape[1]-1), :]
            return(flattened_X)

        def scale(X, scaler):
            '''
            Scale 3D array.
            Inputs
            X            A 3D array for lstm, where the array is sample x timesteps x features.
            scaler       A scaler object, e.g., sklearn.preprocessing.StandardScaler, sklearn.preprocessing.normalize   
            Output
            X            Scaled 3D array.
            '''
            for i in range(X.shape[0]):
                X[i, :, :] = scaler.transform(X[i, :, :])
                
            return X


        # Initialize a scaler using the training data.
        scaler_tactile = StandardScaler().fit(flatten(X_tactile_train))
        scaler_robot = StandardScaler().fit(flatten(X_robot_trajectory_train))

        X_tactile_train_scaled = scale(X_tactile_train, scaler_tactile)
        X_tactile_valid_scaled = scale(X_tactile_valid, scaler_tactile)
        X_tactile_test_scaled = scale(X_tactile_test, scaler_tactile)

        X_robot_trajectory_train_scaled = scale(X_robot_trajectory_train, scaler_robot)
        X_robot_trajectory_valid_scaled = scale(X_robot_trajectory_valid, scaler_robot)
        X_robot_trajectory_test_scaled = scale(X_robot_trajectory_test, scaler_robot)

        label_train_scaled = scale(label_train, scaler_tactile)
        label_valid_scaled = scale(label_valid, scaler_tactile)
        label_test_scaled = scale(label_test, scaler_tactile)

        a = flatten(X_tactile_train_scaled)
        print('colwise mean', np.mean(a, axis=0).round(6))
        print('colwise variance', np.var(a, axis=0))
        a = flatten(X_robot_trajectory_train_scaled)
        print('colwise mean', np.mean(a, axis=0).round(6))
        print('colwise variance', np.var(a, axis=0))
        a = flatten(label_train_scaled)
        print('colwise mean', np.mean(a, axis=0).round(6))
        print('colwise variance', np.var(a, axis=0))


        dataset_train = FullDataSet(X_tactile_train_scaled, X_robot_trajectory_train_scaled, label_train_scaled)
        dataset_valid = FullDataSet(X_tactile_valid_scaled, X_robot_trajectory_valid_scaled, label_valid_scaled)
        dataset_test  = FullDataSet(X_tactile_valid_scaled, X_robot_trajectory_valid_scaled, label_valid_scaled)

        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=True)

        return train_loader, valid_loader, test_loader


class FullDataSet():
    def __init__(self, taxelset, actionset, stateset):
        self.samples = []
        for i in range(len(taxelset)):
            self.samples.append([taxelset[i], actionset[i], stateset[i]])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self,idx):
        return(self.samples[idx])


class ModelTrainer:
    def __init__(self):
        ### Train the LSTM chain:
        DF = DataFormatter()
        self.train_loader, self.valid_loader, self.test_loader = DF.get_datasets()
        self.full_model = FullModel()
        self.criterion = nn.L1Loss()
        self.optimizer = optim.Adam(self.full_model.parameters(), lr=learning_rate)
        self.train_full_model()

    def train_full_model(self):
        previous_test_mean_loss = 1.0
        progress_bar = tqdm.tqdm(range(0, epochs), total=(epochs*len(self.train_loader)))
        mean_test = 0
        for epoch in progress_bar:
            loss = 0
            losses = 0.0
            for index, batch_features in enumerate(self.train_loader):
                # 2. Reshape data and send to device:
                tactile = batch_features[0].permute(1,0,2).to(device)
                action = batch_features[1].permute(1,0,2).to(device)
                label = batch_features[2].permute(1,0,2).to(device)

                print(tactile[0][0][0])
                print(action[0][0][0])
                print(label[0][0][0])

                tactile_predictions = self.full_model.forward(tactiles=tactile, actions=action)  # Step 3. Run our forward pass.
                self.optimizer.zero_grad()
                print(tactile_predictions.shape)
                print(label.shape)
                loss = self.criterion(tactile_predictions.unsqueeze(0).to(device), label)
                loss.backward()
                self.optimizer.step()

                losses += loss.item()
                if index:
                    mean = losses / index
                else:
                    mean = 0
                progress_bar.set_description("epoch: {}, ".format(epoch) + "loss: {:.4f}, ".format(float(loss.item())) + "mean loss: {:.4f}, ".format(mean))
                progress_bar.update()
            # print("Training mean loss: {:.4f}, ".format(losses / index))

            # test_losses = 0.0
            # test_loss = 0.0
            # with torch.no_grad():
            #     for index__, batch_features in enumerate(self.test_full_loader):
            #         # 1. Calculate context model: 
            #         context_data_list = []
            #         for context_data in batch_features[0]:
            #             context = context_data.view(-1, context_epochs*48).to(device)
            #             context = self.context_model.encoder(context)  # [0]
            #             context_list = []
            #             for sequence in range(sequence_length):
            #                 context_list.append(context.cpu().detach().numpy())
            #             context_data_list.append(context_list)
            #         context_data_list = np.asarray(context_data_list).squeeze()
            #         context_data_list = torch.FloatTensor(context_data_list)

            #         # 2. Reshape data and send to device:
            #         context = context_data_list.permute(1,0,2).to(device)
            #         tactile = batch_features[1].permute(1,0,2).to(device)
            #         action = batch_features[2].permute(1,0,2).to(device)
            #         state = batch_features[3].permute(1,0,2).to(device)

            #         tactile_predictions = self.full_model.forward(tactiles=tactile, actions=action, context=context)  # Step 3. Run our forward pass.
            #         self.optimizer.zero_grad()
            #         test_loss = self.criterion(tactile_predictions.unsqueeze(0).to(device), tactile[context_frames:])
            #         test_losses += test_loss.item()

            # print("Test mean loss: {:.4f}, ".format(test_losses / index__))

            # if previous_test_mean_loss < test_losses / index__:
            #     print("Early stopping")
            #     break
            # else:
            #     previous_test_mean_loss = test_losses / index__ 

class FullModel(nn.Module):
    def __init__(self):
        super(FullModel, self).__init__()
        self.lstm1 = nn.LSTM(96, 96).to(device)  # tactile
        self.lstm2 = nn.LSTM(4, 4).to(device)  # pos_vel
        self.fc1 = nn.Linear(96+6, 96)  # tactile + pos_vel
        self.lstm3 = nn.LSTM(96, 96).to(device)  # pos_vel

    def forward(self, tactiles, actions):
        outputs = []
        batch_size__ = tactiles.shape[1]
        hidden1 = (torch.zeros(1,batch_size__,96).to(device), torch.zeros(1,batch_size__,96).to(device))
        hidden2 = (torch.zeros(1,batch_size__,4).to(device), torch.zeros(1,batch_size__,4).to(device))
        hidden3 = (torch.zeros(1,batch_size__,96).to(device), torch.zeros(1,batch_size__,96).to(device))

        for index, sample_action in enumerate(actions.squeeze()):
            sample_action.to(device)
            if index >= lookback:
                out1, hidden1 = self.lstm1(out4, hidden1)
                out2, hidden2 = self.lstm2(sample_action.unsqueeze(0), hidden2)
                robot_and_tactile = torch.cat((out2.squeeze(), out1.squeeze()), 1)
                out3 = self.fc1(robot_and_tactile.unsqueeze(0).cpu().detach())
                out4, hidden3 = self.lstm3(out3.to(device), hidden3)
                outputs.append(out4.squeeze())
            else:
                sample_tactile = tactiles[index]
                sample_tactile.to(device)
                out1, hidden1 = self.lstm1(sample_tactile.unsqueeze(0), hidden1)
                out2, hidden2 = self.lstm2(sample_action.unsqueeze(0), hidden2)
                robot_and_tactile = torch.cat((out2.squeeze(), out1.squeeze()), 1)
                out3 = self.fc1(robot_and_tactile.unsqueeze(0).cpu().detach())
                out4, hidden3 = self.lstm3(out3.to(device), hidden3)

        return torch.stack(outputs)

def main():
    MT = ModelTrainer()

if __name__ == '__main__':
    main()