# -*- coding: utf-8 -*-
### RUN IN PYTHON 3
import os
import cv2
import csv
import glob
import click
import logging
import numpy as np
import pandas as pd

from PIL import Image 
from tqdm import tqdm
from dotenv import find_dotenv, load_dotenv
from sklearn import preprocessing
from sklearn.decomposition import PCA

from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.spatial.transform import Rotation as R
from scipy.ndimage.interpolation import map_coordinates

# -*- coding: utf-8 -*-
import csv
import tqdm
import copy
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

# -*- coding: utf-8 -*-
### RUN IN PYTHON 3
import os
import cv2
import csv
import glob
import click
import logging


from PIL import Image 
from tqdm import tqdm
from dotenv import find_dotenv, load_dotenv

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.ndimage.interpolation import map_coordinates
from scipy.spatial.transform import Rotation as R


SAVE_IMAGES= True
sequence_length = 20
image_height, image_width = 32, 32

def create_image(xela_sensor1_data):
	reshaped_image = cv2.resize(xela_sensor1_data.astype(np.float32), dsize=(image_height, image_width), interpolation=cv2.INTER_CUBIC)
	return reshaped_image

seed = 42
epochs = 100
batch_size = 32
learning_rate = 1e-3
context_frames = 10
sequence_length = 20
lookback = sequence_length

valid_train_split = 0.85  # precentage of train data from total
test_train_split = 0.95  # precentage of train data from total

torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")#  use gpu if available
data_dir = '/home/user/Robotics/Data_sets/slip_detection/will_dataset/data_collection_200/data_collection_001/'
out_dir = '/home/user/Robotics/Data_sets/slip_detection/will_dataset/formated_data/'

SAVE_IMAGES= True
sequence_length = 20
image_height, image_width = 32, 32
save_deriv1 = True
save_deriv2 = True

## Load the data:
files = glob.glob(data_dir + '/*')
path_file = []
index_to_save = 0

xela_sensor1_data_x_final, xela_sensor1_data_y_final, xela_sensor1_data_z_final = [], [], []
xela_sensor1_data_x_final_1stderiv = []
xela_sensor1_data_y_final_1stderiv = []
xela_sensor1_data_z_final_1stderiv = []
xela_sensor1_data_x_final_2stderiv = []
xela_sensor1_data_y_final_2stderiv = []
xela_sensor1_data_z_final_2stderiv = []

ee_positions_final = []
ee_position_x_final = []
ee_position_y_final = []
ee_position_z_final = []
ee_orientation_quat_x_final = []
ee_orientation_quat_y_final = []
ee_orientation_quat_z_final = []
ee_orientation_quat_w_final = []
ee_orientation_x_final = []
ee_orientation_y_final = []
ee_orientation_z_final = []

exp_break_points = []
exp_break_point = 0
cut = 0
for experiment_number in tqdm(range(len(files))):
	meta_data = np.asarray(pd.read_csv(files[experiment_number] + '/meta_data.csv', header=None))
	robot_state  = np.asarray(pd.read_csv(files[experiment_number] + '/robot_state.csv', header=None))
	proximity    = np.asarray(pd.read_csv(files[experiment_number] + '/proximity.csv', header=None))
	xela_sensor1 = np.asarray(pd.read_csv(files[experiment_number] + '/xela_sensor1.csv', header=None))
	xela_sensor2 = np.asarray(pd.read_csv(files[experiment_number] + '/xela_sensor2.csv', header=None))

	ee_positions = []
	ee_position_x, ee_position_y, ee_position_z = [], [], []
	ee_orientation_x, ee_orientation_y, ee_orientation_z = [], [], []
	ee_orientation_quat_x, ee_orientation_quat_y, ee_orientation_quat_z, ee_orientation_quat_w = [], [], [], []

	xela_sensor1_data_x, xela_sensor1_data_y, xela_sensor1_data_z = [], [], []
	xela_sensor2_data_x, xela_sensor2_data_y, xela_sensor2_data_z = [], [], []
	xela_sensor1_data_x_mean, xela_sensor1_data_y_mean, xela_sensor1_data_z_mean = [], [], []
	xela_sensor2_data_x_mean, xela_sensor2_data_y_mean, xela_sensor2_data_z_mean = [], [], []

	####################################### Robot Data ###########################################
	for state in robot_state[1:]:
		ee_positions.append([float(item) for item in robot_state[1][-7:-4]])
		ee_position_x.append(state[-7])
		ee_position_y.append(state[-6])
		ee_position_z.append(state[-5])
		# quat
		ee_orientation_quat_x.append(state[-4])
		ee_orientation_quat_y.append(state[-3])
		ee_orientation_quat_z.append(state[-2])
		ee_orientation_quat_w.append(state[-1])
		# euler
		ee_orientation = R.from_quat([state[-4], state[-3], state[-2], state[-1]]).as_euler('zyx', degrees=True)
		ee_orientation_x.append(ee_orientation[0])
		ee_orientation_y.append(ee_orientation[1])
		ee_orientation_z.append(ee_orientation[2])
		exp_break_point += 1

	# fix the euler angles:
	for i in range(len(ee_orientation_z)):
		if ee_orientation_z[i] < 0:
			ee_orientation_z[i] += 360

	####################################### Xela Data ###########################################
	for sample1, sample2 in zip(xela_sensor1[1:], xela_sensor2[1:]):
		sample1_data_x, sample1_data_y, sample1_data_z = [], [], []
		sample2_data_x, sample2_data_y, sample2_data_z = [], [], []
		for i in range(0, len(xela_sensor1[0]), 3):
			sample1_data_x.append(float(sample1[i]))
			sample1_data_y.append(float(sample1[i+1]))
			sample1_data_z.append(float(sample1[i+2]))
		xela_sensor1_data_x.append(sample1_data_x)
		xela_sensor1_data_y.append(sample1_data_y)
		xela_sensor1_data_z.append(sample1_data_z)

	# mean starting values:
	xela_sensor1_average_starting_value_x = int(sum(xela_sensor1_data_x[0]) / len(xela_sensor1_data_x[0]))
	xela_sensor1_average_starting_value_y = int(sum(xela_sensor1_data_y[0]) / len(xela_sensor1_data_y[0]))
	xela_sensor1_average_starting_value_z = int(sum(xela_sensor1_data_z[0]) / len(xela_sensor1_data_z[0]))
	xela_sensor1_offset_x = [xela_sensor1_average_starting_value_x - tactile_starting_value for tactile_starting_value in xela_sensor1_data_x[0]]
	xela_sensor1_offset_y = [xela_sensor1_average_starting_value_y - tactile_starting_value for tactile_starting_value in xela_sensor1_data_y[0]]
	xela_sensor1_offset_z = [xela_sensor1_average_starting_value_z - tactile_starting_value for tactile_starting_value in xela_sensor1_data_z[0]]

	for time_step in range(len(xela_sensor1_data_x)):
		xela_sensor1_sample_x_test = [offset+real_value for offset, real_value in zip(xela_sensor1_offset_x, xela_sensor1_data_x[time_step])]
		xela_sensor1_sample_y_test = [offset+real_value for offset, real_value in zip(xela_sensor1_offset_y, xela_sensor1_data_y[time_step])]
		xela_sensor1_sample_z_test = [offset+real_value for offset, real_value in zip(xela_sensor1_offset_z, xela_sensor1_data_z[time_step])]
		for i in range(np.asarray(xela_sensor1_data_x).shape[1]):
			xela_sensor1_data_x[time_step][i] = xela_sensor1_sample_x_test[i]
			xela_sensor1_data_y[time_step][i] = xela_sensor1_sample_y_test[i] 
			xela_sensor1_data_z[time_step][i] = xela_sensor1_sample_z_test[i]

	# calculate the derivatives
	if save_deriv1 == True:
		cut = 1
		xela_deriv1_x = np.diff(np.array(xela_sensor1_data_x), axis=0).tolist()
		xela_deriv1_y = np.diff(np.array(xela_sensor1_data_y), axis=0).tolist()
		xela_deriv1_z = np.diff(np.array(xela_sensor1_data_z), axis=0).tolist()
		# store the data:
		xela_sensor1_data_x_final_1stderiv += xela_deriv1_x 
		xela_sensor1_data_y_final_1stderiv += xela_deriv1_y 
		xela_sensor1_data_z_final_1stderiv += xela_deriv1_z 

	if save_deriv2 == True:
		cut = 2
		xela1_deriv2_x = np.diff(np.array(xela_sensor1_data_x), axis=0, n=2).tolist()
		xela1_deriv2_y = np.diff(np.array(xela_sensor1_data_y), axis=0, n=2).tolist()
		xela1_deriv2_z = np.diff(np.array(xela_sensor1_data_z), axis=0, n=2).tolist()
		# store the data:
		xela_sensor1_data_x_final_2stderiv += xela1_deriv2_x 
		xela_sensor1_data_y_final_2stderiv += xela1_deriv2_y 
		xela_sensor1_data_z_final_2stderiv += xela1_deriv2_z        

	xela_sensor1_data_x_final += xela_sensor1_data_x[cut:]
	xela_sensor1_data_y_final += xela_sensor1_data_y[cut:]
	xela_sensor1_data_z_final += xela_sensor1_data_z[cut:]
	ee_positions_final += ee_positions[cut:]
	ee_position_x_final += ee_position_x[cut:]
	ee_position_y_final += ee_position_y[cut:]
	ee_position_z_final += ee_position_z[cut:]
	ee_orientation_quat_x_final += ee_orientation_quat_x[cut:]
	ee_orientation_quat_y_final += ee_orientation_quat_y[cut:]
	ee_orientation_quat_z_final += ee_orientation_quat_z[cut:]
	ee_orientation_quat_w_final += ee_orientation_quat_w[cut:]
	ee_orientation_x_final += ee_orientation_x[cut:]
	ee_orientation_y_final += ee_orientation_y[cut:]
	ee_orientation_z_final += ee_orientation_z[cut:]

	exp_break_points.append(exp_break_point - cut)


xela_sensor1_data_x_final = np.asarray(xela_sensor1_data_x_final)
xela_sensor1_data_y_final = np.asarray(xela_sensor1_data_y_final)
xela_sensor1_data_z_final = np.asarray(xela_sensor1_data_z_final)

scale_together = False
if scale_together == True:
	xela_sensor1_data = np.concatenate((xela_sensor1_data_x_final, xela_sensor1_data_y_final, xela_sensor1_data_z_final), axis=1)
	scaler_full = preprocessing.StandardScaler().fit(xela_sensor1_data)
	xela_sensor1_data_scaled = scaler_full.transform(xela_sensor1_data)
	min_max_scaler_full_data = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(xela_sensor1_data_scaled)
	xela_sensor1_data_scaled_minmax = min_max_scaler_full_data.transform(xela_sensor1_data_scaled)

elif scale_together == False:
	normalizedx = (xela_sensor1_data_x_final-np.min(xela_sensor1_data_x_final))/(np.max(xela_sensor1_data_x_final)-np.min(xela_sensor1_data_x_final))
	normalizedy = (xela_sensor1_data_y_final-np.min(xela_sensor1_data_y_final))/(np.max(xela_sensor1_data_y_final)-np.min(xela_sensor1_data_y_final))
	normalizedz = (xela_sensor1_data_z_final-np.min(xela_sensor1_data_z_final))/(np.max(xela_sensor1_data_z_final)-np.min(xela_sensor1_data_z_final))
	xela_sensor1_data_scaled_minmax = np.concatenate((normalizedx,
										normalizedy,
										normalizedz), axis=1)

if save_deriv1 == True:
	scaler_td1x = preprocessing.StandardScaler().fit(xela_sensor1_data_x_final_1stderiv)
	scaler_td1y = preprocessing.StandardScaler().fit(xela_sensor1_data_y_final_1stderiv)
	scaler_td1z = preprocessing.StandardScaler().fit(xela_sensor1_data_z_final_1stderiv)
	xelax_sensor1d1_data_scaled = scaler_td1x.transform(xela_sensor1_data_x_final_1stderiv)
	xelay_sensor1d1_data_scaled = scaler_td1y.transform(xela_sensor1_data_y_final_1stderiv)
	xelaz_sensor1d1_data_scaled = scaler_td1z.transform(xela_sensor1_data_z_final_1stderiv)
	xelad1_sensor1_data_scaled = np.concatenate((xelax_sensor1d1_data_scaled,
											   xelay_sensor1d1_data_scaled,
											   xelaz_sensor1d1_data_scaled), axis=1)
	
	min_max_scalerd1x_full_data = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(xelax_sensor1d1_data_scaled)
	min_max_scalerd1y_full_data = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(xelay_sensor1d1_data_scaled)
	min_max_scalerd1z_full_data = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(xelaz_sensor1d1_data_scaled)    
	xelad1x_sensor1_data_scaled_minmax = min_max_scalerd1x_full_data.transform(xelax_sensor1d1_data_scaled)
	xelad1y_sensor1_data_scaled_minmax = min_max_scalerd1y_full_data.transform(xelay_sensor1d1_data_scaled)
	xelad1z_sensor1_data_scaled_minmax = min_max_scalerd1z_full_data.transform(xelaz_sensor1d1_data_scaled)

	xelad1_sensor1_data_scaled_minmax = np.concatenate((xelad1x_sensor1_data_scaled_minmax,
										xelad1y_sensor1_data_scaled_minmax,
										xelad1z_sensor1_data_scaled_minmax), axis=1)

if save_deriv2 == True:
	scaler_td2x = preprocessing.StandardScaler().fit(xela_sensor1_data_x_final_2stderiv)
	scaler_td2y = preprocessing.StandardScaler().fit(xela_sensor1_data_y_final_2stderiv)
	scaler_td2z = preprocessing.StandardScaler().fit(xela_sensor1_data_z_final_2stderiv)
	xelax_sensor1d2_data_scaled = scaler_td2x.transform(xela_sensor1_data_x_final_2stderiv)
	xelay_sensor1d2_data_scaled = scaler_td2y.transform(xela_sensor1_data_y_final_2stderiv)
	xelaz_sensor1d2_data_scaled = scaler_td2z.transform(xela_sensor1_data_z_final_2stderiv)
	xelad2_sensor1_data_scaled = np.concatenate((xelax_sensor1d2_data_scaled,
											   xelay_sensor1d2_data_scaled,
											   xelaz_sensor1d2_data_scaled), axis=1)
	
	min_max_scalerd2x_full_data = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(xelax_sensor1d2_data_scaled)
	min_max_scalerd2y_full_data = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(xelay_sensor1d2_data_scaled)
	min_max_scalerd2z_full_data = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(xelaz_sensor1d2_data_scaled)    
	xelad2x_sensor1_data_scaled_minmax = min_max_scalerd2x_full_data.transform(xelax_sensor1d2_data_scaled)
	xelad2y_sensor1_data_scaled_minmax = min_max_scalerd2y_full_data.transform(xelay_sensor1d2_data_scaled)
	xelad2z_sensor1_data_scaled_minmax = min_max_scalerd2z_full_data.transform(xelaz_sensor1d2_data_scaled)

	xelad2_sensor1_data_scaled_minmax = np.concatenate((xelad2x_sensor1_data_scaled_minmax,
										xelad2y_sensor1_data_scaled_minmax,
										xelad2z_sensor1_data_scaled_minmax), axis=1)

# scale between 0 and 1:
ee_position_x_final = np.array(ee_position_x_final) 
ee_position_y_final = np.array(ee_position_y_final) 
ee_position_z_final = np.array(ee_position_z_final) 
ee_orientation_quat_x_final = np.array(ee_orientation_quat_x_final).reshape(-1, 1)
ee_orientation_quat_y_final = np.array(ee_orientation_quat_y_final).reshape(-1, 1)
ee_orientation_quat_z_final = np.array(ee_orientation_quat_z_final).reshape(-1, 1)
ee_orientation_quat_w_final = np.array(ee_orientation_quat_w_final).reshape(-1, 1)
ee_orientation_x_final = np.array(ee_orientation_x_final)
ee_orientation_y_final = np.array(ee_orientation_y_final)
ee_orientation_z_final = np.array(ee_orientation_z_final)
# xela_sensor1_principle_components = np.array(xela_sensor1_principle_components) 
# xela_sensor1_principle_components =  np.array(xela_sensor1_data_scaled)
xela_sensor1_principle_components =  np.array(xela_sensor1_data_scaled_minmax)
if save_deriv1 == True:
	xelad1_sensor1_data_scaled_minmax = np.array(xelad1_sensor1_data_scaled_minmax)
if save_deriv2 == True:
	xelad2_sensor1_data_scaled_minmax = np.array(xelad2_sensor1_data_scaled_minmax)

min_max_scaler_ee_position_x_final = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit(ee_position_x_final.reshape(-1, 1))
ee_position_x_final_scaled = min_max_scaler_ee_position_x_final.transform(ee_position_x_final.reshape(-1, 1))

min_max_scaler_ee_position_y_final = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit(ee_position_y_final.reshape(-1, 1))
ee_position_y_final_scaled = min_max_scaler_ee_position_y_final.transform(ee_position_y_final.reshape(-1, 1))

min_max_scaler_ee_position_z_final = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit(ee_position_z_final.reshape(-1, 1))
ee_position_z_final_scaled = min_max_scaler_ee_position_z_final.transform(ee_position_z_final.reshape(-1, 1))

min_max_scaler_ee_orientation_quat_x_final = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit(ee_orientation_quat_x_final)
ee_orientation_quat_x_final_scaled = min_max_scaler_ee_orientation_quat_x_final.transform(ee_orientation_quat_x_final)

min_max_scaler_ee_orientation_quat_y_final = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit(ee_orientation_quat_y_final)
ee_orientation_quat_y_final_scaled = min_max_scaler_ee_orientation_quat_y_final.transform(ee_orientation_quat_y_final)

min_max_scaler_ee_orientation_quat_z_final = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit(ee_orientation_quat_z_final)
ee_orientation_quat_z_final_scaled = min_max_scaler_ee_orientation_quat_z_final.transform(ee_orientation_quat_z_final)

min_max_scaler_ee_orientation_quat_w_final = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit(ee_orientation_quat_w_final)
ee_orientation_quat_w_final_scaled = min_max_scaler_ee_orientation_quat_w_final.transform(ee_orientation_quat_w_final)

min_max_scaler_ee_orientation_x_final = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit(ee_orientation_x_final.reshape(-1, 1))
ee_orientation_x_final_scaled = min_max_scaler_ee_orientation_x_final.transform(ee_orientation_x_final.reshape(-1, 1))

min_max_scaler_ee_orientation_y_final = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit(ee_orientation_y_final.reshape(-1, 1))
ee_orientation_y_final_scaled = min_max_scaler_ee_orientation_y_final.transform(ee_orientation_y_final.reshape(-1, 1))

min_max_scaler_ee_orientation_z_final = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit(ee_orientation_z_final.reshape(-1, 1))
ee_orientation_z_final_scaled = min_max_scaler_ee_orientation_z_final.transform(ee_orientation_z_final.reshape(-1, 1))

print("scale_together, ", scale_together)
if scale_together == True:
	min_max_scaler_xela_sensor1_principle_components = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(xela_sensor1_principle_components)
	xela_sensor1_principle_components_scaled = min_max_scaler_xela_sensor1_principle_components.transform(xela_sensor1_principle_components)
else:
	xela_sensor1_principle_components_scaled = np.array(xela_sensor1_data_scaled_minmax)

if save_deriv1 == True:
	xelad1_sensor1_principle_components_scaled = xelad1_sensor1_data_scaled_minmax
if save_deriv2 == True:
	xelad2_sensor1_principle_components_scaled = xelad2_sensor1_data_scaled_minmax

# Convert data back into split experiments to create the sequences:
ee_position_x_final_split = np.asarray(np.split(ee_position_x_final_scaled, exp_break_points)[0:-1])
ee_position_y_final_split = np.asarray(np.split(ee_position_y_final_scaled, exp_break_points)[0:-1])
ee_position_z_final_split = np.asarray(np.split(ee_position_z_final_scaled, exp_break_points)[0:-1])
ee_orientation_quat_x_final_split = np.asarray(np.split(ee_orientation_quat_x_final_scaled, exp_break_points)[0:-1])
ee_orientation_quat_y_final_split = np.asarray(np.split(ee_orientation_quat_y_final_scaled, exp_break_points)[0:-1])
ee_orientation_quat_z_final_split = np.asarray(np.split(ee_orientation_quat_z_final_scaled, exp_break_points)[0:-1])
ee_orientation_quat_w_final_split = np.asarray(np.split(ee_orientation_quat_w_final_scaled, exp_break_points)[0:-1])
ee_orientation_x_final_split = np.asarray(np.split(ee_orientation_x_final_scaled, exp_break_points)[0:-1])
ee_orientation_y_final_split = np.asarray(np.split(ee_orientation_y_final_scaled, exp_break_points)[0:-1])
ee_orientation_z_final_split = np.asarray(np.split(ee_orientation_z_final_scaled, exp_break_points)[0:-1])
xela_sensor1_principle_components_split = np.asarray(np.split(xela_sensor1_principle_components_scaled, exp_break_points)[0:-1])
print(xela_sensor1_principle_components_split[0][0])
if save_deriv1 == True:
	xela_sensor1_data_1stderiv_scaled_split = np.asarray(np.split(xelad1_sensor1_principle_components_scaled, exp_break_points)[0:-1])
if save_deriv2 == True:
	xela_sensor1_data_2stderiv_scaled_split = np.asarray(np.split(xelad2_sensor1_principle_components_scaled, exp_break_points)[0:-1])

# Shuffle data:
p = np.random.permutation(len(ee_position_x_final_split))
# p = np.delete(p, np.where(p==106))  # make 106 always in the test set (for comparing graphs)
# p = np.append(p, [106])
print("shuffle order: ", p)
ee_position_x_final_split = ee_position_x_final_split[p]
ee_position_y_final_split = ee_position_y_final_split[p]
ee_position_z_final_split = ee_position_z_final_split[p]
ee_orientation_quat_x_final_split = ee_orientation_quat_x_final_split[p]
ee_orientation_quat_y_final_split = ee_orientation_quat_y_final_split[p]
ee_orientation_quat_z_final_split = ee_orientation_quat_z_final_split[p]
ee_orientation_quat_w_final_split = ee_orientation_quat_w_final_split[p]
ee_orientation_x_final_split = ee_orientation_x_final_split[p]
ee_orientation_y_final_split = ee_orientation_y_final_split[p]
ee_orientation_z_final_split = ee_orientation_z_final_split[p]
xela_sensor1_principle_components_split = xela_sensor1_principle_components_split[p]
xela_sensor1_data_1stderiv_scaled_split = xela_sensor1_data_1stderiv_scaled_split[p]
xela_sensor1_data_2stderiv_scaled_split = xela_sensor1_data_2stderiv_scaled_split[p]
print(xela_sensor1_principle_components_split[0][0])

# convert to sequences:
robot_data_euler_sequence, robot_data_quat_sequence, xela_1_sequence_data, experiment_data_sequence, time_step_data_sequence = [], [], [], [], []
xela1_1stderiv_sequence = []
xela1_2stderiv_sequence = []
for experiment in range(len(ee_position_x_final_split)):
	for sample in range(0, len(ee_position_x_final_split[experiment]) - sequence_length):
		robot_data_euler_sample, robot_data_quat_sample, xela_1_sequ_sample, experiment_data_sample, time_step_data_sample = [], [], [], [], []
		xela_sensor1_1stderiv = []
		xela_sensor1_2stderiv = []
		for t in range(0, sequence_length):
			robot_data_euler_sample.append([ee_position_x_final_split[experiment][sample+t], ee_position_y_final_split[experiment][sample+t], ee_position_z_final_split[experiment][sample+t], ee_orientation_x_final_split[experiment][sample+t], ee_orientation_y_final_split[experiment][sample+t], ee_orientation_z_final_split[experiment][sample+t]])
			robot_data_quat_sample.append([ee_position_x_final_split[experiment][sample+t], ee_position_y_final_split[experiment][sample+t], ee_position_z_final_split[experiment][sample+t], ee_orientation_quat_x_final_split[experiment][sample+t][0], ee_orientation_quat_y_final_split[experiment][sample+t][0], ee_orientation_quat_z_final_split[experiment][sample+t][0], ee_orientation_quat_w_final_split[experiment][sample+t][0]])
			xela_1_sequ_sample.append(xela_sensor1_principle_components_split[experiment][sample+t].reshape(3,4,4).T)
			xela_sensor1_1stderiv.append(xela_sensor1_data_1stderiv_scaled_split[experiment][sample+t])
			xela_sensor1_2stderiv.append(xela_sensor1_data_2stderiv_scaled_split[experiment][sample+t])
			experiment_data_sample.append(experiment)
			time_step_data_sample.append(sample+t)
		robot_data_euler_sequence.append(robot_data_euler_sample)
		robot_data_quat_sequence.append(robot_data_quat_sample)
		xela_1_sequence_data.append(xela_1_sequ_sample)
		xela1_1stderiv_sequence.append(xela_sensor1_1stderiv)
		xela1_2stderiv_sequence.append(xela_sensor1_2stderiv)
		experiment_data_sequence.append(experiment_data_sample)
		time_step_data_sequence.append(time_step_data_sample)
robot_data_euler_sequence = np.array(robot_data_euler_sequence)
robot_data_quat_sequence = np.array(robot_data_quat_sequence)
xela_1_sequence_data = np.array(xela_1_sequence_data)
xela1_1stderiv_sequence = np.array(xela1_1stderiv_sequence)
xela1_2stderiv_sequence = np.array(xela1_2stderiv_sequence)
experiment_data_sequence = np.array(experiment_data_sequence)
time_step_data_sequence = np.array(time_step_data_sequence)

print(xela_1_sequence_data.shape)
## save the data:
# create images from xela_data
index_to_save = 0
for time_step in tqdm(range(len(robot_data_euler_sequence))):
	xela_images_1 = []
	for seq in range(len(robot_data_euler_sequence[0])):
		xela_images_1.append(create_image(xela_1_sequence_data[time_step][seq]))
	np.save(out_dir + 'robot_data_euler_' + str(index_to_save), robot_data_euler_sequence[time_step])
	np.save(out_dir + 'xela_1_image_data_' + str(index_to_save), np.array(xela_images_1))
	np.save(out_dir + 'experiment_number_' + str(index_to_save), experiment_data_sequence[time_step])
	np.save(out_dir + 'time_step_data_' + str(index_to_save), time_step_data_sequence[time_step])
	ref = []
	ref.append('robot_data_euler_' + str(index_to_save) + '.npy')
	ref.append('xela_1_image_data_' + str(index_to_save) + '.npy')
	ref.append('experiment_number_' + str(index_to_save) + '.npy')
	ref.append('time_step_data_' + str(index_to_save) + '.npy')
	path_file.append(ref)
	index_to_save += 1

with open(out_dir + '/map.csv', 'w') as csvfile:
	writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
	if SAVE_IMAGES == True:
		writer.writerow(['robot_data_path_euler', 'xela_1_image_data_path', 'experiment_number', 'time_steps'])
	for row in path_file:
		writer.writerow(row)