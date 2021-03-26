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


data_dir = '/home/user/Robotics/Data_sets/slip_detection/will_dataset/data_collection_001_122/data_collection_001/'
out_dir = '/home/user/Robotics/Data_sets/slip_detection/manual_slip_detection/'
sequence_length = 20

files = glob.glob(data_dir + '/*')

path_file = []

for experiment_number in tqdm(range(len(files))):
	robot_state  = np.asarray(pd.read_csv(files[experiment_number] + '/robot_state.csv', header=None))
	proximity    = np.asarray(pd.read_csv(files[experiment_number] + '/proximity.csv', header=None))
	xela_sensor1 = np.asarray(pd.read_csv(files[experiment_number] + '/xela_sensor1.csv', header=None))
	xela_sensor2 = np.asarray(pd.read_csv(files[experiment_number] + '/xela_sensor2.csv', header=None))
	meta_data = np.asarray(pd.read_csv(files[experiment_number] + '/meta_data.csv', header=None))

	####################################### Robot Data ###########################################
	ee_positions = []
	ee_position_x, ee_position_y, ee_position_z = [], [], []
	ee_orientation_x, ee_orientation_y, ee_orientation_z = [], [], []

	for state in robot_state[1:]:
		ee_positions.append([float(item) for item in robot_state[1][-7:-4]])
		ee_position_x.append(state[-7])
		ee_position_y.append(state[-6])
		ee_position_z.append(state[-5])
		ee_orientation = R.from_quat([state[-4], state[-3], state[-2], state[-1]]).as_euler('zyx', degrees=True)
		ee_orientation_x.append(ee_orientation[0])
		ee_orientation_y.append(ee_orientation[1])
		ee_orientation_z.append(ee_orientation[2])

	ee_position_x = np.asarray(ee_position_x).astype(float)
	ee_position_y = np.asarray(ee_position_y).astype(float)
	ee_position_z = np.asarray(ee_position_z).astype(float)

	ee_orientation_x = np.asarray(ee_orientation_x).astype(float)
	ee_orientation_y = np.asarray(ee_orientation_y).astype(float)
	ee_orientation_z = np.asarray(ee_orientation_z).astype(float)

	# normalise for each value:
	min_x_position_x, max_x_position_x = (min(ee_position_x), max(ee_position_x))
	min_y_position_y, max_y_position_y = (min(ee_position_y), max(ee_position_y))
	min_z_position_z, max_z_position_z = (min(ee_position_z), max(ee_position_z))

	min_x_orientation_x, max_x_orientation_x = (min(ee_orientation_x), max(ee_orientation_x))
	min_y_orientation_y, max_y_orientation_y = (min(ee_orientation_y), max(ee_orientation_y))
	min_z_orientation_z, max_z_orientation_z = (min(ee_orientation_z), max(ee_orientation_z))

	for time_step in range(len(ee_position_x)):
		ee_position_x[time_step] = (ee_position_x[time_step] - min_x_position_x) / (max_x_position_x - min_x_position_x) 
		ee_position_y[time_step] = (ee_position_y[time_step] - min_y_position_y) / (max_y_position_y - min_y_position_y) 
		ee_position_z[time_step] = (ee_position_z[time_step] - min_z_position_z) / (max_z_position_z - min_z_position_z)

		ee_orientation_x[time_step] = (ee_orientation_x[time_step] - min_x_orientation_x) / (max_x_orientation_x - min_x_orientation_x) 
		ee_orientation_y[time_step] = (ee_orientation_y[time_step] - min_y_orientation_y) / (max_y_orientation_y - min_y_orientation_y) 
		ee_orientation_z[time_step] = (ee_orientation_z[time_step] - min_z_orientation_z) / (max_z_orientation_z - min_z_orientation_z)

	ee_position_x = np.asarray(ee_position_x)
	ee_position_y = np.asarray(ee_position_y)
	ee_position_z = np.asarray(ee_position_z)
	ee_orientation_x = np.asarray(ee_orientation_x)
	ee_orientation_y = np.asarray(ee_orientation_y)
	ee_orientation_z = np.asarray(ee_orientation_z)


	####################################### Xela Sensor Data ###########################################
	xela_sensor1_data_x, xela_sensor1_data_y, xela_sensor1_data_z = [], [], []
	xela_sensor2_data_x, xela_sensor2_data_y, xela_sensor2_data_z = [], [], []
	xela_sensor1_data_x_mean, xela_sensor1_data_y_mean, xela_sensor1_data_z_mean = [], [], []
	xela_sensor2_data_x_mean, xela_sensor2_data_y_mean, xela_sensor2_data_z_mean = [], [], []

	for sample1, sample2 in zip(xela_sensor1[1:], xela_sensor2[1:]):
		sample1_data_x, sample1_data_y, sample1_data_z = [], [], []
		sample2_data_x, sample2_data_y, sample2_data_z = [], [], []

		for i in range(0, len(xela_sensor1[0]), 3):
			sample1_data_x.append(float(sample1[i]))
			sample1_data_y.append(float(sample1[i+1]))
			sample1_data_z.append(float(sample1[i+2]))

			sample2_data_x.append(float(sample2[i]))
			sample2_data_y.append(float(sample2[i+1]))
			sample2_data_z.append(float(sample2[i+2]))

		xela_sensor1_data_x.append(sample1_data_x)
		xela_sensor1_data_y.append(sample1_data_y)
		xela_sensor1_data_z.append(sample1_data_z)

		xela_sensor2_data_x.append(sample2_data_x)
		xela_sensor2_data_y.append(sample2_data_y)
		xela_sensor2_data_z.append(sample2_data_z)

	# normalise for each force:
	min_x_sensor1, max_x_sensor1 = (min([min(x) for x in xela_sensor1_data_x]), max([max(x) for x in xela_sensor1_data_x]))
	min_y_sensor1, max_y_sensor1 = (min([min(y) for y in xela_sensor1_data_y]), max([max(y) for y in xela_sensor1_data_y]))
	min_z_sensor1, max_z_sensor1 = (min([min(z) for z in xela_sensor1_data_z]), max([max(z) for z in xela_sensor1_data_z]))

	min_x_sensor2, max_x_sensor2 = (min([min(x) for x in xela_sensor2_data_x]), max([max(x) for x in xela_sensor2_data_x]))
	min_y_sensor2, max_y_sensor2 = (min([min(y) for y in xela_sensor2_data_y]), max([max(y) for y in xela_sensor2_data_y]))
	min_z_sensor2, max_z_sensor2 = (min([min(z) for z in xela_sensor2_data_z]), max([max(z) for z in xela_sensor2_data_z]))

	for time_step in range(len(xela_sensor1_data_x)):
		for i in range(np.asarray(xela_sensor1_data_x).shape[1]):
			xela_sensor1_data_x[time_step][i] = (xela_sensor1_data_x[time_step][i] - min_x_sensor1) / (max_x_sensor1 - min_x_sensor1) 
			xela_sensor1_data_y[time_step][i] = (xela_sensor1_data_y[time_step][i] - min_y_sensor1) / (max_y_sensor1 - min_y_sensor1) 
			xela_sensor1_data_z[time_step][i] = (xela_sensor1_data_z[time_step][i] - min_z_sensor1) / (max_z_sensor1 - min_z_sensor1)

			xela_sensor2_data_x[time_step][i] = (xela_sensor2_data_x[time_step][i] - min_x_sensor2) / (max_x_sensor2 - min_x_sensor2) 
			xela_sensor2_data_y[time_step][i] = (xela_sensor2_data_y[time_step][i] - min_y_sensor2) / (max_y_sensor2 - min_y_sensor2) 
			xela_sensor2_data_z[time_step][i] = (xela_sensor2_data_z[time_step][i] - min_z_sensor2) / (max_z_sensor2 - min_z_sensor2)

	####################################### Format data into time series ###########################################
	robot_data = []
	xela_1_data = []
	xela_2_data = []
	xela_1_labels = []
	xela_2_labels = []

	for sample in range(0, len(ee_position_x) - sequence_length):
		robot_data_sequence, xela_1_sequence_data, xela_2_sequence_data, xela_1_sequence_labels, xela_2_sequence_labels = [], [], [], [], []
		for t in range(0, sequence_length):
			robot_data_sequence.append([ee_position_x[sample+t], ee_position_y[sample+t], ee_position_z[sample+t], ee_orientation_x[sample+t], ee_orientation_y[sample+t], ee_orientation_z[sample+t]])
			xela_1_sequence_data.append(np.column_stack((xela_sensor1_data_x[sample+t], xela_sensor1_data_y[sample+t], xela_sensor1_data_z[sample+t])).flatten())
			xela_2_sequence_data.append(np.column_stack((xela_sensor2_data_x[sample+t], xela_sensor2_data_y[sample+t], xela_sensor2_data_z[sample+t])).flatten())

		robot_data.append(robot_data_sequence)
		xela_1_data.append(xela_1_sequence_data)
		xela_2_data.append(xela_2_sequence_data)

	np.save(out_dir + 'robot_data_' + str(experiment_number), robot_data)
	np.save(out_dir + 'xela_1_data_' + str(experiment_number), xela_1_data)
	np.save(out_dir + 'xela_2_data_' + str(experiment_number), xela_2_data)
	ref = []
	ref.append('robot_data_' + str(experiment_number) + '.npy')
	ref.append('xela_1_data_' + str(experiment_number) + '.npy')
	ref.append('xela_2_data_' + str(experiment_number) + '.npy')
	path_file.append(ref)


with open(out_dir + '/map.csv', 'w') as csvfile:
	writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
	writer.writerow(['robot_data_path', 'xela_1_data_path', 'xela_2_data_path', 'experiment', 'sample_in_experiment'])
	for row in path_file:
		writer.writerow(row)

