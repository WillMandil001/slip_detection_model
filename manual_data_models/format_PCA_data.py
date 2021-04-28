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
from sklearn.decomposition import PCA

from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.spatial.transform import Rotation as R
from scipy.ndimage.interpolation import map_coordinates

data_dir = '/home/user/Robotics/Data_sets/slip_detection/will_dataset/data_collection_001_122/data_collection_001/'
out_dir = '/home/user/Robotics/Data_sets/slip_detection/manual_slip_detection/'
SAVE_IMAGES= True
sequence_length = 20
image_height, image_width = 32, 32

def create_image(xela_sensor1_data_x, xela_sensor1_data_y, xela_sensor1_data_z):
	image = np.zeros((4,4,3), np.float32)
	index = 0
	for x in range(4):
		for y in range(4):
			image[x][y] =  [xela_sensor1_data_x[index], 
							xela_sensor1_data_y[index], 
							xela_sensor1_data_z[index]]
			index += 1
	reshaped_image = np.rot90(cv2.resize(image.astype(np.float32), dsize=(image_height, image_width), interpolation=cv2.INTER_CUBIC), k=1, axes=(0, 1))
	return reshaped_image


files = glob.glob(data_dir + '/*')
path_file = []
index_to_save = 0

ee_positions = []
ee_position_x, ee_position_y, ee_position_z = [], [], []
ee_orientation_x, ee_orientation_y, ee_orientation_z = [], [], []
ee_orientation_quat_x, ee_orientation_quat_y, ee_orientation_quat_z, ee_orientation_quat_w = [], [], [], []

xela_sensor1_data_x, xela_sensor1_data_y, xela_sensor1_data_z = [], [], []
xela_sensor2_data_x, xela_sensor2_data_y, xela_sensor2_data_z = [], [], []
xela_sensor1_data_x_mean, xela_sensor1_data_y_mean, xela_sensor1_data_z_mean = [], [], []
xela_sensor2_data_x_mean, xela_sensor2_data_y_mean, xela_sensor2_data_z_mean = [], [], []

exp_break_points = []
exp_break_point = 0 

for experiment_number in tqdm(range(len(files))):
	robot_state  = np.asarray(pd.read_csv(files[experiment_number] + '/robot_state.csv', header=None))
	proximity    = np.asarray(pd.read_csv(files[experiment_number] + '/proximity.csv', header=None))
	xela_sensor1 = np.asarray(pd.read_csv(files[experiment_number] + '/xela_sensor1.csv', header=None))
	xela_sensor2 = np.asarray(pd.read_csv(files[experiment_number] + '/xela_sensor2.csv', header=None))
	meta_data = np.asarray(pd.read_csv(files[experiment_number] + '/meta_data.csv', header=None))

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
	
	exp_break_points.append(exp_break_point)
	break

	# PCA:
my_array = np.array([[11,22,33],[44,55,66]])

pca = PCA(n_components=16)
pca.fit(xela_sensor1_offset_x)
cov = pca.get_covariance()
print(cov)
# xela_sensor1_offset_x_pandas = pd.DataFrame(xela_sensor1_offset_x)#, columns = ['Column_A','Column_B','Column_C'])
# xela_sensor1_offset_y_pandas = pd.DataFrame(xela_sensor1_offset_y)#, columns = ['Column_A','Column_B','Column_C'])
# xela_sensor1_offset_z_pandas = pd.DataFrame(xela_sensor1_offset_z)#, columns = ['Column_A','Column_B','Column_C'])

print(df)
print(type(df))
	# chop the data back into experiments for sequence creation:

	# format the data:

	# save the data:

# ####################################### Format data into time series ###########################################
# for sample in range(0, len(ee_position_x) - sequence_length):
# 	robot_data_euler_sequence, robot_data_quat_sequence, xela_1_sequence_data, xela_2_sequence_data, experiment_data_sequence, time_step_data_sequence, xela_image_1_data_sequence, xela_image_2_data_sequence = [], [], [], [], [], [], [], []
# 	for t in range(0, sequence_length):
# 		robot_data_euler_sequence.append([ee_position_x[sample+t], ee_position_y[sample+t], ee_position_z[sample+t], ee_orientation_x[sample+t], ee_orientation_y[sample+t], ee_orientation_z[sample+t]])
# 		robot_data_quat_sequence.append([ee_position_x[sample+t], ee_position_y[sample+t], ee_position_z[sample+t], ee_orientation_quat_x[sample+t], ee_orientation_quat_y[sample+t], ee_orientation_quat_z[sample+t], ee_orientation_quat_w[sample+t]])
# 		xela_1_sequence_data.append(np.column_stack((xela_sensor1_data_x[sample+t], xela_sensor1_data_y[sample+t], xela_sensor1_data_z[sample+t])).flatten())
# 		# xela_2_sequence_data.append(np.column_stack((xela_sensor2_data_x[sample+t], xela_sensor2_data_y[sample+t], xela_sensor2_data_z[sample+t])).flatten())
# 		if SAVE_IMAGES == True:
# 			xela_image_1_data_sequence.append(xela_images_1[sample+t])
# 			# xela_image_2_data_sequence.append(xela_images_2[sample+t])
# 		experiment_data_sequence.append(experiment_number)
# 		time_step_data_sequence.append(sample+t)
# 	# robot_data.append(robot_data_sequence)
# 	# xela_1_data.append(xela_1_sequence_data)
# 	# xela_2_data.append(xela_2_sequence_data)
# 	# xela_image_1_data.append(xela_image_1_data_sequence)
# 	# xela_image_2_data.append(xela_image_2_data_sequence)
# 	# experiment_data.append(experiment_number)
# 	# time_step_data.append(time_step_data_sequence)

# 	np.save(out_dir + 'robot_data_euler_' + str(index_to_save), robot_data_euler_sequence)
# 	np.save(out_dir + 'robot_data_quat_' + str(index_to_save), robot_data_quat_sequence)
# 	np.save(out_dir + 'xela_1_data_' + str(index_to_save), xela_1_sequence_data)
# 	np.save(out_dir + 'xela_2_data_' + str(index_to_save), xela_2_sequence_data)
# 	if SAVE_IMAGES == True:
# 		np.save(out_dir + 'xela_1_image_data_' + str(index_to_save), xela_image_1_data_sequence)
# 		# np.save(out_dir + 'xela_2_image_data_' + str(index_to_save), xela_image_2_data_sequence)
# 	np.save(out_dir + 'experiment_number_' + str(index_to_save), experiment_data_sequence)
# 	np.save(out_dir + 'time_step_data_' + str(index_to_save), time_step_data_sequence)
# 	ref = []
# 	ref.append('robot_data_euler_' + str(index_to_save) + '.npy')
# 	ref.append('robot_data_quat_' + str(index_to_save) + '.npy')
# 	ref.append('xela_1_data_' + str(index_to_save) + '.npy')
# 	ref.append('xela_2_data_' + str(index_to_save) + '.npy')
# 	if SAVE_IMAGES == True:
# 		ref.append('xela_1_image_data_' + str(index_to_save) + '.npy')
# 		# ref.append('xela_2_image_data_' + str(index_to_save) + '.npy')
# 	ref.append('experiment_number_' + str(index_to_save) + '.npy')
# 	ref.append('time_step_data_' + str(index_to_save) + '.npy')
# 	path_file.append(ref)
# 	index_to_save += 1



# ee_position_x = np.asarray(ee_position_x).astype(float)
# ee_position_y = np.asarray(ee_position_y).astype(float)
# ee_position_z = np.asarray(ee_position_z).astype(float)
# # quat:
# ee_orientation_quat_x = np.asarray(ee_orientation_quat_x).astype(float)
# ee_orientation_quat_y = np.asarray(ee_orientation_quat_y).astype(float)
# ee_orientation_quat_z = np.asarray(ee_orientation_quat_z).astype(float)
# ee_orientation_quat_w = np.asarray(ee_orientation_quat_w).astype(float)
# # euler:
# ee_orientation_x = np.asarray(ee_orientation_x).astype(float)
# ee_orientation_y = np.asarray(ee_orientation_y).astype(float)
# ee_orientation_z = np.asarray(ee_orientation_z).astype(float)



with open(out_dir + '/map.csv', 'w') as csvfile:
	writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
	if SAVE_IMAGES == True:
		writer.writerow(['robot_data_path_euler', 'robot_data_path_quat', 'xela_1_data_path', 'xela_2_data_path', 'xela_1_image_data_path', 'experiment_number', 'time_steps'])
	if SAVE_IMAGES == False:
		writer.writerow(['robot_data_path_euler', 'robot_data_path_quat', 'xela_1_data_path', 'xela_2_data_path', 'experiment_number', 'time_steps'])
	for row in path_file:
		writer.writerow(row)

