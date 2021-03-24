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


class DataFormatter():
	def __init__(self, data_set_length, data_dir, out_dir, sequence_length, image_original_width, image_original_height, image_original_channel, state_action_dimension, create_img, create_img_prediction, context_data_length):
		self.data_dir = data_dir
		self.out_dir = out_dir
		self.sequence_length = sequence_length
		self.image_original_width = image_original_width
		self.image_original_height = image_original_height
		self.image_original_channel = image_original_channel
		self.state_action_dimension = state_action_dimension
		self.create_img = create_img
		self.create_img_prediction = create_img_prediction
		self.image_original_width = image_original_width
		self.image_original_height = image_original_height
		self.context_data_length = context_data_length

		self.csv_ref = []

		self.logger = logging.getLogger(__name__)
		self.logger.info('making final data set from raw data')

		files = glob.glob(data_dir + '/*')
		if len(files) == 0:
			self.logger.error("No files found with extensions .tfrecords in directory {0}".format(self.out_dir))
			exit()

		for file in files:
			robot_state  = np.asarray(pd.read_csv(files + '/robot_state.csv', header=None))
			proximity    = np.asarray(pd.read_csv(files + '/proximity.csv', header=None))
			xela_sensor1 = np.asarray(pd.read_csv(files + '/xela_sensor1.csv', header=None))
			xela_sensor2 = np.asarray(pd.read_csv(files + '/xela_sensor2.csv', header=None))
			xela_sensor2 = np.asarray(pd.read_csv(files + '/meta_data.csv', header=None))

			images_visualise = []
			for j in range(self.context_data_length, len(robot_state) - sequence_length):
				robot_positions__ = []
				xela_data = []
				images_labels = []
				slip_labels_sample__ = []

				image =	self.visualise_time_sequence(images_new_sample[j])
				images_visualise.append(image)
				for t in range(0, sequence_length):
					robot_positions__.append(self.convert_to_state(robot_positions_files[j+t]))  # Convert from HTM to euler task space and quaternion orientation. [[was just [t]]]]
					images_labels.append(images_new_sample[j+t+1])  # [video location, frame]
					slip_labels_sample__.append(slip_labels_sample[j+t][2])
				# self.process_data_sample(index, np.asarray([state for state in robot_positions__]), image, np.asarray(slip_labels_sample__), np.asarray(context_data), context_written, context_index, j)
				context_written = 1
				index += 1
			context_index += 1




@click.command()
@click.option('--data_set_length', type=click.INT, default=70, help='size of dataset to format.')
@click.option('--data_dir', type=click.Path(exists=True), default='/home/user/Robotics/Data_sets/slip_detection/will_dataset/data_collection_001_45/data_collection_001/', help='Directory containing data.')  # xela_validation/ /home/user/Robotics/Data_sets/data_set_003/
@click.option('--out_dir', type=click.Path(), default='/home/user/Robotics/Data_sets/manual_slip_detection/', help='Output directory of the converted data.')
@click.option('--sequence_length', type=click.INT, default=16, help='Sequence length, including context frames.')
@click.option('--image_original_width', type=click.INT, default=4, help='Original width of the images.')
@click.option('--image_original_height', type=click.INT, default=4, help='Original height of the images.')
@click.option('--image_original_channel', type=click.INT, default=3, help='Original channels amount of the images.')
@click.option('--state_action_dimension', type=click.INT, default=5, help='Dimension of the state and action.')
@click.option('--create_img', type=click.INT, default=0, help='Create the bitmap image along the numpy RGB values')
@click.option('--create_img_prediction', type=click.INT, default=0, help='Create the bitmap image used in the prediction phase')
@click.option('--context_data_length', type=click.INT, default=20, help='Size of context data length.')
def main(data_set_length, data_dir, out_dir, sequence_length, image_original_width, image_original_height, image_original_channel, state_action_dimension, create_img, create_img_prediction, context_data_length):
	data_formatter = DataFormatter(data_set_length, data_dir, out_dir, sequence_length, image_original_width, image_original_height, image_original_channel, state_action_dimension, create_img, create_img_prediction, context_data_length)

if __name__ == '__main__':
	main()







	# 	robot_positions = []
	# 	image_names = []
	# 	image_names_labels = []
	# 	slip_labels = []
	# 	frequency_rate = 10
	# 	images_for_viewing = []


	# 	# ## shuffle the experiments:
	# 	# robot_pos_files = np.asarray(robot_pos_files)
	# 	# tactile_sensor_files = np.asarray(tactile_sensor_files)
	# 	# slip_labels_files = np.asarray(slip_labels_files)

	# 	# indices = np.arange(robot_pos_files.shape[0])
	# 	# np.random.shuffle(indices)
	# 	# robot_pos_files = robot_pos_files[indices]
	# 	# tactile_sensor_files = tactile_sensor_files[indices]
	# 	# slip_labels_files = slip_labels_files[indices]

	# 	# calculate min max values for scaling:
	# 	min_max_calc = []
	# 	for i in range(1, data_set_length):
	# 		vals = np.asarray(pd.read_csv(tactile_sensor_files[i], header=None))[1:]
	# 		for val in vals:
	# 			min_max_calc.append(val)
	# 	self.min_max = self.find_min_max(min_max_calc)

	# 	index = 0
	# 	context_index = 0
	# 	for i in tqdm(range(1, data_set_length)):
	# 		i = 69 # TWISTING=69 with scale=10 || DROP=59, 61 with scale=10
	# 		images_new_sample = np.asarray(pd.read_csv(tactile_sensor_files[i], header=None))[1:]
	# 		robot_positions_new_sample = np.asarray(pd.read_csv(robot_pos_files[i], header=None))
	# 		robot_positions_files = np.asarray([robot_positions_new_sample[j*frequency_rate] for j in range(1, min(len(images_new_sample), int(len(robot_positions_new_sample)/frequency_rate)))])
	# 		images_new_sample = images_new_sample[1:len(robot_positions_files)+1]
	# 		slip_labels_sample = np.asarray(pd.read_csv(slip_labels_files[i], header=None)[1:])

	# 		context_data = []
	# 		context_written = 0
	# 		for k in range(0, self.context_data_length):  # create context data for each sample: 
	# 			context_data.append(self.create_sample(images_new_sample[k]))

	# 		self.image = []
	# 		self.base = images_new_sample[0]
	# 		images_visualise = []
	# 		for j in range(self.context_data_length, len(robot_positions_files) - sequence_length):  # 1 IGNORES THE HEADER
	# 			robot_positions__ = []
	# 			images = []
	# 			images_labels = []
	# 			slip_labels_sample__ = []

	# 			image =	self.visualise_time_sequence(images_new_sample[j])
	# 			images_visualise.append(image)
	# 			for t in range(0, sequence_length):
	# 				robot_positions__.append(self.convert_to_state(robot_positions_files[j+t]))  # Convert from HTM to euler task space and quaternion orientation. [[was just [t]]]]
	# 				images_labels.append(images_new_sample[j+t+1])  # [video location, frame]
	# 				slip_labels_sample__.append(slip_labels_sample[j+t][2])
	# 			# self.process_data_sample(index, np.asarray([state for state in robot_positions__]), image, np.asarray(slip_labels_sample__), np.asarray(context_data), context_written, context_index, j)
	# 			context_written = 1
	# 			index += 1
	# 		context_index += 1

	# 		self.run_the_tape(images_visualise)
	# 	print(aaa)
	# 	self.save_data_to_map()


	# def grab_frame(self):
	# 	print(self.indexyyy)
	# 	frame = self.images[self.indexyyy]
	# 	return frame

	# def update(self, i):
	# 	self.im1.set_data(self.grab_frame())
	# 	self.indexyyy+=1
	# 	if self.indexyyy == len(self.images):
	# 		self.indexyyy = 0

	# def run_the_tape(self, images):
	# 	self.indexyyy = 0
	# 	self.images = images
	# 	ax1 = plt.subplot(1,2,1)
	# 	self.im1 = ax1.imshow(self.grab_frame())
	# 	ani = FuncAnimation(plt.gcf(), self.update, interval=10)
	# 	plt.show()

	# def visualise_time_sequence(self, data):
	# 	width  = 300  # Width of the image
	# 	height = 300  # Height of the image
	# 	margin = 60   # Margin of the taxel in the image
	# 	scale  = 10
	# 	radius = 6

	# 	if self.image == []:
	# 		self.image = np.zeros((height,width,3), np.uint8)
	# 	else:
	# 		self.image = (self.image - 5.0).clip(min=0.0).astype(np.float32)

	# 	cv2.namedWindow('xela-sensor', cv2.WINDOW_NORMAL)

	# 	diff = np.array(data).astype(np.float32) - np.array(self.base).astype(np.float32)
	# 	diff = diff.reshape(4,4,3)
	# 	diff = diff.T.reshape(3,4,4)
	# 	dx = np.rot90((np.flip(diff[0], axis=0) / scale), k=3, axes=(0,1)).flatten()
	# 	dy = np.rot90((np.flip(diff[1], axis=0) / scale), k=3, axes=(0,1)).flatten()
	# 	dz = np.rot90((np.flip(diff[2], axis=0) / scale), k=3, axes=(0,1)).flatten()

	# 	image_positions = []
	# 	for x in range(margin, 5*margin, margin):
	# 		for y in range(margin, 5*margin, margin):
	# 			image_positions.append([y,x])

	# 	for xx, yy ,zz, image_position in zip(dx, dy, dz, image_positions):
	# 		z = radius # + (abs(xx))
	# 		x = image_position[0] + int(-xx) # int(zz)
	# 		y = image_position[1] + int(-yy) # int(yy)
	# 		# color = (255-int(z/100 * 255), 210, 255-int(z/100 * 255))  # Draw sensor circles]
	# 		color = (255,255,255)
	# 		cv2.circle(self.image, (x, y), int(z), color=color, thickness=-1)  # Draw sensor circles
	# 		gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY).astype(np.uint8)

	# 	# cv2.imshow("xela-sensor", gray)   # Display sensor image
	# 	# key = cv2.waitKey()
	# 	# cv2.destroyAllWindows()
	# 	# # key = cv2.waitKey(40)
	# 	# # if key == 27:
	# 	# # 	cv2.destroyAllWindows()
	# 	return gray


	# def create_sample(self, sample):
	# 	image = np.asarray(sample).astype(float)
	# 	image = image.reshape(self.image_original_width, self.image_original_height,3)
	# 	for x in range(0, len(image[0])):
	# 		for y in range(0, len(image[1])):
	# 			image[x][y][0] = ((image[x][y][0] - self.min_max[0]) / (self.min_max[1] - self.min_max[0]))  # Normalise normal
	# 			image[x][y][1] = ((image[x][y][1] - self.min_max[2]) / (self.min_max[3] - self.min_max[2]))  # Normalise shearx
	# 			image[x][y][2] = ((image[x][y][2] - self.min_max[4]) / (self.min_max[5] - self.min_max[4]))  # Normalise sheary

	# 	return image.astype(np.float32).flatten()

	# def find_min_max(self, imlist):
	# 	for values in imlist:
	# 		vals = np.asarray(values).astype(float).reshape(4,4,3)
	# 		try:
	# 			normal_min = min([min(vals[:, :, 0].flatten()), normal_min])
	# 			normal_max = max([max(vals[:, :, 0].flatten()), normal_max])
	# 			sheerx_min = min([min(vals[:, :, 1].flatten()), sheerx_min])
	# 			sheerx_max = max([max(vals[:, :, 1].flatten()), sheerx_max])
	# 			sheery_min = min([min(vals[:, :, 2].flatten()), sheery_min])
	# 			sheery_max = max([max(vals[:, :, 2].flatten()), sheery_max])
	# 		except:
	# 			normal_min = min(vals[:, :, 0].flatten())
	# 			normal_max = max(vals[:, :, 0].flatten())
	# 			sheerx_min = min(vals[:, :, 1].flatten())
	# 			sheerx_max = max(vals[:, :, 1].flatten())
	# 			sheery_min = min(vals[:, :, 2].flatten())
	# 			sheery_max = max(vals[:, :, 2].flatten())

	# 	return [normal_min, normal_max, sheerx_min, sheerx_max, sheery_min, sheery_max]

	# def convert_to_state(self, pose):
	# 	state = [pose[16], pose[17], pose[18], pose[-3], pose[-2], pose[-1]]
	# 	return state

	# def process_data_sample(self, index, robot_positions, image_names, slip_labels, context_data, context_written, context_index, time_step):
	# 	raw = []
	# 	tmp = image_names.astype(np.float32)
	# 	raw.append(tmp)
	# 	raw = np.array(raw)

	# 	ref = []
	# 	ref.append(index)

	# 	ref.append('')

	# 	### save np images
	# 	np.save(self.out_dir + '/image_batch_' + str(index), raw)

	# 	### save np action
	# 	np.save(self.out_dir + '/action_batch_' + str(index), robot_positions)

	# 	### save np states
	# 	np.save(self.out_dir + '/state_batch_' + str(index), robot_positions)  # original

	# 	### save np images
	# 	np.save(self.out_dir + '/slip_label_batch_' + str(index), slip_labels)

	# 	### save np context images:
	# 	if not context_written:
	# 		np.save(self.out_dir + '/context_data_' + str(context_index), context_data)

	# 	# save names for map file
	# 	ref.append('image_batch_' + str(index) + '.npy')
	# 	ref.append('action_batch_' + str(index) + '.npy')
	# 	ref.append('state_batch_' + str(index) + '.npy')
	# 	ref.append('')
	# 	ref.append('')
	# 	ref.append('slip_label_batch_' + str(index) + '.npy')
	# 	ref.append('context_data_' + str(context_index) + '.npy')
	# 	ref.append('sample_time_step_' + str(time_step))

	# 	### Append all file names for this sample to CSV file for training.
	# 	self.csv_ref.append(ref)

	# def save_data_to_map(self):
	# 	self.logger.info("Writing the results into map file '{0}'".format('map.csv'))
	# 	with open(self.out_dir + '/map.csv', 'w') as csvfile:
	# 		writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
	# 		writer.writerow(['id', 'img_bitmap_path', 'img_np_path', 'action_np_path', 'state_np_path', 'img_bitmap_pred_path', 'img_np_pred_path', 'slip_label', 'context_data', 'sample_time_step'])
	# 		for row in self.csv_ref:
	# 			writer.writerow(row)