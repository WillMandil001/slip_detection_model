# -*- coding: utf-8 -*-
### RUN IN PYTHON 3
import os
import cv2
import csv
import glob
import click
import random
import logging
import numpy as np
import pandas as pd

from PIL import Image 
from tqdm import tqdm
from dotenv import find_dotenv, load_dotenv
from sklearn import preprocessing
from matplotlib.animation import FuncAnimation
from sklearn.decomposition import PCA
from scipy.ndimage.interpolation import map_coordinates

from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation as R

data_dir = '/home/user/Robotics/Data_sets/slip_detection/will_dataset/data_collection_200/data_collection_001/'
out_dir = '/home/user/Robotics/Data_sets/slip_detection/manual_slip_detection/'
SAVE_IMAGES= True
sequence_length = 20
image_height, image_width = 32, 32




## Load the data:
files = glob.glob(data_dir + '/*')
random.shuffle(files)
path_file = []
index_to_save = 0

xela_sensor1_data_x_final, xela_sensor1_data_y_final, xela_sensor1_data_z_final = [], [], []

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

for experiment_number in tqdm(range(len(files))):
	meta_data = np.asarray(pd.read_csv(files[experiment_number] + '/meta_data.csv', header=None))
	robot_state  = np.asarray(pd.read_csv(files[experiment_number] + '/robot_state.csv', header=None))
	proximity    = np.asarray(pd.read_csv(files[experiment_number] + '/proximity.csv', header=None))
	xela_sensor1 = np.asarray(pd.read_csv(files[experiment_number] + '/xela_sensor1.csv', header=None))
	xela_sensor2 = np.asarray(pd.read_csv(files[experiment_number] + '/xela_sensor2.csv', header=None))
	if files[experiment_number].split("/")[-1] == "data_sample_2021-03-26-11-18-41":
		file_tested = files[experiment_number]
		print(files[experiment_number])
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

		xela_sensor1_data_x_final += xela_sensor1_data_x
		xela_sensor1_data_y_final += xela_sensor1_data_y
		xela_sensor1_data_z_final += xela_sensor1_data_z

		ee_positions_final += ee_positions
		ee_position_x_final += ee_position_x
		ee_position_y_final += ee_position_y
		ee_position_z_final += ee_position_z
		ee_orientation_quat_x_final += ee_orientation_quat_x
		ee_orientation_quat_y_final += ee_orientation_quat_y
		ee_orientation_quat_z_final += ee_orientation_quat_z
		ee_orientation_quat_w_final += ee_orientation_quat_w
		ee_orientation_x_final += ee_orientation_x
		ee_orientation_y_final += ee_orientation_y
		ee_orientation_z_final += ee_orientation_z

		exp_break_points.append(exp_break_point)
		break

class Quaternion:
	"""Quaternions for 3D rotations"""
	def __init__(self, x):
		self.x = np.asarray(x, dtype=float)

	@classmethod
	def from_v_theta(cls, v, theta):
		# Construct quaternion from unit vector v and rotation angle theta
		theta = np.asarray(theta)
		v = np.asarray(v)
		
		s = np.sin(0.5 * theta)
		c = np.cos(0.5 * theta)
		vnrm = np.sqrt(np.sum(v * v))

		q = np.concatenate([[c], s * v / vnrm])
		return cls(q)

	def __repr__(self):
		return "Quaternion:\n" + self.x.__repr__()

	def __mul__(self, other):
		# multiplication of two quaternions.
		prod = self.x[:, None] * other.x

		return self.__class__([(prod[0, 0] - prod[1, 1]- prod[2, 2] - prod[3, 3]),
								(prod[0, 1] + prod[1, 0] + prod[2, 3] - prod[3, 2]),
								(prod[0, 2] - prod[1, 3] + prod[2, 0] + prod[3, 1]),
								(prod[0, 3] + prod[1, 2] - prod[2, 1] + prod[3, 0])])

	def as_v_theta(self):
		"""Return the v, theta equivalent of the (normalized) quaternion"""
		# compute theta
		norm = np.sqrt((self.x ** 2).sum(0))
		theta = 2 * np.arccos(self.x[0] / norm)

		# compute the unit vector
		v = np.array(self.x[1:], order='F', copy=True)
		v /= np.sqrt(np.sum(v ** 2, 0))

		return v, theta

	def as_rotation_matrix(self):
		"""Return the rotation matrix of the (normalized) quaternion"""
		v, theta = self.as_v_theta()
		c = np.cos(theta)
		s = np.sin(theta)

		return np.array([[v[0] * v[0] * (1. - c) + c, v[0] * v[1] * (1. - c) - v[2] * s, v[0] * v[2] * (1. - c) + v[1] * s],
						 [v[1] * v[0] * (1. - c) + v[2] * s, v[1] * v[1] * (1. - c) + c, v[1] * v[2] * (1. - c) - v[0] * s],
						 [v[2] * v[0] * (1. - c) - v[1] * s, v[2] * v[1] * (1. - c) + v[0] * s, v[2] * v[2] * (1. - c) + c]])


class CubeAxes(plt.Axes):
	"""An Axes for displaying a 3D cube"""
	# fiducial face is perpendicular to z at z=+1
	one_face = np.array([[1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1], [1, 1, 1]])

	# construct six rotators for the face
	x, y, z = np.eye(3)
	rots = [Quaternion.from_v_theta(np.eye(3)[0], theta) for theta in (np.pi / 2, -np.pi / 2)]
	rots += [Quaternion.from_v_theta(np.eye(3)[1], theta) for theta in (np.pi / 2, -np.pi / 2)]
	rots += [Quaternion.from_v_theta(np.eye(3)[1], theta) for theta in (np.pi, 0)]

	# colors of the faces
	colors = ['blue', 'green', 'pink', 'yellow', 'orange', 'red']

	def __init__(self, fig, rect=[0, 0, 1, 1], *args, **kwargs):
		# We want to set a few of the arguments
		kwargs.update(dict(xlim=(-2.5, 2.5), ylim=(-2.5, 2.5), frameon=False, xticks=[], yticks=[], aspect='equal'))
		super(CubeAxes, self).__init__(fig, rect, *args, **kwargs)
		self.xaxis.set_major_formatter(plt.NullFormatter())
		self.yaxis.set_major_formatter(plt.NullFormatter())
		
		# define the current rotation
		self.current_rot = Quaternion.from_v_theta((1, 1, 0), np.pi)

	# def animate(self, rotations):
	# 	fig.add_axes(ax)
	# 	ax.draw_cube(current_quat = rotations[0])
	# 	plt.show()

	def draw_cube(self, current_quat):
		# current_quat = [1.0, 0.0, 0.0, 0.0]
		self.current_rot = Quaternion.from_v_theta((current_quat[0], current_quat[1], current_quat[2]), current_quat[3])

		"""draw a cube rotated by theta around the given vector"""
		# rotate the six faces
		Rs = [(self.current_rot * rot).as_rotation_matrix() for rot in self.rots]
		faces = [np.dot(self.one_face, R.T) for R in Rs]

		# project the faces: we'll use the z coordinate for the z-order
		faces_proj = [face[:, :2] for face in faces]
		zorder = [face[:4, 2].sum() for face in faces]

		# create the polygons if needed. if they're already drawn, then update them
		if not hasattr(self, '_polys'):
			self._polys = [plt.Polygon(faces_proj[i], fc=self.colors[i],alpha=0.7, zorder=zorder[i]) for i in range(6)]
			for i in range(6):
				self.add_patch(self._polys[i])
		else:
			for i in range(6):
				self._polys[i].set_xy(faces_proj[i])
				self._polys[i].set_zorder(zorder[i])

		self.figure.canvas.draw()
		return plt

# plot the quaternion:
image_array = []
for i in range(len(ee_orientation_quat_x_final)):
	fig = plt.figure(figsize=(4, 4))
	ax = CubeAxes(fig)
	fig.add_axes(ax)
	plt = ax.draw_cube(current_quat=np.array([ee_orientation_quat_x_final[i],
										ee_orientation_quat_y_final[i],
										ee_orientation_quat_z_final[i],
										ee_orientation_quat_w_final[i]]).astype(np.float32))

	w, h = fig.canvas.get_width_height()
	image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8').reshape(h, w, 3)
	image_array.append(image)
	plt.close()


class image_player():
	def __init__(self, images):
		self.run_the_tape(images)

	def grab_frame(self):
		print(self.indexyyy,  end="\r")
		frame = self.images[self.indexyyy]
		return frame

	def update(self, i):
		print(ee_orientation_quat_x_final[self.indexyyy])
		plt.title("data_sample_2021-03-26-11-18-41  time_step: " + str(self.indexyyy))
		self.im1.set_data(self.grab_frame())
		self.indexyyy+=1
		if self.indexyyy == len(self.images):
			self.indexyyy = 0

	def run_the_tape(self, images):
		self.indexyyy = 0
		self.images = images
		ax1 = plt.subplot(1,2,1)
		self.im1 = ax1.imshow(self.grab_frame())
		ani = FuncAnimation(plt.gcf(), self.update, interval=20.8, save_count=len(self.images))
		# ani.save(str(file_tested) + '/orientation_quaternion.gif')
		plt.show()

image_player(image_array)

# images = []
# for file in sorted(glob.glob('test/*'), key=os.path.getmtime):
# 	image = plt.imread('/home/user/Robotics/slip_detection_model/slip_detection_model/manual_data_models/' + file)
# 	images.append(image)

# for i in range(len(ee_orientation_z_final)):
# 	print(ee_orientation_z_final[i])
# 	if ee_orientation_z_final[i] < 0:
# 		ee_orientation_z_final[i] += 360

# plt.plot([i for i in range(len(ee_orientation_x_final))], [float(i) for i in ee_orientation_x_final])
# plt.plot([i for i in range(len(ee_orientation_y_final))], [float(i) for i in ee_orientation_y_final])
# plt.plot([i for i in range(len(ee_orientation_z_final))], [float(i) for i in ee_orientation_z_final])
# plt.show()

