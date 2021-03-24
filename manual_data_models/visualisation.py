# -*- coding: utf-8 -*-
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

import matplotlib
import matplotlib.animation as animation
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import mpl_toolkits.mplot3d.art3d as art3d
from scipy.ndimage.interpolation import map_coordinates

def animate_robot_positions():
	fig = plt.figure()
	ax = fig.add_subplot(221,projection='3d')
	sc = ax.scatter([],[],[], c='darkblue', marker=',' , alpha=0.5)

	def update(iteration):
		sc._offsets3d = (ee_position_x[0:iteration], ee_position_y[0:iteration], ee_position_z[0:iteration])

	ax.set_xlabel('X')
	ax.set_ylabel('Y')
	ax.set_zlabel('Z')
	ax.set_xlim(min(ee_position_x),max(ee_position_x))
	ax.set_ylim(min(ee_position_y),max(ee_position_y))
	ax.set_zlim(min(ee_position_z),max(ee_position_z))

	fps = 48 # time between each frame in milliseconds
	ani = matplotlib.animation.FuncAnimation(fig, update, frames=len(ee_position_x), interval=20.8)
	plt.tight_layout()
	plt.show()


data_dir = '/home/user/Robotics/Data_sets/slip_detection/will_dataset/data_collection_001_45/data_collection_001/'

files = glob.glob(data_dir + '/*')

robot_state  = np.asarray(pd.read_csv(files[3] + '/robot_state.csv', header=None))
proximity    = np.asarray(pd.read_csv(files[3] + '/proximity.csv', header=None))
xela_sensor1 = np.asarray(pd.read_csv(files[3] + '/xela_sensor1.csv', header=None))
xela_sensor2 = np.asarray(pd.read_csv(files[3] + '/xela_sensor2.csv', header=None))
meta_data = np.asarray(pd.read_csv(files[3] + '/meta_data.csv', header=None))

ee_positions  = []
ee_position_x = []
ee_position_y = []
ee_position_z = []

for state in robot_state[1:]:
	ee_positions.append([float(item) for item in robot_state[1][-7:-4]])
	ee_position_x.append(state[-7])
	ee_position_y.append(state[-6])
	ee_position_z.append(state[-5])

ee_position_x = np.asarray(ee_position_x).astype(float)
ee_position_y = np.asarray(ee_position_y).astype(float)
ee_position_z = np.asarray(ee_position_z).astype(float)


def visual_representation():
	# robot ee state data:
	fig = plt.figure()
	ax1 = fig.add_subplot(332 ,projection='3d')
	sc = ax1.scatter([],[],[], c='darkblue', marker=',' , alpha=0.5)

	def update(iteration):
		p = matplotlib.patches.Circle((ee_position_x[0:iteration], ee_position_y[0:iteration]), 0.01)
		ax1.add_patch(p)
		art3d.pathpatch_2d_to_3d(p, z=ee_position_z[0:iteration], zdir="z")
		sc._offsets3d = (ee_position_x[0:iteration], ee_position_y[0:iteration], ee_position_z[0:iteration])

	ax1.set_xlabel('X')
	ax1.set_ylabel('Y')
	ax1.set_zlabel('Z')
	ax1.set_xlim(min(ee_position_x),max(ee_position_x))
	ax1.set_ylim(min(ee_position_y),max(ee_position_y))
	ax1.set_zlim(min(ee_position_z),max(ee_position_z))

	fps = 48 # time between each frame in milliseconds
	anim1 = matplotlib.animation.FuncAnimation(fig, update, frames=len(ee_position_x), interval=20.8)
	# plt.tight_layout()

	# tactile data:
	xela_sensor1_data_x = []
	xela_sensor1_data_y = []
	xela_sensor1_data_z = []
	xela_sensor2_data_x = []
	xela_sensor2_data_y = []
	xela_sensor2_data_z = []

	xela_sensor1_data_x_mean = []
	xela_sensor1_data_y_mean = []
	xela_sensor1_data_z_mean = []

	xela_sensor2_data_x_mean = []
	xela_sensor2_data_y_mean = []
	xela_sensor2_data_z_mean = []

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

	for i in range(len(xela_sensor1_data_x)):
		xela_sensor1_data_x_mean.append(sum(xela_sensor1_data_x[i]) / len(xela_sensor1_data_x[i]))
		xela_sensor1_data_y_mean.append(sum(xela_sensor1_data_y[i]) / len(xela_sensor1_data_y[i]))
		xela_sensor1_data_z_mean.append(sum(xela_sensor1_data_z[i]) / len(xela_sensor1_data_z[i]))

		xela_sensor2_data_x_mean.append(sum(xela_sensor2_data_x[i]) / len(xela_sensor2_data_x[i]))
		xela_sensor2_data_y_mean.append(sum(xela_sensor2_data_y[i]) / len(xela_sensor2_data_y[i]))
		xela_sensor2_data_z_mean.append(sum(xela_sensor2_data_z[i]) / len(xela_sensor2_data_z[i]))

	def update_barx(i):
		ax2.clear()
		ax2.set_ylim(min(min(xela_sensor1_data_x_mean), min(xela_sensor2_data_x_mean)), max(max(xela_sensor1_data_x_mean), max(xela_sensor2_data_x_mean)))
		for bar in ax2.containers:
			bar.remove()
		ax2.bar([("xela_1  " + str(i)), "xela_2"], [xela_sensor1_data_x_mean[i], xela_sensor2_data_x_mean[i]], color=['blue', 'red'])
		ax2.set_title("X sheer force")
		plt.tight_layout()

	def update_bary(i):
		ax3.clear()
		ax3.set_ylim(min(min(xela_sensor1_data_y_mean), min(xela_sensor2_data_y_mean)), max(max(xela_sensor1_data_y_mean), max(xela_sensor2_data_y_mean)))
		for bar in ax3.containers:
			bar.remove()
		ax3.bar([("xela_1  " + str(i)), "xela_2"], [xela_sensor1_data_y_mean[i], xela_sensor2_data_y_mean[i]], color=['blue', 'red'])
		ax3.set_title("Y sheer force")

	def update_barz(i):
		ax4.clear()
		ax4.set_ylim(min(min(xela_sensor1_data_z_mean), min(xela_sensor2_data_z_mean)), max(max(xela_sensor1_data_z_mean), max(xela_sensor2_data_z_mean)))
		for bar in ax4.containers:
			bar.remove()
		ax4.bar([("xela_1  " + str(i)), "xela_2"], [xela_sensor1_data_z_mean[i], xela_sensor2_data_z_mean[i]], color=['blue', 'red'])
		ax4.set_title("Z sheer force")

	ax2 = fig.add_subplot(331)
	anim2 = FuncAnimation(fig=fig, func=update_barx, frames=len(ee_position_x), interval=20.8)

	ax3 = fig.add_subplot(333)
	anim3 = FuncAnimation(fig=fig, func=update_bary, frames=len(ee_position_y), interval=20.8)

	ax4 = fig.add_subplot(335)
	anim4 = FuncAnimation(fig=fig, func=update_barz, frames=len(ee_position_z), interval=20.8)



	plt.tight_layout()
	plt.show()

visual_representation()
# animate_robot_positions()