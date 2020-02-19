import numpy as np 
import os
import cv2 
import time
from datetime import datetime
import pandas as pd

"""
*******************************************************
This file generates a dataset using the mapping provided by Semantic KITTI to reduce the number of classes to 19
*******************************************************
"""
learning_map= {
  0 : 0,     # "unlabeled"
  1 : 0,     # "outlier" mapped to "unlabeled" --------------------------mapped
  10: 1,     # "car"
  11: 2,     # "bicycle"
  13: 5,     # "bus" mapped to "other-vehicle" --------------------------mapped
  15: 3,     # "motorcycle"
  16: 5,     # "on-rails" mapped to "other-vehicle" ---------------------mapped
  18: 4,     # "truck"
  20: 5,     # "other-vehicle"
  30: 6,     # "person"
  31: 7,     # "bicyclist"
  32: 8,     # "motorcyclist"
  40: 9,     # "road"
  44: 10,    # "parking"
  48: 11,   # "sidewalk"
  49: 12,    # "other-ground"
  50: 13,   # "building"
  51: 14,    # "fence"
  52: 0,     # "other-structure" mapped to "unlabeled" ------------------mapped
  60: 9,     # "lane-marking" to "road" ---------------------------------mapped
  70: 15,    # "vegetation"
  71: 16,    # "trunk"
  72: 17,    # "terrain"
  80: 18,    # "pole"
  81: 19,    # "traffic-sign"
  99: 0,     # "other-object" to "unlabeled" ----------------------------mapped
  252: 1,    # "moving-car" to "car" ------------------------------------mapped
  253: 7,    # "moving-bicyclist" to "bicyclist" ------------------------mapped
  254: 6,    # "moving-person" to "person" ------------------------------mapped
  255: 8,    # "moving-motorcyclist" to "motorcyclist" ------------------mapped
  256: 5,    # "moving-on-rails" mapped to "other-vehicle" --------------mapped
  257: 5,    # "moving-bus" mapped to "other-vehicle" -------------------mapped
  258: 4,    # "moving-truck" to "truck" --------------------------------mapped
  259: 5,    # "moving-other"-vehicle to "other-vehicle" ----------------mapped
}

def load_velodyne_binary_labels(velodyne_bin_path, labels_path):
	"""Decode a binary Velodyne example (of the form '<timestamp>.bin') **** From Oxfort Robot Car
	Args:
	    example_path (AnyStr): Oxford Radar RobotCar Dataset binary Velodyne pointcloud example path
	Returns:
	    ptcld (np.ndarray): XYZI pointcloud from the binary Velodyne data Nx4
	Notes:
	    - The pre computed points are *NOT* motion compensated.
	    - Converting a raw velodyne scan to pointcloud can be done using the
	        `velodyne_ranges_intensities_angles_to_pointcloud` function.
	"""
	ext = os.path.splitext(velodyne_bin_path)[1]
	if ext != ".bin":
		raise RuntimeError("Velodyne binary pointcloud file should have `.bin` extension but had: {}".format(ext))
	if not os.path.isfile(velodyne_bin_path):
		raise FileNotFoundError("Could not find velodyne bin example: {}".format(velodyne_bin_path))
	data = np.fromfile(velodyne_bin_path, dtype=np.float32)
	ptcld = data.reshape(-1,4)
	ptcld = np.transpose(ptcld)

	#Read labels
	ext = os.path.splitext(labels_path)[1]
	if ext != ".label":
		raise RuntimeError("label file should have `.label` extension but had: {}".format(ext))
	if not os.path.isfile(velodyne_bin_path):
		raise FileNotFoundError("Could not find label file: {}".format(labels_path))

	# if all goes well, open label
	label = np.fromfile(labels_path, dtype=np.uint32)
	label= label & 0xFFFF #Following SemanticKITTI example (laser_scan.py ln:247)
	label = label.reshape(1,-1)

	label = map_labels(label)
	#ptcld = np.array([[ptcld],[labels_path]])
	#ptcld = np.concatenate((ptcld,label),axis = 0)
	return ptcld,label

def  map_labels(label):
	new_label = np.zeros(label.shape)

	for i in range (label.shape[1]):
		#print(label[0,i])
		new_label[0,i] = learning_map[label[0,i]]
	return new_label




def pc2ri_pw(pc,label):
	"""
	HDL-64E parameters by default
	"""
	############################## PARAM CONFIG ##########################################
	#vertical parameters
	v_beams=64.0
	v_FOV_degrees = np.array([-23.9,3])  #Total of 26.9 as specified in the datasheet 
	#horizontal parameters
	#h_res = 0.17578125
	h_res = 0.08
	horizontal_grids = 512.0
	h_FOV_degrees = np.array([45,135]) #degrees

	number_channels_ri = 6
	########################################################################################
	v_res = (v_FOV_degrees[1]-v_FOV_degrees[0])/v_beams
	h_res = (h_FOV_degrees[1]-h_FOV_degrees[0])/horizontal_grids

	range_image = np.zeros([int(v_beams),int(horizontal_grids),number_channels_ri])
	counter = 0

	x = pc[0,:]
	y = pc[1,:]
	z = pc[2,:]
	i = pc[3,:] 
	d = np.sqrt((x ** 2) + (y ** 2) + (z ** 2))
	angle_vertical = np.rad2deg((np.arcsin(z / d)))
	angle_azimuth = np.rad2deg(np.arctan2(x,y)) 
	#print(label.shape)

	for index in range (pc.shape[1]):

		if(h_FOV_degrees[0]<angle_azimuth[index] and h_FOV_degrees[1]>angle_azimuth[index] and (v_FOV_degrees[0])<angle_vertical[index] and (v_FOV_degrees[1])>angle_vertical[index]):

			r_index = np.floor((angle_vertical[index]+(-v_FOV_degrees[0])) / v_res).astype(int)
			if (r_index > 63):
				r_index = 63

			c_index = np.floor((angle_azimuth[index]-h_FOV_degrees[0]) / h_res).astype(int)
			if (c_index > 511):
				c_index = 511

			range_image[r_index,c_index,0:4] =  pc[:,index]
			range_image[r_index,c_index,4] =  d[index]
			range_image[r_index,c_index,5] =  label[0,index]
	range_image = np.flip(range_image,0)
	return range_image



if __name__ == "__main__":
	save_path = "/home/daniel/Documents/Generated_Datasets/All_Maped_labels/"
	file_name = save_path+"MyFile.txt"
	file1 = open(file_name,"w")

	max_num_sequence =  21
	max_num_file = 10000
	for s_index in range (max_num_sequence):
		for f_index in range (max_num_file):

			seq =  "%02d" % (s_index)
			file = "%06d" % (f_index)

			pc_path = "/media/daniel/FILES/UB/Data/data_odometry_velodyne/dataset/sequences/" + seq + "/velodyne/"+file+".bin"
			label_path = "/home/daniel/Documents/SemanticKITTI/data_odometry_labels/dataset/sequences/" + seq + "/labels/"+file+".label" 
			save_name = save_path + "%s_%s.npy" % (seq, file)

			if not os.path.isfile(pc_path):
				print("pc_path",pc_path)
				break
			if not os.path.isfile(label_path):
				print("lb_path",label_path)
				break

			print("Starting processing sequence ", seq, " file " , file)
			ptcld,label = load_velodyne_binary_labels(pc_path,label_path)
			ri = pc2ri_pw(ptcld,label)
			
			line = "%s_%s\n" % (seq, file)
			file1.write(line) 

			"""
			color_image = np.uint8(255*ri[:,:,4]/np.max(ri[:,:,4]))
			color_image = cv2.applyColorMap(color_image, cv2.COLORMAP_JET)
			cv2.imshow('my_version', color_image)
			c = cv2.waitKey(0)
			if 'q' == chr(c & 255):
				print("finish")
			"""
			np.save(save_name,ri)
	file1.close()