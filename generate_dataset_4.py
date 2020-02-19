import numpy as np 
import cv2 
import yaml
import os
from auxiliary_SK.laserscan import LaserScan, SemLaserScan
"""
*******************************************************
This file generates a dataset using the Tools provided by SemanticKITTI
*******************************************************
"""

if __name__ == "__main__":
	# open config file
	config = "/home/daniel/Documents/SemanticKITTI/semantic-kitti-api/config/semantic-kitti.yaml"
	dataset = "/media/daniel/FILES/UB/Data/data_odometry_velodyne/dataset"
	sequence = "00"
	semantics = True
	try:
		print("Opening config file %s" % config)
		CFG = yaml.safe_load(open(config, 'r'))
	except Exception as e:
		print(e)
		print("Error opening yaml file.")
		quit()

	# List the pointcloud files 
	scan_paths = os.path.join(dataset, "sequences",
								sequence, "velodyne")
	scan_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(
		os.path.expanduser(scan_paths)) for f in fn]
	scan_names.sort()
	# List the label files
	label_paths = os.path.join(dataset, "sequences",
                                 sequence, "labels")
	label_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(
		os.path.expanduser(label_paths)) for f in fn]
	label_names.sort()

	color_dict = CFG["color_map"]
	nclasses = len(color_dict)
	scan = SemLaserScan(nclasses, color_dict, project=True)
	

	offset = 0
	scan.open_scan(scan_names[offset])	
	if semantics:
		scan.open_label(label_names[offset])
		scan.colorize()

	
	print(scan.proj_range.shape)
	ri = scan.proj_range
	color_image = np.uint8(255*ri/np.max(ri))
	color_image = cv2.applyColorMap(color_image, cv2.COLORMAP_JET)
	
	power = 16
	# now do all the range image stuff
	# plot range image
	data = np.copy(scan.proj_range)
	#print(data[data<0 and data >-1])
	
	# print(data[data > 0].max(), data[data > 0].min())
	data[data > 0] = data[data > 0]**(1 / power)
	data[data < 0] = data[data > 0].min()

	# print(data.max(), data.min())
	data = (data - data[data > 0].min()) / \
		(data.max() - data[data > 0].min())

	#data = cv2.applyColorMap(np.uint8(255*data), cv2.COLORMAP_JET)	
	data = scan.proj_inst_color[..., ::-1]
	cv2.imshow('my_version', data)
	c = cv2.waitKey(0)
	if 'q' == chr(c & 255):
		print("finish")

