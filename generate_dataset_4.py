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

def get_range_image(CFG,bin_file,label_names):
	semantics = True
	color_dict = CFG["color_map"]
	learning_map = CFG["learning_map"]

	nclasses = len(color_dict)
	scan = SemLaserScan(nclasses, map_dict = learning_map, sem_color_dict=color_dict,project=True )
	scan.open_scan(bin_file)	
	if semantics:
		scan.open_label(label_names)
		scan.colorize()

	#Construct tensor with x,y,z,i,d,l
	range_image = np.concatenate( (scan.proj_xyz,   \
								scan.proj_remission.reshape((scan.proj_H,scan.proj_W,1)), \
								scan.proj_range.reshape((scan.proj_H,scan.proj_W,1)), \
								scan.proj_sem_label_map.reshape((scan.proj_H,scan.proj_W,1))), \
								axis=2)
	return range_image

def visualize(range_image):
		
	# TRANSFORM RANGE IMAGE INTO COLOR MAP FOR DISPLAY
	
	ri = range_image[:,:,4]
	color_image = np.uint8(255*ri/np.max(ri))
	data = cv2.applyColorMap(color_image, cv2.COLORMAP_JET)
	

	# TRANSFORM RANGE IMAGE GRAY SCALE FOR DISPLAY
	"""
	power = 16
	data = np.copy(range_image[:,:,5])
	data[data > 0] = data[data > 0]**(1 / power)
	data[data < 0] = data[data > 0].min()
	data = (data - data[data > 0].min()) / \
		(data.max() - data[data > 0].min())
	"""
	# FOR DISPLAYING ONLY THE SEMANTICS
	#data = ri
	#data = scan.proj_sem_color[..., ::-1]

	return data




if __name__ == "__main__":
	save_path = "/media/daniel/FILES/UB/Data/Generated_Datasets/20022020_00/"
	config = "/home/daniel/Documents/SemanticKITTI/semantic-kitti-api/config/semantic-kitti.yaml"
	dataset = "/media/daniel/FILES/UB/Data/data_odometry_velodyne/dataset"
	max_sequence = 10

	# open config file
	try:
		print("Opening config file %s" % config)
		CFG = yaml.safe_load(open(config, 'r'))
	except Exception as e:
		print(e)
		print("Error opening yaml file.")
		quit()

	for s_index in range (max_sequence+1):
		sequence =  "%02d" % (s_index)

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

		for file_index in range (len(scan_names)):
			file = "%06d" % (file_index)
			save_name = save_path + "%s_%s.npy" % (sequence, file)
			print("Starting processing sequence ", sequence, " file " , file)

			range_image = get_range_image(CFG,scan_names[file_index],label_names[file_index])	
			
			np.save(save_name,range_image)
			
			################### For visualizing purposes
			"""
			data = visualize(range_image)
			cv2.imshow('my_version', data)
			c = cv2.waitKey(0)
			if 'n' == chr(c & 255):
				pass
			if 's' == chr(c & 255):
				break
			if 'q' == chr(c & 255):
				quit()
			"""
			





