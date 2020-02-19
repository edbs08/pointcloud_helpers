import numpy as np 
import cv2 
import time
import sys

#np.set_printoptions(threshold=sys.maxsize)
#pc = np.load('lidar_2d/2011_09_26_0001_0000000000.npy')
#pc = np.load('lidar_2d/2011_09_26_0001_0000000001.npy')
color_map = {
0 : [0, 0, 0],
1 : [0, 0, 255],
10: [245, 150, 100],
11: [245, 230, 100],
13: [250, 80, 100],
15: [150, 60, 30],
16: [255, 0, 0],
18: [180, 30, 80],
20: [255, 0, 0],
30: [30, 30, 255],
31: [200, 40, 255],
32: [90, 30, 150],
40: [255, 0, 255],
44: [255, 150, 255],
48: [75, 0, 75],
49: [75, 0, 175],
50: [0, 200, 255],
51: [50, 120, 255],
52: [0, 150, 255],
60: [170, 255, 150],
70: [0, 175, 0],
71: [0, 60, 135],
72: [80, 240, 150],
80: [150, 240, 255],
81: [0, 0, 255],
99: [255, 255, 50],
252: [245, 150, 100],
256: [255, 0, 0],
253: [200, 40, 255],
254: [30, 30, 255],
255: [90, 30, 150],
257: [250, 80, 100],
258: [180, 30, 80],
259: [255, 0, 0],
}

small_color_map = {
0 : [0, 0, 0],
1 : [0, 0, 255],
2: [245, 150, 100],
3: [245, 230, 100],
}

use_color_map_SemanticKITTI = True

def comprate_ruttine():
	path_A = np.array(["/media/daniel/FILES/UB/Data/Generated_Datasets/All_labels/00_","/media/daniel/FILES/UB/Data/Generated_Datasets/All_labels/01_"])
	img_color_A = np.zeros((64,512,3),dtype=np.uint8)

	path_B = np.array(["/home/daniel/Documents/LU_Net_Original/lidar_2d/2011_09_26_0001_","/home/daniel/Documents/LU_Net_Original/lidar_2d/2011_09_26_0020_"])
	img_color_B = np.zeros((64,512,3),dtype=np.uint8)

	for ii in range(2):
		print(path_A[ii])
		print(path_B[ii])
		for i in range(50):
			nameA = "%06d.npy" % i
			nameA = path_A[ii]+nameA
			pc = np.load(nameA)
			labels_ri = pc[:,:,5]
			for r in range (labels_ri.shape[0]):
				for c in range (labels_ri.shape[1]):
					#print(img_color_A[r,c,:])
					#print(labels_ri[r,c])
					#print(color_map[labels_ri[r,c]])
					img_color_A[r,c,:] =  color_map[labels_ri[r,c]]



			nameB = "%010d.npy" % i
			nameB = path_B[ii]+nameB
			pc = np.load(nameB)
			labels_ri = pc[:,:,5]
			for r in range (labels_ri.shape[0]):
				for c in range (labels_ri.shape[1]):
					#print (img_color_A[r,c,:])
					#print(labels_ri[r,c])
					#print(color_map[labels_ri[r,c]])
					img_color_B[r,c,:] =  small_color_map[labels_ri[r,c]]

			cv2.namedWindow("image Semantic KITTI")
			cv2.imshow('image Semantic KITTI', img_color_A)


			cv2.namedWindow("image SqueezeSeg")
			cv2.imshow('image SqueezeSeg', img_color_B)
			#cv2.imshow('image', rgb_img)
			c = cv2.waitKey(0)
			if 'q' == chr(c & 255):
				#print("finish")
				break


def normal_display():
	for i in range(1):
		###############
		#name = "lidar_2d/2011_09_26_0001_%010d.npy" % i
		#pc = np.load('lidar_2d/2011_09_26_0070_0000000336.npy')
		#print(name)
		###############
		name = "/media/daniel/FILES/UB/Data/Generated_Datasets/All_labels/00_000000.npy"
		##############

		
		pc = np.load(name)
		#print(pc.shape)
		labels_ri = pc[:,:,5]
		img_color = np.zeros((labels_ri.shape[0],labels_ri.shape[1],3),dtype=np.uint8)
		if (use_color_map_SemanticKITTI == True):
			for r in range (labels_ri.shape[0]):
				for c in range (labels_ri.shape[1]):
					img_color[r,c,:] =  color_map[labels_ri[r,c]]
			
		else:
			img = pc[:,:,5]
			img = img / np.amax(img)
			img = 255*img
			img = img.astype(np.uint8)
			img_color = img_color.astype(np.uint8)
			img_color[:,:,0] = img
			img_color[:,:,1] = 255
			img_color[:,:,2] = 255
			img_color = cv2.cvtColor(img_color,cv2.COLOR_HSV2BGR)
			img_color.astype(np.uint8)
			
		cv2.namedWindow("image")
		cv2.imshow('image', img_color)
		#cv2.imshow('image', rgb_img)
		c = cv2.waitKey(0)
		if 'q' == chr(c & 255):
			print("finish")
			#do nothing

		#count number of classes
		print("******************** ")
		print(np.amax(labels_ri))

def visual_data_function(range_img):

	pc = range_img
	#pc = np.load('lidar_2d/2011_09_26_0070_0000000336.npy')
	img = pc[:,:,5]
	img = img / np.amax(img)
	img = 255*img
	img = img.astype(np.uint8)

	img_color = np.zeros((img.shape[0],img.shape[1],3),dtype=np.uint8)
	img_color = img_color.astype(np.uint8)
	img_color[:,:,0] = img
	img_color[:,:,1] = 255
	img_color[:,:,2] = 255

	rgb_img = cv2.cvtColor(img_color,cv2.COLOR_HSV2BGR)
	img_color.astype(np.uint8)
	#print(img_color.shape)
	
	cv2.namedWindow("image")
	#cv2.imshow('image', img_color)
	cv2.imshow('image', rgb_img)
	c = cv2.waitKey(0)
	if 'q' == chr(c & 255):
		print("finish")
	time.sleep(5)

if __name__ == "__main__":
	#normal_display()
	comprate_ruttine()