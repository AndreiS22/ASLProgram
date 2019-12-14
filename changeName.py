import cv2
import numpy as np
import os

num = 1


for file_type in ['source']:
	print file_type + "\n"
 	for imgurl in os.listdir(file_type):
 		if imgurl.endswith(".jpg"):
 			print imgurl