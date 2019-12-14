import cv2
import numpy as np
import os

num = 1

if not os.path.exists('neg'):
	os.makedirs('neg')

for file_type in ['../data/Haar/original_neg']:
	print file_type + "\n"
 	for imgurl in os.listdir(file_type):
		print imgurl
 		if imgurl.endswith(".jpg"):
 			path = '../data/Haar/original_neg/' + imgurl
			#print path
 			img = cv2.imread(path)
			img = cv2.resize(img, (200, 200))
 			gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			cv2.imwrite('../data/Haar/test_neg/'+ str(num) +'.jpg', gray)
			num += 1