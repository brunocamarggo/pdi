# -*- coding: utf-8 -*-

import cv2
import sys
import numpy as np
# Bruno's library of pdi
import lib

#	TO RUN: python 03.py -t [a,b] -i [PATH TO YOUR IMAGE]

# Methods used in this college project are in lib.py
# 	def otsu;
# 	def thresh_truncate
# 	def thresh_binary
# 	def thresh_zero
# 	def get_4neighbors
# 	def region_growing

if __name__ == '__main__':
	
	op, filename = lib.args()
				
	img = cv2.imread(filename,cv2.CV_LOAD_IMAGE_GRAYSCALE)

	if img is None:
		print 'Error: file not found.'
		sys.exit(0)

	if op == 'a':
		threshold = lib.otsu(img)
		method = raw_input('Informe o tipo de método de limiarização: \n 1 - Limiarização Binária; \n 2 - Limiarização Truncada; \n 3 - Limiarização a zero.\n')
		if method == '1':
			g = lib.thresh_binary(img, threshold)
		elif method == '2':
			g = lib.thresh_truncate(img, threshold)
		elif method == '3':
			g = lib.thresh_zero(img, threshold)
		else:
			print('Método inválido.')
			sys.exit(0)
		
		lib.imshow('out',g)
	elif op == 'b':
		#368, 258 	
		similar = int(raw_input('Informe o valor de similaridade: '))
		num_seeds = int(raw_input('Informe o numero de seeds point: '))
		seeds_point = []
		for seed in range(num_seeds):
			seeds_point.append(tuple(int(x.strip()) 
				for x in raw_input('Informe as coord. (x,y) da seed: ').replace('(',"").replace(')','').split(',')))

		g = lib.region_growing(img, seeds_point,similar)
		lib.imshow('out',g)

	else:
		print 'Error: operation not valid.'
		sys.exit(0)
