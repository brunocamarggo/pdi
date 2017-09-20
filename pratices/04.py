# -*- coding: utf-8 -*-

import cv2
import sys
import numpy as np
# Bruno's library of pdi
import lib

#	TO RUN: python 03.py -t [a,b] -i [PATH TO YOUR IMAGE]

# Methods used in this college project are in lib.py
# 	def gradient
# 	def roberts
# 	def prewitt
# 	def sobel

if __name__ == '__main__':
	
	op, filename = lib.args()
				
	img = cv2.imread(filename,cv2.CV_LOAD_IMAGE_GRAYSCALE)

	if img is None:
		print 'Error: file not found.'
		sys.exit(0)

	if op == 'a':
		threshold = int(raw_input('Informe o valor do limiar (threshold): '))
		lib.imshow('Roberts', lib.roberts(img, threshold))
	elif op == 'b':
		threshold = int(raw_input('Informe o valor do limiar (threshold): '))
		lib.imshow('Prewitt', lib.prewitt(img, threshold))
	elif op == 'c':
		threshold = int(raw_input('Informe o valor do limiar (threshold): '))
		lib.imshow('Sobel', lib.sobel(img, threshold))

	else:
		print 'Error: operation not valid.'
		sys.exit(0)
