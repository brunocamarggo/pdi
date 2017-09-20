# -*- coding: utf-8 -*-

import cv2
import sys
import numpy as np
# Bruno's library of pdi
import lib

#	TO RUN: python 05.py -t [a,b,c,d] -i [PATH TO YOUR IMAGE]

# Methods used in this college project are in lib.py
#	def getChosenStructuringElement
# 	def dilation
# 	def erosion
# 	def opening
# 	def closing

if __name__ == '__main__':
	
	op, filename = lib.args()
				
	img = cv2.imread(filename,cv2.CV_LOAD_IMAGE_GRAYSCALE)

	if img is None:
		print 'Error: file not found.'
		sys.exit(0)

	if op == 'a':		
		strct_element = lib.getChosenStructuringElement()
		lib.imshow('Dilation', lib.dilation(img, strct_element))
	elif op == 'b':
		strct_element = lib.getChosenStructuringElement()
		lib.imshow('Erosion', lib.erosion(img, strct_element))		
	elif op == 'c':
		strct_element = lib.getChosenStructuringElement()		
		lib.imshow('Opening',  lib.opening(img, strct_element))
	elif op == 'd':
		strct_element = lib.getChosenStructuringElement()
		lib.imshow('Closing',  lib.closing(img, strct_element ))

	else:
		print 'Error: operation not valid.'
		sys.exit(0)