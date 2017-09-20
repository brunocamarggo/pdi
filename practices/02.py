# -*- coding: utf-8 -*-

import cv2
import sys
# Bruno's library of pdi
import lib

#TO RUN: python 02.py -t a -i img.png

if __name__ == '__main__':
	
	op, filename = lib.args()

	f = cv2.imread(filename,cv2.CV_LOAD_IMAGE_GRAYSCALE)

	if f is None:
		print 'Error: file not found.'
		sys.exit(0)

	if op == 'a':
		g = lib.equalize_hist(f)
		lib.imshow('Equalize Histogram',g)
		cv2.imwrite('out.png',g)

	elif op == 'b': 		
		shape = tuple(int(x.strip()) for x in raw_input('Informe o tamanho do filtro da media (m,n): ').replace('(',"").replace(')','').split(','))
		g = lib.convolution(f, lib.average_filter(shape))
		lib.imshow('Average filter',g)
		cv2.imwrite('out.png',g)

	elif op == 'c': 		
		shape = tuple(int(x.strip()) for x in raw_input('Informe o tamanho do filtro da mediana (m,n): ').replace('(',"").replace(')','').split(','))
		g = lib.median_filter(f, shape)
		lib.imshow('Median filter',g)
		cv2.imwrite('out.png',g)

	elif op == 'd': 		
		shape = tuple(int(x.strip()) for x in raw_input('Informe o tamanho do filtro gaussiano (m,n): ').replace('(',"").replace(')','').split(','))
		g = lib.convolution(f, lib.gaussian_filter(shape))
		lib.imshow('Gaussian filter',g)
		cv2.imwrite('out.png',g)
	else:
		print 'Error: operation not valid.'
		sys.exit(0)



