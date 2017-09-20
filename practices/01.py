#!/usr/bin/env python
# -*- coding: utf-8 -*- 
import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys




def imshow(name_window,matrix):
	cv2.imshow(name_window,matrix)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def a(matrix):
#	a) inverter os valores de intensidade da imagem, tal que o valor 255 passa a ser 0, 254 passa a ser 1,
#		assim por diante;
	for i, row in enumerate(matrix):
		for j, val in enumerate(row):
			matrix[i][j] =255 - val
	
	imshow('a)',matrix)
		
	return matrix

def b(matrix):
#	b) os valores de intensidade da imagem presentes nas colunas pares são trocados com os valores de
#		intensidade das colunas ímpares;
	for i in range( matrix.shape[1]-1):
		matrix[:, i], matrix[:, i+1] = matrix[:, i+1], matrix[:, i].copy()
	imshow('b)',matrix)
	return matrix

def c(matrix):
#	c) os valores de intensidade da imagem presentes nas linhas pares são trocados com os valores de
#		intensidade das linhas ímpares;
	for i in range( matrix.shape[0]-1):
		matrix[i,:], matrix[i+1,:] = matrix[i+1,:], matrix[i,:].copy()
	imshow('c)',matrix)
	return matrix
def d(matrix):
#	d) realizar o processo de alargamento de contraste (histogram strechting)	
	fmax = matrix.max()
	fmin = matrix.min()
	gmax = 255
	gmin = 0
	for i, row in enumerate(matrix):
		for j, val in enumerate(row):
			matrix[i][j] = ( ( (gmax - gmin) / (fmax - fmin) ) * (val - fmin) ) + gmin
	imshow('d)',matrix)
	return matrix


if __name__ == '__main__':
	file_name =''
	op = None
	for i, arg in enumerate(sys.argv):
		if arg == '-t':
			op = sys.argv[i+1]
		if arg == '-i':
			file_name = sys.argv[i+1]
			break
	if op == None:
		print 'usage: python program.py -t [operation={a,b,c,d}] -i [image.png]'
		sys.exit(0)
	img = cv2.imread(file_name,cv2.CV_LOAD_IMAGE_GRAYSCALE)
	imshow('Entrada',img)
	
	if op == 'a':
		a(img)
	if op == 'b': 
		b(img)
	if op == 'c':
		c(img)
	if op == 'd':
		d(img)
	






