import cv2
import numpy as np
from optparse import OptionParser
import sys


# Method used to read the input by the user
def args():
	parser = OptionParser()
	parser.add_option("-i", dest="filename",
                  help="image")

	parser.add_option("-t", dest="op",
                  help="operation")

	(options, args) = parser.parse_args()
	return options.op, options.filename

# Shows a image using cv2 library
def imshow(name_window,matrix):
	cv2.imshow(name_window,matrix)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

# Just a method to show a matrix (m) in the console
def show_matrix(m):
	for row in m:
		for val in row:
			print val,
		print
	print

# Do the convolution in a image (img) using a filter (mask). Is possible to
# use a average or a gaussian filter.
def convolution(img,mask):
	g = np.zeros(shape=img.shape,dtype='uint8')
	x1 = mask.shape[0]/2
	y1 = mask.shape[1]/2
	for x in range(img.shape[0]-mask.shape[0]):
		for y in range(img.shape[1]-mask.shape[1]):	
			soma = 0.0
			for i in range(mask.shape[0]):
				for j in range(mask.shape[1]):
					soma = int(round(soma + ( mask[i][j] * img[x+i][y+j] )))
			g[x+x1][y+y1] = soma		
	return g


def average_filter(maskshape):
	mask = np.ones(maskshape)
	t = sum(sum(mask))
	for i, row in enumerate(mask):
		for j, val in enumerate(row):
			mask[i][j] = val/t
	return mask

# Returns the pascals triangle with the size of number of rows (n_rows)
def pascals_triangle(n_rows):
	results = [] 
	for _ in range(n_rows): 
	    row = [1]
	    if results:
	        last_row = results[-1]
	        row.extend([sum(pair) for pair in zip(last_row, last_row[1:])])
	        row.append(1)
	    results.append(row)
	return results

def gaussian_filter(maskshape):   
	pa = pascals_triangle(maskshape[0])
	# last line of pascals triangle
	line = pa[len(pa)-1] 
	linelize = len(line)
	out = g = np.zeros(shape=(linelize,linelize))
	for i in range(linelize):
		for j in range(linelize):
			out[i][j] = (line[i]*line[j])
	t = sum(sum(out))
	for i in range(linelize):
		for j in range(linelize):
			out[i][j] = out[i][j]/t
	return out

# Do the convolution in a image (img) using a median filter.
def median_filter(img,maskshape):
	g = np.zeros(shape=img.shape,dtype='uint8')
	x1 = maskshape[0]/2
	y1 = maskshape[1]/2
	for x in range(img.shape[0]-maskshape[0]):
		for y in range(img.shape[1]-maskshape[1]):	
			l = []
			for i in range(maskshape[0]):
				for j in range(maskshape[1]):
					l.append(img[x+i][y+j])
			l.sort()
			g[x+x1][y+y1] = l[len(l)/2]		
	return g

# return the histogram for a image (img)
def histogram(img):
	h = [0] * 256
	for i, row in enumerate(img):
		for j, val in enumerate(row):
			h[val] = h[val] + 1

	return h

# return the equalized histogram for a image (img)
def equalize_hist(img):
	hist = histogram(img)

	histsize = len(hist)
	eq_hist = [0.0] * histsize
	t = img.shape[0] * img.shape[1]
	hist_sum = [0] * histsize
	
	hist_sum[0] = hist[0]

	i = 1
	for val in hist[1:]:
		hist_sum[i] = hist_sum[i-1] + val
		i = i + 1

	for i, val in enumerate(hist_sum):
		hist_sum[i] = hist_sum[i] * 255 / t

	g = np.zeros(shape=img.shape, dtype='uint8')
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			g[i][j] = hist_sum[img[i][j]]
	return g

# This method (oust) was implemented with collaboration of	
# Regis Rufino Rodrigues.
def otsu(img):
	hist = histogram(img)
	threshold = 0
	sumA = 0.0
	sumB = 0.0
	q2 = 0.0
	N = img.shape[0]*img.shape[1]
	varMax = 0.0
	q1 = 0.0
	for i in range(len(hist)):
		sumA  = sumA + i * hist[i]
	
	for i in range(len(hist)):
		q1 = q1 + hist[i]
		q2 = N - q1
		if q2 == 0:
			q2 = 0.000000000000001
		sumB = sumB + i * hist[i]
		if q1 == 0:
			q1 = 0.000000000000001
		u1 = sumB/q1
		u2 = (sumA - sumB)/q2

		varSquared = q1 * q2 * (u1-u2) * (u1-u2)

		if varSquared > varMax:
			threshold = i
			varMax = varSquared

	return threshold


def thresh_truncate(img, threshold):
	g = np.zeros(img.shape, dtype='uint8')
	#truncade
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			if img[i][j] > threshold:
				g[i][j] = threshold
			else:
				g[i][j] = 0
	return g


def thresh_binary(img, threshold):
	g = np.zeros(img.shape, dtype='uint8')
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			if img[i][j] > threshold:
				g[i][j] = 255
			else:
				g[i][j] = 0
	return g

def thresh_zero(img, threshold):
	g = np.zeros(img.shape, dtype='uint8')
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			if img[i][j] > threshold:
				g[i][j] = img[i][j]
			else:
				g[i][j] = 0
	return g

def get_4neighbors(point,img):
	max_l = img.shape[0]
	max_c = img.shape[1]
	l = point[0]
	c = point[1]
	nb = []
	if c+1 < max_c:
		nb.append( (l,c+1) )
	if c-1 >= 0:
		nb.append( (l,c-1) )
	if l-1 >= 0:
		nb.append( (l-1,c ))
	if l+1 < max_l:
		nb.append( (l+1,c ))
	return nb

def region_growing(img, seeds_point, similar):
	g = np.zeros(img.shape, dtype='uint8')
	print seeds_point
	for seed_point in seeds_point:
		visited = np.zeros( img.shape, dtype=bool)
		visited[seed_point] = True
		point_queue = []
		point_queue.append(seed_point)
		while point_queue:
			this_point = point_queue.pop(0)

			for neighbour in get_4neighbors(this_point, img):
				if not visited[neighbour] and abs( int(img[neighbour]) - int(img[seed_point]) ) <= similar:
					point_queue.append(neighbour)
					visited[neighbour] = True
					g[neighbour] = img[neighbour]
	return g

def gradient(partial1, partial2, threshold):
	out = np.zeros(partial1.shape, dtype='uint8')
	for i in range(partial1.shape[0]):
		for j in range(partial1.shape[1]):
			out[i][j] = abs(int(partial1[i][j])+int(partial2[i][j]))
			if out[i][j] > 255:
				out[i][j] = 255
			if out[i][j] >= threshold:
				out[i][j] = 255
			else:
				out[i][j] = 0
	return out
def roberts(img, threshold):
	mask = np.array([
		[1,0],
		[0,-1]	
	])
	partial1 = convolution(img,mask)
	mask = np.array([
		[0,-1],
		[1,0]	
		])
	partial2 = convolution(img,mask)
	return gradient(partial1, partial2, threshold)
	

def prewitt(img, threshold):
	mask = np.array([
		[-1,0,1],
		[-1,0,1],
		[-1,0,1],	
	])
	partial1 = convolution(img,mask)
	mask = np.array([
		[-1,-1,-1],
		[0,0,0],
		[1,1,1],	
	])
	partial2 = convolution(img,mask)

	return gradient(partial1, partial2, threshold)

def sobel(img, threshold):
	mask = np.array([
		[-1,0,1],
		[-2,0,2],
		[-1,0,1],	
	])
	partial1 = convolution(img,mask)
	mask = np.array([
		[-1,-2,-1],
		[0,0,0],
		[1,2,1],	
	])
	partial2 = convolution(img,mask)

	return gradient(partial1, partial2, threshold)

def erosion(img, strct_element):
	m , n = strct_element.shape[0]/2,strct_element.shape[1]/2
	out = np.zeros(img.shape, dtype='uint8')
	img = np.lib.pad(img,((m,n),(m,n)),'constant')	

	for i in range(img.shape[0]-(2*m)):
		for j in range(img.shape[1]-(2*n)):
			aux = np.zeros(strct_element.shape)
			for k in range(strct_element.shape[0]):
				for l in range(strct_element.shape[1]):
					aux[k,l] = img[i+k,j+l]
			out[i,j] = (aux - strct_element).min()
				
	out = out -255

	return out

def dilation(img, strct_element):
	m , n = strct_element.shape[0]/2,strct_element.shape[1]/2
	out = np.zeros(img.shape, dtype='uint8')
	img = np.lib.pad(img,((m,n),(m,n)),'constant')	

	for i in range(img.shape[0]-(2*m)):
		for j in range(img.shape[1]-(2*n)):
			aux = np.zeros(strct_element.shape)
			for k in range(strct_element.shape[0]):
				for l in range(strct_element.shape[1]):
					aux[k,l] = img[i+k,j+l]
			out[i,j] = sum(sum(np.logical_and(strct_element,aux)))
			if out[i,j]  != 0:
				out[i,j] = 255;	


	return out

def opening(img, strct_element):
	img_erosion = erosion(img, strct_element)
	return dilation(img_erosion, strct_element)

def closing(img, strct_element):
	img_dilation = dilation(img, strct_element)
	return erosion(img_dilation, strct_element)

def getStructuringElement(option='rectangular'):
	
	if option == 'rectangular':
		print 'Using rectangular 5x5 structuring element'
		return np.array([
				[1, 1, 1, 1, 1],
	       		[1, 1, 1, 1, 1],
	       		[1, 1, 1, 1, 1],
	       		[1, 1, 1, 1, 1],
	       		[1, 1, 1, 1, 1]], dtype='uint8')

	elif option == 'elliptical':
		print 'Using elliptical 5x5 structuring element'
		return np.array([
				[0, 0, 1, 0, 0],
       			[1, 1, 1, 1, 1],
       			[1, 1, 1, 1, 1],
       			[1, 1, 1, 1, 1],
       			[0, 0, 1, 0, 0]], dtype='uint8')

	elif option == 'cross-shaped':
		print 'Using cross-shaped 5x5 structuring element'
		return np.array([
				[0, 0, 1, 0, 0],
       			[0, 0, 1, 0, 0],
       			[1, 1, 1, 1, 1],
       			[0, 0, 1, 0, 0],
       			[0, 0, 1, 0, 0]], dtype='uint8')

def getChosenStructuringElement():
	option = raw_input('Defina o elemento estruturante: \n a -> rectangular \n b -> elliptical \n c -> cross-shaped:\n')
	if option == 'a':
		return getStructuringElement('rectangular')
	elif option == 'b':
		return getStructuringElement('elliptical')
	elif option == 'c':
		return getStructuringElement('cross-shaped')
	else:
		return getStructuringElement()