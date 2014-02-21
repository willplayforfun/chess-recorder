# Visual Chess Game Move Analysis
# Written by Matt Levonian
# 2013-2014
# Python 2.7

# NOTE TO SELF:
# Upload Journals to
# 2014aganapat/web-docs/project/period6

# ===================================================================================================================================================================================
# 	IMPORTS
# ===================================================================================================================================================================================

import sys
sys.path.append("opencv\\modules\\python\\src2")
import cv2
import cv
import numpy
import itertools
from math import sin,cos,floor,sqrt
import copy
import time

# ===================================================================================================================================================================================
# 	CONSTANTS AND GLOBAL FUNCTIONS
# ===================================================================================================================================================================================

#definitions
SQUARE_SIZE = 32

#--------------------------------------------------------------------------

# this function is the meat and potatoes of the program. 
# Given an image of a single square, it determines the color of the square
def detect_square_color(img):
	
	width  = img.width
	height = img.height
	
	average_color = 0
	n = 0
	for x in range(width):
		for y in range(height):
			average_color = (average_color*n + grayscale(cv.Get2D(img,y,x)))/(n+1)
			n += 1
	
	return average_color
	
# Given an image of a single square, it determines whether a piece is on the square
def detect_piece_present(img, white):
	
	width  = img.width
	height = img.height
	
	# print img.size, canny.size
	
	piecePresent = True
	
	# note, the tuple is Pythonization from the cv.Size framework in C++
	# out = cv.CreateImage( (frame.width,frame.height), frame.depth, frame.nChannels );
	# 3 channels of 8 bits 
	canny = cv.CreateImage( (img.width,img.height), cv.IPL_DEPTH_8U, 1 )
	
	#Canny detector
	# highThreshold = lowThreshold * ratio
	kernel_size = 3 #size of Sobel kernel
	cv.Canny( img, canny, LowThreshold, HighThreshold, kernel_size );
	
	# run hough circle transform
	
	hough_input = [] #cv.CreateImage( (img.width,img.height), cv.IPL_DEPTH_8U, 1 )
	for x in range(canny.width):
		hough_input.append([])
		for y in range(canny.height):
			# if x<5 or x>width-5 or y<5 or y>height/5:
				# hough_input[x].append( 0 )
			# else:
			hough_input[x].append( cv.Get2D(img,y,x)[0] )
	
    # src_gray: Input image (grayscale)
    # circles: A vector that stores sets of 3 values: x_{c}, y_{c}, r for each detected circle.
    # CV_HOUGH_GRADIENT: Define the detection method. Currently this is the only one available in OpenCV
    # dp = 1: The inverse ratio of resolution
    # min_dist = src_gray.rows/8: Minimum distance between detected centers
    # param_1 = 200: Upper threshold for the internal Canny edge detector
    # param_2 = 100*: Threshold for center detection.
    # min_radius = 0: Minimum radio to be detected. If unknown, put zero as default.
    # max_radius = 0: Maximum radius to be detected. If unknown, put zero as default
	
	circles = cv2.HoughCircles( numpy.array(hough_input).astype('uint8'), cv2.cv.CV_HOUGH_GRADIENT, 1, width, param1=HighThreshold, param2=HoughThreshold, minRadius=width/4, maxRadius=width/2)#, 0, 0 ) 
	
	# numpy.asarray(canny[:,:]),
	
	n = 0
	border = 5
	for x in range(width):
		for y in range(height):
			if (x<width-border and x>border) and (y<height-border and y>border) and cv.Get2D(canny,y,x)[0]==255:
				# cv.Circle(img, (x,y), 1, cv.RGB(255,0,0))
				n += 1
	
	# print circles
	
	if circles!=None:
		displayimage = cv.CreateImage( (canny.width,canny.height), cv.IPL_DEPTH_8U, 3 )
		cv.CvtColor( canny, displayimage, cv.CV_GRAY2BGR );
		for circle in circles[0]:
			cv.Circle(displayimage, (circle[0],circle[1]), circle[2], cv.RGB(0,255,0))
		canny = displayimage
	else:
		displayimage = cv.CreateImage( (canny.width,canny.height), cv.IPL_DEPTH_8U, 3 )
		cv.CvtColor( canny, displayimage, cv.CV_GRAY2BGR );
		canny = displayimage
		
	piecePresent = (circles!=None) or (n > 30)
	# piecePresent = False
	
	# img = canny
	
	return piecePresent, canny

def draw_visual_aids(img, white, piece):

	width  = img.width
	height = img.height
	
	if piece:
		# pass
		cv.Circle(img, (width/2,height/2), 10, cv.RGB(255,0,0))
		
	if white:
		cv.Rectangle(img, (5,5), (width-5,height-5), cv.RGB(255,255,255))
	else:
		cv.Rectangle(img, (5,5), (width-5,height-5), cv.RGB(0,0,0))
	
	return

def signum(x):
	return cmp(x,0)
	
def grayscale(pixel_in):
	return .299*pixel_in[0] + 0.587*pixel_in[1] + 0.114*pixel_in[2]

def on_mouse(event, x, y, flag, param):
	if event == cv.CV_EVENT_LBUTTONDOWN:
		print "Click on",param
		print x,y
		
		# add to list
		global raw_box
		if len(raw_box)>=4:
			raw_box=[]
		raw_box.append((x,y))
		
		#draw markers
		global frame
		global newframe
		newframe = cv.CreateImage( (frame.width,frame.height), frame.depth, frame.nChannels )
		cv.Copy( frame, newframe )
		
		cv.Circle( newframe, (x,y), 2, cv.RGB(0,0,255), 1, cv.CV_AA)
		if len(raw_box)>1:
			cv.Line(newframe, raw_box[-2], raw_box[-1], cv.RGB(0,0,255), 1, 8)
		if len(raw_box)==4:
			cv.Line(newframe, raw_box[0], raw_box[-1], cv.RGB(0,0,255), 1, 8)
		
		# display the image
		cv.ShowImage('Raw Input', newframe)

# ===================================================================================================================================================================================
# 	TRACKBAR FUNCTIONS
# ===================================================================================================================================================================================

GaussianKernel = 13

LowThreshold = 25
HighThreshold = 30

#threshold for detecting circle centers
HoughThreshold = 12

cv.NamedWindow('TrackBars', cv.CV_WINDOW_NORMAL)

#gaussian

def change_GaussianKernel(value):
	global GaussianKernel
	GaussianKernel = 3 + (value//2)*2
	print "Gaussian Kernel:",GaussianKernel
# cv.CreateTrackbar("Gaussian Kernel", "TrackBars", 5, 12, change_GaussianKernel)

#canny

def change_LowThreshold(value):
	global LowThreshold
	LowThreshold = value
	print "Low Threshold:",LowThreshold
cv.CreateTrackbar("Low Threshold", "TrackBars", LowThreshold, 50, change_LowThreshold)

def change_HighThreshold(value):
	global HighThreshold
	HighThreshold = value
	print "High Threshold:",HighThreshold
cv.CreateTrackbar("High Threshold", "TrackBars", HighThreshold, 100, change_HighThreshold)

# hough circle

def change_HoughThreshold(value):
	global HoughThreshold
	HoughThreshold = value
	print "Hough Threshold:",HoughThreshold
cv.CreateTrackbar("Hough Threshold", "TrackBars", HoughThreshold, 50, change_HoughThreshold)

# ===================================================================================================================================================================================
# 	INITIALIZATION
# ===================================================================================================================================================================================

if __name__ == "__main__":
	print "Press ESC to exit ..."

# -- OPENCV WINDOW CREATION --

# create windows
cv.NamedWindow('Raw Input',    cv.CV_WINDOW_NORMAL)
cv.NamedWindow('Areas', cv.CV_WINDOW_NORMAL)
cv.NamedWindow('Subimage', cv.CV_WINDOW_NORMAL)
cv.NamedWindow('Canny', cv.CV_WINDOW_NORMAL)
cv.NamedWindow('Results', cv.CV_WINDOW_NORMAL)

#--------------------------------------------------------------------------

# -- USER INPUT --

# mouse input on raw image
s = "Raw Input"
cv.SetMouseCallback("Raw Input", on_mouse, param = s)

# holds the vertices from user input on raw image
raw_box = []

# global temp variable for drawing the user input lines
newframe=None

#--------------------------------------------------------------------------

# -- IMAGE LOADING --

# video feed?
live_input = len(sys.argv)<2

if not live_input:
	#image specified
	frame = cv.LoadImage(sys.argv[1])
	newframe=frame

#--------------------------------------------------------------------------
	
# -- VIDEO LOADING --

# create capture device
device = 0
print "Aquiring device %d..." % device
capture = cv.CaptureFromCAM(device)

cv.SetCaptureProperty(capture, cv.CV_CAP_PROP_FRAME_WIDTH, 640)
cv.SetCaptureProperty(capture, cv.CV_CAP_PROP_FRAME_HEIGHT, 480)

print "Device aquired."

# check if capture device is OK
if not capture:
	print "Error opening capture device"
	sys.exit(1)

#--------------------------------------------------------------------------

# -- PERSISTENT LOOP VARIABLES --

#flag to run analysis on next loop
process = False 

# for benchmarking
import time
last_time = time.time()

# ===================================================================================================================================================================================
# 	START MAIN LOOP
# ===================================================================================================================================================================================

# do forever
while 1:
	
	# -- CATCH NEXT FRAME --
	
	if live_input:
		# capture the current frame from webcam
		frame = cv.QueryFrame(capture)
		if frame is None:
			print "frame query returned none."
			continue
			#break
	
	# display the image
	cv.ShowImage('Raw Input', newframe)

	#--------------------------------------------------------------------------
	
	# === ANALYZE ===
	
	if process:
		process = False
		
		# -- CROP, GRAYSCALE, AND SMOOTH RAW IMAGE --
		
		gray = None
		
		if len(raw_box)!=4:
			gray = cv.CreateImage( (frame.width,frame.height), frame.depth, 1) #frame.nChannels )
			cv.CvtColor( frame, gray, cv.CV_BGR2GRAY )
		else:
			print "warping",raw_box
			# create the recieving image
			target_size = 512
			
			# set up parameters
			width, height = target_size, target_size
			corners = raw_box
			target = [(0,0),(target_size,0),(target_size,target_size),(0,target_size)]
			
			mat = cv.CreateMat(3, 3, cv.CV_32F)
			cv.GetPerspectiveTransform(corners, target, mat)
			out = cv.CreateMat(height, width, cv.CV_8UC3)
			cv.WarpPerspective(frame, out, mat, cv.CV_INTER_CUBIC)
			
			temp = cv.CreateImage( (target_size,target_size), 16, 3) #frame.nChannels )
			
			print temp.channels, out.channels
			print temp.depth, out.type
			arr = numpy.asarray(out)
			temp = cv.fromarray(arr)
			
			gray = cv.CreateImage( (target_size,target_size), cv.IPL_DEPTH_8U, 1) #frame.nChannels )
			for x in range(gray.width):
				for y in range(gray.height):
					pixel_in = cv.Get2D(temp,y,x)
					pixel = grayscale(pixel_in)
					cv.Circle( gray, (x,y), 1, (pixel,pixel,pixel), 1, cv.CV_AA)
		
		# same thing about size here
		cv.Smooth( gray, gray, cv.CV_GAUSSIAN, GaussianKernel, GaussianKernel )
		
		
		#--------------------------------------------------------------------------
		
		# -- RUN HARRIS CORNER DETECTION --
		
		#pass 1: find verticies
		print "Finding verticies (Harris)"
		dst = cv.CreateMat( gray.height, gray.width, cv.CV_32FC1);
		src = cv.CreateMat( gray.height, gray.width, cv.CV_32FC1);
		cv.Convert(gray,src)

		# Detector parameters
		blockSize = 2;
		apertureSize = 3;
		k = 0.1;

		# Detecting corners
		# harris = cv.CreateImage (cv.GetSize(dst), cv.IPL_DEPTH_32F, 1)
		cv.CornerHarris( src, dst, blockSize, apertureSize, k );
		
		corners = []
		threshold = 255
		for j in range( dst.rows ):
			for i in range( dst.cols ):
				if cv.Get2D(dst, j, i)[0] >= threshold:
					corners.append((i,j))
		
		# pass 2: collate vertices
		temp = []
		radius = 5
		for corner in corners:
			for neighbor in corners:
				if (neighbor[0]-corner[0])**2 + (neighbor[1]-corner[1])**2 <= radius**2 and neighbor in temp:
					break
			else:
				temp.append(corner)
		corners = temp
		
		# -- DRAW HARRIS CORNERS --
		
		processed = cv.CreateImage( (gray.width,gray.height), cv.IPL_DEPTH_8U, 3 )
		cv.CvtColor( gray, processed, cv.CV_GRAY2BGR )
		
		for corner in corners:
			cv.Circle( processed, corner, 3, (0,0,255), 1, cv.CV_AA)
		
		# -- PREDICT SQUARE CORNERS --
		
		proc_corners = []
		
		width=gray.width
		height=gray.height
		for col in range(1,8):
		
			x=col*width/8
			proc_corners.append([])
			
			for row in range(1,8):
				
				y=row*height/8
				proc_corners[col-1].append((x,y))
				
				cv.Circle( processed, (x,y), 3, (255,0,0), 1, cv.CV_AA)
		
		# -- MATCH PREDICTED AND IDENTIFIED CORNERS --
		
		found_corners = []
		
		for i in range(len(proc_corners)):
			found_corners.append([])
			for j in range(len(proc_corners[i])):
				pcorner = proc_corners[i][j]
				closest = corners[0]
				distance = width+height
				for corner in corners:
					dx = corner[0] - pcorner[0]
					dy = corner[1] - pcorner[1]
					d  = sqrt(dx**2 + dy**2)
					if d <= distance:
						closest = corner 
						distance = d
				found_corners[i].append(closest)
		
		for corner in sum(found_corners,[]):
			cv.Circle( processed, corner, 3, (0,255,0), 1, cv.CV_AA)
		
		for col in range(0,7):
			for row in range(0,6):
				cv.Line(processed, found_corners[col][row], found_corners[col][row+1], cv.RGB(0,0,255), 1, 8)
		
		for row in range(0,7):
			for col in range(0,6):
				cv.Line(processed, found_corners[col][row], found_corners[col+1][row], cv.RGB(0,0,255), 1, 8)
		
		# cv.ShowImage('Areas', processed)
		
		#--------------------------------------------------------------------------
		
		# -- BREAK BOARD INTO SQUARES --
		
		# add regular (interior) boxes
		
		# holds all of the square coordinates tuples [vertex1, vertex2
		boxes = []
		
		for i in range(len(found_corners)-1):
			for j in range(len(found_corners[i])-1):
				corner = found_corners[i][j]
				beneath = found_corners[i][j+1]
				right = found_corners[i+1][j]
				
				box = [corner, beneath, (int(right[0]),int(beneath[1])), right, i+1, j+1]
				boxes.append(box)
		
		# address boundaries
		
		corner_boxes = []
		
		# top and bottom
		for i in range(len(found_corners)-1):
			#     /   /
			#    i - o
			#   /   /
			#  o - o
			#
			
			j1=0
			j2=1
			
			left_slope  = (float(found_corners[i  ][j1][0])-float(found_corners[i  ][j2][0])) / (float(found_corners[i  ][j2][1])-float(found_corners[i  ][j1][1]))
			right_slope = (float(found_corners[i+1][j1][0])-float(found_corners[i+1][j2][0])) / (float(found_corners[i+1][j2][1])-float(found_corners[i+1][j1][1]))
			
			box = [(found_corners[i  ][0][0]+int(left_slope *float(found_corners[i  ][0][1])),0),
				   (found_corners[i+1][0][0]+int(right_slope*float(found_corners[i+1][0][1])),0),
					found_corners[i+1][0],
					found_corners[i  ][0], i+1, j1]
			print left_slope, right_slope
			print found_corners[i  ][j1], box[0]
			print found_corners[i+1][j1], box[1]
			boxes.append(copy.deepcopy(box))
			
			for ind in range(-1,3):
				cv.Line(processed, box[:4][ind], box[:4][ind+1], cv.RGB(0,0,0))
			
			if i==len(found_corners)-2:
				corner_boxes.append(box[1])
			if i==0:
				corner_boxes.append(box[0])
				
			#
			
			j1=len(found_corners[0])-2
			j2=len(found_corners[0])-1
			
			left_slope  = (float(found_corners[i  ][j1][0])-float(found_corners[i  ][j2][0])) / (float(found_corners[i  ][j2][1])-float(found_corners[i  ][j1][1]))
			right_slope = (float(found_corners[i+1][j1][0])-float(found_corners[i+1][j2][0])) / (float(found_corners[i+1][j2][1])-float(found_corners[i+1][j1][1]))
			
			box = [(found_corners[i  ][0][0]+int(left_slope *float(found_corners[i  ][0][1])),height-1),
				   (found_corners[i+1][0][0]+int(right_slope*float(found_corners[i+1][0][1])),height-1),
				    found_corners[i+1][j2],
				    found_corners[i  ][j2], i+1, j2+1]
			boxes.append(copy.deepcopy(box))
			
			for ind in range(-1,3):
				cv.Line(processed, box[:4][ind], box[:4][ind+1], cv.RGB(0,0,0))
			
			if i==len(found_corners)-2:
				corner_boxes.append(box[1])
			if i==0:
				corner_boxes.append(box[0])
		
		# left and right
		for j in range(len(found_corners[0])-1):
			#    
			#   \
			#    j
			#  \ | \
			#    o  i2
			#	  \ |
			#	   j+1
			
			i1=0
			i2=1
			
			top_slope  = (float(found_corners[i2][j  ][1])-float(found_corners[i1][j  ][1])) / (float(found_corners[i2][j  ][0])-float(found_corners[i1][j  ][0]))
			bot_slope = (float(found_corners[i2][j+1][1])-float(found_corners[i1][j+1][1])) / (float(found_corners[i2][j+1][0])-float(found_corners[i1][j+1][0]))
			
			box = [(0,found_corners[i1][j  ][1]+int(top_slope *float(found_corners[i1][j  ][1]))),
				   (0,found_corners[i1][j+1][1]+int(bot_slope*float(found_corners[i1][j+1][1]))),
					found_corners[i1][j+1],
					found_corners[i1][j  ], i1, j+1]
			print left_slope, right_slope
			print found_corners[i  ][j1], box[0]
			print found_corners[i+1][j1], box[1]
			boxes.append(copy.deepcopy(box))
			
			for ind in range(-1,3):
				cv.Line(processed, box[:4][ind], box[:4][ind+1], cv.RGB(0,0,0))
			
			if j==len(found_corners[0])-2:
				corner_boxes.append(box[1])
			if j==0:
				corner_boxes.append(box[0])
				
			#
			
			i1=len(found_corners)-1
			i2=len(found_corners)-2
			
			top_slope  = (float(found_corners[i2][j  ][1])-float(found_corners[i1][j  ][1])) / (float(found_corners[i2][j  ][0])-float(found_corners[i1][j  ][0]))
			bot_slope = (float(found_corners[i2][j+1][1])-float(found_corners[i1][j+1][1])) / (float(found_corners[i2][j+1][0])-float(found_corners[i1][j+1][0]))
			
			box = [(width-1,found_corners[i1][j  ][1]+int(top_slope *float(found_corners[i1][j  ][1]))),
				   (width-1,found_corners[i1][j+1][1]+int(bot_slope*float(found_corners[i1][j+1][1]))),
					found_corners[i1][j+1],
					found_corners[i1][j  ], i1+1, j+1]
			print left_slope, right_slope
			print found_corners[i  ][j1], box[0]
			print found_corners[i+1][j1], box[1]
			boxes.append(copy.deepcopy(box))
			
			for ind in range(-1,3):
				cv.Line(processed, box[:4][ind], box[:4][ind+1], cv.RGB(0,0,0))
			
			if j==len(found_corners[0])-2:
				corner_boxes.append(box[1])
			if j==0:
				corner_boxes.append(box[0])
		
		# address corners
		
		#	  0      2
		#	4		   5
		#
		#	6		   7
		#	  1      3
		
		box = [(0,0),corner_boxes[0],found_corners[0][0],corner_boxes[4], 0, 0]
		boxes.append(copy.deepcopy(box))
		
		for ind in range(-1,3):
			cv.Line(processed, box[:4][ind], box[:4][ind+1], cv.RGB(0,0,0))
			
		box = [(width-1,0),corner_boxes[5],found_corners[-1][0],corner_boxes[2], 8-1, 0]
		boxes.append(copy.deepcopy(box))
		
		for ind in range(-1,3):
			cv.Line(processed, box[:4][ind], box[:4][ind+1], cv.RGB(0,0,0))
			
		box = [(width-1,height-1),corner_boxes[7],found_corners[-1][-1],corner_boxes[3], 8-1, 8-1]
		boxes.append(copy.deepcopy(box))
		
		for ind in range(-1,3):
			cv.Line(processed, box[:4][ind], box[:4][ind+1], cv.RGB(0,0,0))
			
		box = [(0,height-1),corner_boxes[6],found_corners[0][-1],corner_boxes[1], 0, 8-1]
		boxes.append(copy.deepcopy(box))
		
		for ind in range(-1,3):
			cv.Line(processed, box[:4][ind], box[:4][ind+1], cv.RGB(0,0,0))
		
		# -- SAMPLE SUBREGIONS --
		
		sampled_image = cv.CreateImage( (gray.width,gray.height), cv.IPL_DEPTH_8U, 3 )
		cv.CvtColor( gray, sampled_image, cv.CV_GRAY2BGR )
		
		# holds cropped images
		# [image, board x, board y]
		grid = []
		
		for box in boxes:
			# create a subimage from each chess square
			print "creating subimage ",box
			
			# set up parameters
			width, height = SQUARE_SIZE, SQUARE_SIZE
			corners = box[:4]
			target = [(0,0),(SQUARE_SIZE,0),(SQUARE_SIZE,SQUARE_SIZE),(0,SQUARE_SIZE)]
			
			# create the recieving image
			mat = cv.CreateMat(3, 3, cv.CV_32F)
			cv.GetPerspectiveTransform(corners, target, mat)
			out = cv.CreateMat(height, width, cv.CV_8UC3)
			cv.WarpPerspective(sampled_image, out, mat, cv.CV_INTER_CUBIC)
			
			temp = cv.CreateImage( (SQUARE_SIZE,SQUARE_SIZE), 16, 1) #frame.nChannels )
			
			print temp.channels, out.channels
			print temp.depth, out.type
			arr = numpy.asarray(out)
			temp = cv.fromarray(arr)
			
			grid.append([temp,box[4],box[5]])
		
		cv.ShowImage('Areas', processed)
		
		#--------------------------------------------------------------------------
		
		# -- RUN INDIVIDUAL ANALYSIS --
		
		# avg pixel value across square
		avg_colors = [ [0 for i in range(8)] for j in range(8) ]
		
		# is the square white?
		colors = [ [False for i in range(8)] for j in range(8) ]
		
		# is there a piece there?
		piece = [ [False for i in range(8)] for j in range(8) ]
		
		width=gray.width
		height=gray.height
		
		# detect square with most pure color
		
		highest_match = 0
		match_index = [0,0]
		for img in grid:
			avg_colors[img[1]][img[2]] = detect_square_color(img[0])
			
			if highest_match < abs(127.5 - avg_colors[img[1]][img[2]]):
				highest_match = abs(127.5 - avg_colors[img[1]][img[2]])
				match_index = [img[1],img[2]]
		
		colors[match_index[0]][match_index[1]] = True if avg_colors[match_index[0]][match_index[1]] > 127.5 else False
		
		# extrapolate all square colors from single square
		
		for x in range(len(colors)):
			for y in range(len(colors[x])):
				colors[x][y] = colors[match_index[0]][match_index[1]] if not ((abs(match_index[0]-x)%2==0) ^ (abs(match_index[1]-y)%2==0)) else (not colors[match_index[0]][match_index[1]])
		
		# look for pieces in each square
		
		for img in grid:
			piece[img[1]][img[2]], temp = detect_piece_present(img[0],colors[img[1]][img[2]])
			img.append(temp)
		
		#--------------------------------------------------------------------------
		
		# -- VISUALIZE RESULTS --
		
		# draw results on subimages
		for img in grid:
			draw_visual_aids(img[0],colors[img[1]][img[2]],piece[img[1]][img[2]])
		
		# mosaic of result visualizations
		result_visual = cv.CreateImage( (SQUARE_SIZE * 8, SQUARE_SIZE * 8), cv.IPL_DEPTH_8U, 3 )
		
		for img in grid:
			# creating the bounding rectangle on output image
			roi = (SQUARE_SIZE * img[1], SQUARE_SIZE * img[2], SQUARE_SIZE, SQUARE_SIZE) # (x,y,w,h)
			cv.SetImageROI(result_visual, roi)
			
			# copy small subimage onto mosaic
			cv.Copy(img[0], result_visual)
			cv.ResetImageROI(result_visual) 
		
		# mosaic of cannys
		result_canny = cv.CreateImage( (SQUARE_SIZE * 8, SQUARE_SIZE * 8), cv.IPL_DEPTH_8U, 3 )
		
		for img in grid:
			# creating the bounding rectangle on output image
			roi = (SQUARE_SIZE * img[1], SQUARE_SIZE * img[2], SQUARE_SIZE, SQUARE_SIZE) # (x,y,w,h)
			cv.SetImageROI(result_canny, roi)
			
			# copy small subimage onto mosaic
			cv.Copy(img[3], result_canny)
			cv.ResetImageROI(result_canny) 
		
		# display mosaics
		cv.ShowImage('Results', result_visual)
		cv.ShowImage('Canny',   result_canny)
		
		# fun animation
		
		# print "Displaying subimages"
		# for image in grid:
			# cv.ShowImage('Subimage', image[3])
			# k = cv.WaitKey(100)
		# print "Done."
		
		#--------------------------------------------------------------------------
		
		# -- RESIZE DISPLAYS --
		
		# cv.ShowImage('Raw', processed)
		
		new_width = 512
		small_width = 256
		cv.ResizeWindow('Areas', new_width, new_width * processed.height/processed.width)
		cv.ResizeWindow('Input', small_width, small_width * frame.height/frame.width)
		# cv.ResizeWindow('2', small_width, small_width * gray.height/gray.width)
		# cv.ShowImage('Canny', lined_image)
		print "Done procressing"
	else:
		# cv.ShowImage('Raw', frame)
		pass
	# process = True
	# handle events
	
	# === ANALYSIS COMPLETE ===
	
	#--------------------------------------------------------------------------
	
	# -- HANDLE INPUT EVENTS --
	
	# number is milliseconds to wait (0=indefinite)
	# returns ascii code
	k = cv.WaitKey(10)
	
	if k == 0x1b: # ESC
		print 'ESC pressed. Exiting ...'
		break

	if k == 0x63 or k == 0x43: # 'c' or 'C'
		if live_input:
			print 'capturing!'
			cv.SaveImage("test.jpg",frame) 
	
	if k == 0x70 or k == 0x50:
		print "processing..."
		process=True

# ===================================================================================================================================================================================
# 	CLEANUP
# ===================================================================================================================================================================================

cv.DestroyWindow("Raw Input")
cv.DestroyWindow("Canny")
cv.DestroyWindow("Areas")
cv.DestroyWindow("TrackBars")
cv.DestroyWindow("Subimage")
cv.DestroyWindow("Results")
print "Loop ended. Program terminating."

#EOF