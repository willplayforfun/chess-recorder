import sys
sys.path.append("opencv\\modules\\python\\src2")
import cv2
import cv
import numpy
import itertools
from math import sin,cos,floor

GaussianKernel = 11

LowThreshold = 7
CannyRatio = 10

HoughIntersectThreshold = 100
HoughMinLen = 0
HoughMaxGap = 0

if __name__ == "__main__":
	print "Press ESC to exit ..."

# create windows
# cv.NamedWindow('Raw',    cv.CV_WINDOW_AUTOSIZE)
# cv.NamedWindow('Canny',  cv.CV_WINDOW_AUTOSIZE)
# cv.NamedWindow('Points', cv.CV_WINDOW_AUTOSIZE)
cv.NamedWindow('Raw',    cv.CV_WINDOW_NORMAL)
# cv.NamedWindow('Canny',  cv.CV_WINDOW_NORMAL)
cv.NamedWindow('Points', cv.CV_WINDOW_NORMAL)
cv.NamedWindow('TrackBars', cv.CV_WINDOW_NORMAL)

#TRACKBARS

#gaussian

def change_GaussianKernel(value):
	global GaussianKernel
	GaussianKernel = 3 + (value//2)*2
	print "Gaussian Kernel:",GaussianKernel
cv.CreateTrackbar("Gaussian Kernel", "TrackBars", 5, 12, change_GaussianKernel)

#canny

def change_LowThreshold(value):
	global LowThreshold
	LowThreshold = value
	print "Low Threshold:",LowThreshold
cv.CreateTrackbar("Low Threshold", "TrackBars", 7, 20, change_LowThreshold)

def change_CannyRatio(value):
	global CannyRatio
	CannyRatio = value
	print "Canny Ratio:",CannyRatio
cv.CreateTrackbar("Hi/Lo Ratio", "TrackBars", 10, 30, change_CannyRatio)

#hough

def change_HoughIntersectThreshold(value):
	global HoughIntersectThreshold
	HoughIntersectThreshold = value*10
	print "Hough Intersect Threshold:",HoughIntersectThreshold
cv.CreateTrackbar("Hough Intersect Threshold", "TrackBars", 10, 20, change_HoughIntersectThreshold)

def change_HoughMinLen(value):
	global HoughMinLen
	HoughMinLen = value
	print "Hough Min Len:",HoughMinLen
cv.CreateTrackbar("Hough Min Len", "TrackBars", 0, 20, change_HoughMinLen)

def change_HoughMaxGap(value):
	global HoughMaxGap
	HoughMaxGap = value
	print "Hough Max Gap:",HoughMaxGap
cv.CreateTrackbar("Hough Max Gap", "TrackBars", 0, 20, change_HoughMaxGap)


#---------

live_input = len(sys.argv)<2

if not live_input:
	#image specified
	frame = cv.LoadImage(sys.argv[1])

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

#flag to run canny on next pass
process = False

import time
last_time = time.time()

def getLine(rho, theta):
	a = cos(theta)
	b = sin(theta)
	x0 = a*rho
	y0 = b*rho
	if b==0: return [None,x0,y0]
	return [a/b,x0,y0]

def getIntersection(line1, line2):
	
	if abs(line1[1] - cv.CV_PI/2)<0.001 and abs(line2[1] - 0)<0.001:
		return (int(line2[0]),int(line1[0]))
	if abs(line2[1] - cv.CV_PI/2)<0.001 and abs(line1[1] - 0)<0.001:
		return (int(line1[0]),int(line2[0]))
	
	temp = getLine(line1[0],line1[1])
	if not temp[0]: return None
	A1 = temp[0]
	x1 = -temp[1]
	y1 = temp[2]
	b1 = (y1 - A1*x1)
	
	temp = getLine(line2[0],line2[1])
	if not temp[0]: return None
	A2 = temp[0]
	x2 = -temp[1]
	y2 = temp[2]
	b2 = (y2 - A2*x2)
	
	#y = A*x + (y0 - A*x0)
	#y = A*x + b
	
	#parallel?
	parallelThreshold = 0.1
	if abs(A1 - A2)<parallelThreshold: return None
	
	Xa = (b2 - b1) / (A1 - A2)
	# if not A1 * Xa + b1 == A2 * Xa + b2: return None
	
	Ya = A1 * Xa + b1
	# if not Ya == A2 * Xa + b2: return None
	
	if 0 <= -Xa <= frame.width:
		return (-int(Xa), int(Ya))
	# else:
		# print "(%d, %d)" % (-Xa,Ya)
	
	return None

# do forever
while 1:
	# print(time.time()-last_time)
	# last_time = time.time()
	
	if live_input:
		# capture the current frame from webcam
			#
			# cvQueryFrame:
			# The function cvQueryFrame [...] is just a combination of GrabFrame and RetrieveFrame.
			#
			# RetrieveFrame:
			# The function cvRetrieveFrame returns the pointer to the image grabbed.
			#
		frame = cv.QueryFrame(capture)
		if frame is None:
			print "frame query returned none."
			continue
			#break

	# display the image
	# cv.ShowImage('Raw', frame)

	if process:
		process = False
		
		gray = cv.CreateImage( (frame.width,frame.height), frame.depth, 1) #frame.nChannels )
		cv.CvtColor( frame, gray, cv.CV_BGR2GRAY )
		
		# note, the tuple is Pythonization from the cv.Size framework in C++
		# out = cv.CreateImage( (frame.width,frame.height), frame.depth, frame.nChannels );
		# 3 channels of 8 bits 
		out = cv.CreateImage( (gray.width,gray.height), cv.IPL_DEPTH_8U, 1 )
		
		# same thing about size here
		cv.Smooth( gray, gray, cv.CV_GAUSSIAN, GaussianKernel, GaussianKernel )
		
		#Canny detector
		# highThreshold = lowThreshold * ratio
		kernel_size = 3 #size of Sobel kernel
		cv.Canny( gray, out, LowThreshold, LowThreshold*CannyRatio, kernel_size );
		
		#Perform probabilistic Hough line transformations
		# HoughLines2(edge_detector_output, storage?, method, radius_resolution, theta_resolution, intersection_thresholds, min_line_length, max_gap_between_points_in_line ) -> lines
		'''
		lines = cv.HoughLines2(out, cv.CreateMemStorage(), cv.CV_HOUGH_PROBABILISTIC, 1, cv.CV_PI/180, 50, 30, 4 )
		
		#add channels
		lined_image = cv.CreateImage( (out.width,out.height), cv.IPL_DEPTH_8U, 3 )
		cv.CvtColor( out, lined_image, cv.CV_GRAY2BGR )
		
		for line in lines:
			# print line
			cv.Line( lined_image, line[0], line[1], cv.RGB(0,255,0), 2)#, cv.CV_AA)
		'''
		
		lines = cv.HoughLines2(out, cv.CreateMemStorage(), cv.CV_HOUGH_STANDARD, 1, cv.CV_PI/180, HoughIntersectThreshold, HoughMinLen, HoughMaxGap );

		lined_image = cv.CreateImage( (out.width,out.height), cv.IPL_DEPTH_8U, 3 )
		cv.CvtColor( out, lined_image, cv.CV_GRAY2BGR )
		
		for line in lines:
			
			temp = getLine(line[0],line[1])
			if not temp[0]:
				#vertical line
				pt1 = (cv.Round(temp[1]), cv.Round(temp[2] + 1000))
				pt2 = (cv.Round(temp[1]), cv.Round(temp[2] - 1000))
			else:
				pt1 = (cv.Round(temp[1] + 1000), cv.Round(temp[2] - 1000*temp[0]))
				pt2 = (cv.Round(temp[1] - 1000), cv.Round(temp[2] + 1000*temp[0]))
			cv.Line( lined_image, pt1, pt2, cv.RGB(0,0,255), 3, cv.CV_AA)
			cv.Circle( lined_image, (int(temp[1]),int(temp[2])), 1, cv.RGB(0,255,0), 3, cv.CV_AA)
		
		# cv.Circle( lined_image, (5,100), 3, cv.RGB(255,0,255), 3, cv.CV_AA)
		#------------------------------------------------------------------------------------	
		#find set of intersection points
		
		points = []
		for (line1, line2) in itertools.combinations(lines,2):
			pt = getIntersection(line1, line2)
			#point draw hack
			if pt:
				points.append(pt)
				# print pt
				# cv.Circle( lined_image, pt, 1, cv.RGB(255,0,0), 3, cv.CV_AA)
		
		temp = []
		radius = 10
		for point in points:
			for neighbor in points:
				if (neighbor[0]-point[0])**2 + (neighbor[1]-point[1])**2 <= radius**2 and neighbor in temp:
					newneighbor    = [neighbor[0], neighbor[1]]
					newneighbor[0] = (neighbor[0] + point[0])/2
					newneighbor[1] = (neighbor[1] + point[1])/2
					neighbor = newneighbor
					break
			else:
				temp.append(point)
		points = temp
		
		#show points
		# point_image = cv.CreateImage( (out.width,out.height), cv.IPL_DEPTH_8U, 3 )
		for pt in points:
			cv.Circle( lined_image, pt, 1, cv.RGB(255,0,0), 3, cv.CV_AA)
		
		cv.ShowImage('Points',lined_image)
		
		#------------------------------------------------------------------------------------
		
		## out = beta*img1 + alpha*img2
		# alpha = 0.5
		# beta = 0.5
		
		## multichannel_cannied = cv.CreateImage( (frame.width,frame.height), cv.IPL_DEPTH_8U, 3 )
		## cv.Merge(out,out,out, None, multichannel_cannied )
		
		# compound = cv.CreateImage( (frame.width,frame.height), cv.IPL_DEPTH_8U, 3 )
		# cv.AddWeighted( frame, alpha, lined_image, beta, 0.0, compound);
		
		# cv.ShowImage('Canny', lined_image)
		
		processed = cv.CreateImage( (gray.width,gray.height), cv.IPL_DEPTH_8U, 3 )
		cv.CvtColor( gray, processed, cv.CV_GRAY2BGR )
		cv.ShowImage('Raw', processed)
		
		new_width = 512
		cv.ResizeWindow('Points', new_width, new_width * frame.height/frame.width)
		# cv.ResizeWindow('Raw', RawWindowSize, RawWindowSize * frame.height/frame.width)
		# cv.ShowImage('Canny', lined_image)
		
	process = True
	# handle events
	
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

# cv.DestroyWindow("Raw")
# cv.DestroyWindow("Canny")
cv.DestroyWindow("Points")
cv.DestroyWindow("TrackBars")
print "Loop ended. Program terminating."
