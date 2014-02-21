import sys
sys.path.append("opencv\\modules\\python\\src2")
import cv2
import cv
import numpy
import itertools
from math import sin,cos,floor

if __name__ == "__main__":
	print "Press ESC to exit ..."

# create windows
# cv.NamedWindow('Raw',    cv.CV_WINDOW_AUTOSIZE)
# cv.NamedWindow('Canny',  cv.CV_WINDOW_AUTOSIZE)
# cv.NamedWindow('Points', cv.CV_WINDOW_AUTOSIZE)
cv.NamedWindow('Raw',    cv.CV_WINDOW_NORMAL)
cv.NamedWindow('Canny',  cv.CV_WINDOW_NORMAL)
cv.NamedWindow('Points', cv.CV_WINDOW_NORMAL)

live_input = len(sys.argv)<2

if not live_input:
	#image specified
	frame = cv.LoadImage(sys.argv[1])

# create capture device
device = 0
print "Aquiring device %d..." % device
capture = cv.CaptureFromCAM(device)

# cv.SetCaptureProperty(capture, cv.CV_CAP_PROP_FRAME_WIDTH, 640)
# cv.SetCaptureProperty(capture, cv.CV_CAP_PROP_FRAME_HEIGHT, 480)

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
		kernel_size = 11
		cv.Smooth( gray, out, cv.CV_GAUSSIAN, kernel_size, kernel_size )
		
		#Canny detector
		lowThreshold = 7
		ratio = 10 # highThreshold = lowThreshold * ratio
		kernel_size = 3 #size of Sobel kernel
		cv.Canny( out, out, lowThreshold, lowThreshold*ratio, kernel_size );
		
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
		
		lines = cv.HoughLines2(out, cv.CreateMemStorage(), cv.CV_HOUGH_STANDARD, 1, cv.CV_PI/180, 100, 0, 0 );

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
		
		#show points
		point_image = cv.CreateImage( (out.width,out.height), cv.IPL_DEPTH_8U, 3 )
		for pt in points:
			# cv.Circle( point_image, pt, 1, cv.RGB(255,0,0), 3, cv.CV_AA)
			cv.Circle( lined_image, pt, 1, cv.RGB(255,0,0), 3, cv.CV_AA)
		
		cv.ShowImage('Points',point_image)
		
		#------------------------------------------------------------------------------------
		
		# out = beta*img1 + alpha*img2
		alpha = 0.5
		beta = 0.5
		
		# multichannel_cannied = cv.CreateImage( (frame.width,frame.height), cv.IPL_DEPTH_8U, 3 )
		# cv.Merge(out,out,out, None, multichannel_cannied )
		
		compound = cv.CreateImage( (frame.width,frame.height), cv.IPL_DEPTH_8U, 3 )
		cv.AddWeighted( frame, alpha, lined_image, beta, 0.0, compound);
		
		cv.ShowImage('Canny', compound)
		cv.ShowImage('Raw', out)
		
		new_width = 800
		cv.ResizeWindow('Canny', new_width, new_width * frame.height/frame.width)
		new_width = 512
		cv.ResizeWindow('Raw', new_width, new_width * frame.height/frame.width)
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

cv.DestroyWindow("Raw")
cv.DestroyWindow("Canny")
print "Loop ended. Program terminating."
