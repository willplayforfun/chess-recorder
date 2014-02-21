import sys
sys.path.append("opencv\\modules\\python\\src2")
import cv2
import cv
import numpy
import itertools
from math import sin,cos,floor,sqrt
import copy

def signum(x):
	return cmp(x,0)

GaussianKernel = 13

LowThreshold = 7
CannyRatio = 10

HoughIntersectThreshold = 100
HoughMinLen = 0
HoughMaxGap = 0

FrameMaxX=639
FrameMinX=0
FrameMaxY=479
FrameMinY=0

if __name__ == "__main__":
	print "Press ESC to exit ..."

# create windows
# cv.NamedWindow('Raw',    cv.CV_WINDOW_AUTOSIZE)
# cv.NamedWindow('Canny',  cv.CV_WINDOW_AUTOSIZE)
# cv.NamedWindow('Points', cv.CV_WINDOW_AUTOSIZE)
cv.NamedWindow('Raw',    cv.CV_WINDOW_NORMAL)
# cv.NamedWindow('Canny',  cv.CV_WINDOW_NORMAL)
cv.NamedWindow('Areas', cv.CV_WINDOW_NORMAL)
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

def change_FrameMaxX(value):
	global FrameMaxX
	FrameMaxX = value
	print "Crop X Max:",FrameMaxX
cv.CreateTrackbar("Crop X Max", "TrackBars", 639, 639, change_FrameMaxX)

def change_FrameMinX(value):
	global FrameMinX
	FrameMinX = value
	print "Crop X Min:",FrameMinX
cv.CreateTrackbar("Crop X Min", "TrackBars", 0, 639, change_FrameMinX)

def change_FrameMaxY(value):
	global FrameMaxY
	FrameMaxY = value
	print "Crop Y Max:",FrameMaxY
cv.CreateTrackbar("Crop Y Max", "TrackBars", 479, 479, change_FrameMaxY)

def change_FrameMinY(value):
	global FrameMinY
	FrameMinY = value
	print "Crop Y Min:",FrameMinY
cv.CreateTrackbar("Crop Y Min", "TrackBars", 0, 479, change_FrameMinX)

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
process = True 

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
		
		# cvSetImageROI(vid_frame, box);
		
		# FrameMaxX
		# cv.CreateMat( area_image.height, area_image.width, cv.CV_32FC1);
		
		# cropped = cv.CreateImage( (FrameMaxX-FrameMinX,FrameMaxY-FrameMinY), frame.depth, frame.nChannels)
		# for i in range(FrameMinX,FrameMaxX):
			# for j in range(FrameMinY,FrameMaxY):
				# cropped.set
		
		#TODO HEIGHT-WIDTH and X-Y indicies are probably flipped somewhere, causing errant behavior. Check Rect tuple order.
		
		# cv.SetImageROI(frame, ((FrameMaxX-FrameMinX)/2, (FrameMaxY-FrameMinY)/2, FrameMaxX-FrameMinY, FrameMaxY-FrameMinY))
		# FrameMinX=1
		# FrameMinY=1
		cv.SetImageROI(frame, (FrameMinX, FrameMinY, FrameMaxX, FrameMaxY))

		# temp = cv.CreateImage( (frame.width,frame.height), frame.depth, frame.nChannels)
		# cropped = cv.CreateImage( (FrameMaxX,FrameMaxY), frame.depth, frame.nChannels)
		cropped = cv.CreateImage( (FrameMaxX,FrameMaxY), frame.depth, frame.nChannels)

		print FrameMaxX-FrameMinX,cropped.width
		print FrameMaxY-FrameMinY,cropped.height
		print frame.depth, 		  cropped.depth
		cv.Copy(frame, cropped)
		cv.ResetImageROI(frame)

		
		gray = cv.CreateImage( (cropped.width,cropped.height), cropped.depth, 1) #frame.nChannels )
		cv.CvtColor( cropped, gray, cv.CV_BGR2GRAY )
		
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
		
		cv.ShowImage('Raw', gray)
		
		#--------------------------------------------------------------------------
		#flood fill
		area_image = cv.CreateImage( (gray.width,gray.height), cv.IPL_DEPTH_8U, 3 )
		
		#create id array
		global idarr
		idarr = [[0 for j in range(gray.width)] for i in range(gray.height)]
		#how many areas
		global numids
		numids = 0
		
		print "Non-Contiguous grouping"
		#pass 1: find non-contiguous blocks of color
		threshold = 80
		for x in range(gray.width):
			for y in range(gray.height):
				if idarr[y][x]==0:
					numids+=1
					baseline = cv.Get2D(gray, y, x)
					idarr[y][x]=numids
					for i in range(gray.width):
						for j in range(gray.height):
							pixel = cv.Get2D(gray, j, i)
							if idarr[j][i]==0 and all(abs(baseline[a] - pixel[a]) <= threshold for a in range(len(baseline))):
								idarr[j][i]=numids
		
		#pass 2: find verticies
		print "Finding verticies (Harris)"
		dst = cv.CreateMat( area_image.height, area_image.width, cv.CV_32FC1);
		src = cv.CreateMat( area_image.height, area_image.width, cv.CV_32FC1);
		cv.Convert(gray,src)

		# Detector parameters
		blockSize = 2;
		apertureSize = 3;
		k = 0.1;

		# Detecting corners
		# harris = cv.CreateImage (cv.GetSize(dst), cv.IPL_DEPTH_32F, 1)
		cv.CornerHarris( src, dst, blockSize, apertureSize, k );

		# Normalizing
		# print "Normalizing corner probabilities"
		# temp = numpy.asarray(dst)
		# cv2.normalize( numpy.asarray(dst), temp, 0, 255, cv2.NORM_MINMAX, cv.CV_32FC1 );
		# cv2.convertScaleAbs( temp, temp );
		# dst = cv.fromarray(temp)
		
		corners = []
		threshold = 255
		for j in range( dst.rows ):
			for i in range( dst.cols ):
				if cv.Get2D(dst, j, i)[0] >= threshold:
					corners.append((i,j))
		
		# pass 3: collate vertices
		temp = []
		radius = 5
		for corner in corners:
			for neighbor in corners:
				if (neighbor[0]-corner[0])**2 + (neighbor[1]-corner[1])**2 <= radius**2 and neighbor in temp:
					break
			else:
				temp.append(corner)
		corners = temp
		
		print "Visualizing Areas"
		# pass 5: visualize blocks as shades of gray
		for x in range(gray.width):
			for y in range(gray.height):
				pixel_color = 255*idarr[y][x]/numids
				cv.Circle( area_image, (x,y), 1, cv.RGB(pixel_color,pixel_color,pixel_color), 1, cv.CV_AA)
				# maybe draw text?
		
		# pass 4:
		# if two neighbors are equidistant and opposite of each other, add point to candidate list
		print "finding candidates"
		candidates = []
		slopethreshold    = 0.0003 # difference between the slopes of the two segments (multiplied by negative one)
		distancethreshold = 1 # difference between the two distances from center
		maxdistance = frame.width/4 # max distance from center
		print "max distance",maxdistance
		for corner in corners:
			temp = 0
			lines = []
			for (p1, p2) in itertools.combinations(corners,2):
				if p1==corner or p2==corner: continue
				dx1 = corner[0] - p1[0]
				dy1 = corner[1] - p1[1]
				dx2 = corner[0] - p2[0]
				dy2 = corner[1] - p2[1]
				# print signum(dx1 * dy1),",",signum(dx2 * dy2)
				if signum(dx1 * dy1) == signum(dx2 * dy2) and \
				   ( (dx1!=0 and dx2!=0 and abs((dy1/dx1) - (dy2/dx2))<=slopethreshold) or \
				     (dy1!=0 and dy2!=0 and abs((dx1/dy1) - (dx2/dy2))<=slopethreshold) ) and \
				   abs(sqrt(dx1**2 + dy1**2) - sqrt(dx2**2 + dy2**2))<=distancethreshold and \
				   sqrt(dx1**2 + dy1**2)<=maxdistance and sqrt(dx2**2 + dy2**2)<=maxdistance:
				   #or ((dy1!=0 and dy2!=0) and abs((dx1/dy1) + (dx2/dy2))<=slopethreshold)) and
					# try:
					# print str(str(dx1/dy1)+" vs "+str(dx2/dy2)) if ((dy1!=0 and dy2!=0) and (dx1==0 or dx2==0)) else str(str(dy1/dx1)+" vs "+str(dy2/dx2))
					# except:
						# pass
					temp += 1
					lines.append((p1,p2))
					
			if temp >= 4:
				candidates.append(corner)
				for (p1, p2) in lines:
					cv.Line(area_image, corner, lines[0][0], cv.RGB(0,0,255), 1, 8)
					cv.Line(area_image, lines[0][1], corner, cv.RGB(0,0,255), 1, 8)
				# print "adding",corner,"to candidates"
		
		
		
		# Drawing a circle around corners
		print "Visualizing Corners"
		for corner in corners:
			if corner in candidates:
				# cv.Circle( area_image, corner, distancethreshold,  cv.RGB(0,255,0), 1, 8, 0 ) # large circle
				cv.Circle( area_image, corner, 5,  cv.RGB(0,255,0), 1, 8, 0 )
			else:
				# cv.Circle( area_image, corner, distancethreshold,  cv.RGB(255,0,0), 1, 8, 0 ) # large circle
				cv.Circle( area_image, corner, 5,  cv.RGB(255,0,0), 1, 8, 0 )
			# cv.Circle( area_image, ( i, j ), 1,  cv.RGB(255,0,0), 1, 8, 0 ) # small dot
		
		cv.ShowImage('Areas',area_image)
		
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
		# cv.ShowImage('Raw', processed)
		
		new_width = 512
		cv.ResizeWindow('Areas', new_width, new_width * gray.height/gray.width)
		# cv.ResizeWindow('Raw', RawWindowSize, RawWindowSize * frame.height/frame.width)
		# cv.ShowImage('Canny', lined_image)
		print "Done procressing"
	else:
		# cv.ShowImage('Raw', frame)
		pass
	# process = True
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
# cv.DestroyWindow("Canny")
cv.DestroyWindow("Areas")
cv.DestroyWindow("TrackBars")
print "Loop ended. Program terminating."
