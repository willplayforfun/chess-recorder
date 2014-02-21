import sys
sys.path.append("opencv\\modules\\python\\src2")
import cv2
import cv
import numpy

if __name__ == "__main__":
	print "Press ESC to exit ..."

# create windows
cv.NamedWindow('Raw',   cv.CV_WINDOW_AUTOSIZE)
cv.NamedWindow('Canny', cv.CV_WINDOW_AUTOSIZE)

live_input = len(sys.argv)<2

if not live_input:
	#image specified
	frame = cv.LoadImage(sys.argv[1])

# create capture device
device = 0
print "Aquiring device %d..." % device
capture = cv.CaptureFromCAM(device)

#cv.SetCaptureProperty(capture, cv.CV_CAP_PROP_FRAME_WIDTH, 640)
#cv.SetCaptureProperty(capture, cv.CV_CAP_PROP_FRAME_HEIGHT, 480)

print "Device aquired."

# check if capture device is OK
if not capture:
	print "Error opening capture device"
	sys.exit(1)

#flag to run canny on next pass
process = False

# do forever
while 1:
	
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
	cv.ShowImage('Raw', frame)

	if process:
		process = False
		
		gray = cv.CreateImage( (frame.width,frame.height), frame.depth, 1) #frame.nChannels )
		# gray = frame
		
		#cv.ShowImage('Canny', gray)
		#continue
		
		cv.CvtColor( frame, gray, cv.CV_BGR2GRAY )
		# cv2.cvtColor( numpy.asarray(frame), cv2.COLOR_RGB2GRAY, gray );
		
		# note, the tuple is Pythonization from the cv.Size framework in C++
		# out = cv.CreateImage( (frame.width,frame.height), frame.depth, frame.nChannels );
		# 3 channels of 8 bits 
		out = cv.CreateImage( (gray.width,gray.height), cv.IPL_DEPTH_8U, 1 )
		
		# same thing about size here
		kernel_size = 9
		cv.Smooth( gray, out, cv.CV_GAUSSIAN, kernel_size, kernel_size )
		#cv2.blur( gray, (3,3), out );
		
		#Canny detector
		lowThreshold = 10
		ratio = 10
		kernel_size = 3
		cv.Canny( out, out, lowThreshold, lowThreshold*ratio, kernel_size );

		# cv.ShowImage('Canny', out)
		# continue
		
		#Using Canny's output as a mask, we display our result
		
		# cv.Scalar.all(0) just creates a scalar struct (number) of value 0
		# dst = cv.Scalar.all(0)
		#src.copyTo( dst, detected_edges)
		
		# out = beta*img1 + alpha*img2
		alpha = 1.0
		beta = 0.5
		
		multichannel_cannied = cv.CreateImage( (frame.width,frame.height), cv.IPL_DEPTH_8U, 3 )
		cv.Merge(out,out,out, None, multichannel_cannied )
		
		compound = cv.CreateImage( (frame.width,frame.height), cv.IPL_DEPTH_8U, 3 )
		cv.AddWeighted( frame, alpha, multichannel_cannied, beta, 0.0, compound);
		
		# cv.ShowImage('Canny', compound)
		cv.ShowImage('Canny', out)
		
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
