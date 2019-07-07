import cv2 as cv
import numpy as np
import sys

from glob import glob

def inside(r, q):
	rx, ry, rw, rh = r
	qx, qy, qw, qh = q

	return rx > qx and ry > qy and rx + rw < qx + qw and ry + rh < qy + qh

def draw_detections(image, rects, thickness=1):
	for x, y, w, h in rects:
		# the HOG detector returns slightly larger rectangles than the real objects.
		# so we slightly shrink the rectangles to get a nicer output.
		pad_w, pad_h = int(0.15 * w), int(0.05 * h)
		
		cv.rectangle(image, (x + pad_w, y + pad_h), (x + w - pad_w, y + h - pad_h), (0, 255, 0), thickness)

print(__doc__)

hog = cv.HOGDescriptor()
hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())

def test(strr):
	return strr

image_files = ['../images/pedestrians.jpg'] # ['../images/test_epic.png', '../images/test_high.png', '../images/test_medium.png', '../images/test_low.png']

for image_filename in [image_filenames[0] for image_filenames in map(glob, image_files)]:
	print(image_filename)
	
	try:
		image = cv.imread(image_filename, cv.IMREAD_UNCHANGED)
		
		if image is None:
			print('Failed to load image file:', image_filename)
			
			continue
	except:
		print('loading error')
		
		continue

	found, w = hog.detectMultiScale(image, winStride=(8, 8), padding=(32, 32), scale=1.05)
	found_filtered = []
	
	for ri, r in enumerate(found):
		for qi, q in enumerate(found):
			if ri != qi and inside(r, q):
				break
			else:
				found_filtered.append(r)
	
	draw_detections(image, found)
	draw_detections(image, found_filtered, 3)
	
	print('%d (%d) found' % (len(found_filtered), len(found)))
	
	cv.imshow('image', image)

	if cv.waitKey() == 27:
		break

cv.destroyAllWindows()
