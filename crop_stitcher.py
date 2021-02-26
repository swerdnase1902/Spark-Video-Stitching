# import the necessary packages
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
from pyspark.sql import SparkSession
from PIL import Image
import io

def stitch(img_pair):
	stitcher = cv2.createStitcher() if imutils.is_cv3() else cv2.Stitcher_create()
	(status, stitched) = stitcher.stitch([img_pair[0], img_pair[1]])
	# create a 10 pixel border surrounding the stitched image
	stitched = cv2.copyMakeBorder(stitched, 10, 10, 10, 10, cv2.BORDER_CONSTANT, (0, 0, 0))
	# convert the stitched image to grayscale and threshold it
	# such that all pixels greater than zero are set to 255
	# (foreground) while all others remain 0 (background)
	gray = cv2.cvtColor(stitched, cv2.COLOR_BGR2GRAY)
	thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]
	# find all external contours in the threshold image then find
	# the *largest* contour which will be the contour/outline of
	# the stitched image
	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	c = max(cnts, key=cv2.contourArea)
	# allocate memory for the mask which will contain the
	# rectangular bounding box of the stitched image region
	mask = np.zeros(thresh.shape, dtype="uint8")
	(x, y, w, h) = cv2.boundingRect(c)
	cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
	# create two copies of the mask: one to serve as 
	# minimum rectangular region and another to serve as a counter
	# for how many pixels need to be removed to form the minimum
	# rectangular region
	minRect = mask.copy()
	sub = mask.copy()
	# keep looping until there are no non-zero pixels left in the
	# subtracted image
	while cv2.countNonZero(sub) > 0:
		# erode the minimum rectangular mask and then subtract
		# the thresholded image from the minimum rectangular mask
		# so we can count if there are any non-zero pixels left
		minRect = cv2.erode(minRect, None)
		sub = cv2.subtract(minRect, thresh)
	# find contours in the minimum rectangular mask and then
	# extract the bounding box (x, y)-coordinates
	cnts = cv2.findContours(minRect.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	c = max(cnts, key=cv2.contourArea)
	(x, y, w, h) = cv2.boundingRect(c) # width and height
	# use the bounding box coordinates to extract the our final
	# stitched image
	stitched = stitched[y:y + h, x:x + w]
	ret = (status, stitched)
	return ret

def binary_to_cv(binaryfile):
	img = np.array(Image.open(io.BytesIO(binaryfile[1])).convert('RGB'))[:, :, ::-1].copy()
	return img


if __name__ == '__main__':
	ap = argparse.ArgumentParser()
	ap.add_argument("-o", "--output", type=str, required=True,
		help="out/")
	args = vars(ap.parse_args())

	spark = SparkSession \
	    .builder \
	    .appName("PythonStitch") \
	    .getOrCreate()
	sc = spark.sparkContext
	left = sc.binaryFiles('images/left', minPartitions=sc.defaultMinPartitions)
	left = left.map(lambda img: binary_to_cv(img))
	right = sc.binaryFiles('images/right', minPartitions=sc.defaultMinPartitions)
	right = right.map(lambda img: binary_to_cv(img))
	# images = sc.binaryFiles('images', minPartitions=sc.defaultMinPartitions)
	# images = images.map(lambda img: binary_to_cv(img))

	pairs = left.zip(right)
	processed = pairs.map(lambda p: stitch(p))
	processed = processed.collect()

	for (status, stitched) in processed:
		if status==0:
			cv2.imwrite(args["output"], stitched)
			cv2.imshow('stitched image', stitched)
			cv2.waitKey(0)
	exit(0)

