from pyspark.sql import SparkSession
import numpy as np
from PIL import Image
import cv2
import io
import imutils

def stitch(img_pair):
    stitcher = cv2.createStitcher() if imutils.is_cv3() else cv2.Stitcher_create()
    ret = (status, stitched) = stitcher.stitch([img_pair[0], img_pair[1]])
    return ret

def binary_to_cv(binaryfile):
    img = np.array(Image.open(io.BytesIO(binaryfile[1])).convert('RGB'))[:, :, ::-1].copy()
    return img


if __name__ == '__main__':
    spark = SparkSession \
        .builder \
        .config("spark.driver.memory", "15g") \
        .config("spark.driver.maxResultSize", "4g") \
        .appName("PythonSimpleStitch") \
        .getOrCreate()
    sc = spark.sparkContext
    left_camera = cv2.VideoCapture('example_video/left/ny_left_short.mp4')
    right_camera = cv2.VideoCapture('example_video/right/ny_right_short.mp4')
    left_feed = []
    right_feed = []
    while(True):
        ret, frame = left_camera.read()
        if ret==True:
            left_feed.append(frame)
        else:
            break
    while (True):
        ret, frame = right_camera.read()
        if ret == True:
            right_feed.append(frame)
        else:
            break
    left_data = sc.parallelize(left_feed)
    right_data = sc.parallelize(right_feed)
    pairs = left_data.zip(right_data)
    processed = pairs.map(lambda p: stitch(p)).collect()
    for (status, img) in processed:
        if status==0:
            cv2.imshow('image', img)
            cv2.waitKey(1)
        else:
            print("Error")
    exit(0)
