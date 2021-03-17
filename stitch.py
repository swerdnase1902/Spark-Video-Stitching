import sys

from pip._vendor.distlib.compat import raw_input
from pyspark.sql import SparkSession
import numpy as np
from PIL import Image
import cv2
import io
import imutils
import math
import time

def stitch(partitions):
    stitcher = cv2.createStitcher() if imutils.is_cv3() else cv2.Stitcher_create()
    for element in partitions:
        ret = (status, stitched) = stitcher.stitch([element[0], element[1]])
        yield ret


def binary_to_cv(binaryfile):
    img = np.array(Image.open(io.BytesIO(binaryfile[1])).convert('RGB'))[:, :, ::-1].copy()
    return img


# a must be smaller than the shape
# write a into the 0,0,0 of a new shape
# so that we can write frames of different sizes to the same video
def zero_resize(a, shape):
    ret = np.zeros(shape, dtype=a.dtype)
    ret[0:a.shape[0], 0:a.shape[1], 0:a.shape[2]] = a
    return ret


def img_gen(left_camera, right_camera):
    while (True):
        l_ret, l_frame = left_camera.read()
        r_ret, r_frame = right_camera.read()
        if l_ret and r_ret:
            yield (l_frame, r_frame)
        else:
            break


if __name__ == '__main__':
    #in bytes
    start = time.perf_counter()
    video_name = sys.argv[1]
    video_len = sys.argv[2]
    MESSAGE_SIZE = 250000000
    spark = SparkSession \
        .builder \
        .config("spark.driver.memory", "15g") \
        .config("spark.driver.maxResultSize", "15g") \
        .appName("PythonSimpleStitch") \
        .getOrCreate()
    sc = spark.sparkContext
    sc.setLogLevel("INFO")
    left_camera = cv2.VideoCapture('example_video/left/{}_{}_l.mp4'.format(video_name, video_len))
    right_camera = cv2.VideoCapture('example_video/right/{}_{}_r.mp4'.format(video_name, video_len))
    FRAME_COUNT = max(left_camera.get(cv2.CAP_PROP_FRAME_COUNT),
                      right_camera.get(cv2.CAP_PROP_FRAME_COUNT))
    FRAME_WIDTH = max(left_camera.get(cv2.CAP_PROP_FRAME_WIDTH),
                      right_camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    FRAME_HEIGHT = max(left_camera.get(cv2.CAP_PROP_FRAME_HEIGHT),
                       right_camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #the two is a magic number, idk why its needed to adjust the size correctly
    numSlices = math.ceil((FRAME_COUNT * FRAME_WIDTH * FRAME_HEIGHT * 3 * 2) / MESSAGE_SIZE)
    imgs = img_gen(left_camera, right_camera)

    data = sc.parallelize(imgs, numSlices=numSlices)
    processed = data.mapPartitions(lambda p: stitch(p)).collect()

    # height, width, depth
    shape = (0, 0, 0)
    for (status, img) in processed:
        if status == 0:
            shape = tuple([max(current, new) for current, new in zip(shape, img.shape)])
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter("out.avi", fourcc, 24.0, (shape[1], shape[0]))
    for (status, img) in processed:
        if status == 0:
            # cv2.imshow("image", zero_resize(img, shape))
            writer.write(zero_resize(img, shape))
            #cv2.waitKey(1)
    writer.release()
    print("{},{},{}".format(video_name, video_len, time.perf_counter() - start))
