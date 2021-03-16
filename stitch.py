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


# a must be smaller than the shape
# write a into the 0,0,0 of a new shape
# so that we can write frames of different sizes to the same video
def zero_resize(a, shape):
    ret = np.zeros(shape, dtype=a.dtype)
    ret[0:a.shape[0], 0:a.shape[1], 0:a.shape[2]] = a
    return ret

def img_gen(l_loc, r_loc):
    left_camera = cv2.VideoCapture(l_loc)
    right_camera = cv2.VideoCapture(r_loc)
    while (True):
        l_ret, l_frame = left_camera.read()
        r_ret, r_frame = right_camera.read()
        if l_ret and r_ret:
            yield (l_frame, r_frame)
        else:
            break

if __name__ == '__main__':
    spark = SparkSession \
        .builder \
        .config("spark.driver.memory", "15g") \
        .config("spark.driver.maxResultSize", "4g") \
        .appName("PythonSimpleStitch") \
        .getOrCreate()
    sc = spark.sparkContext
    imgs = img_gen('example_video/left/ny_left_short.mp4','example_video/right/ny_right_short.mp4')

    data = sc.parallelize(imgs)
    processed = data.map(lambda p: stitch(p)).collect()

    # height, width, depth
    shape = (0, 0, 0)
    for (status, img) in processed:
        if status == 0:
            shape = tuple([max(current, new) for current, new in zip(shape, img.shape)])
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter("out.avi", fourcc, 24.0, (shape[1], shape[0]))
    for (status, img) in processed:
        if status == 0:
            #cv2.imshow("image", zero_resize(img, shape))
            writer.write(zero_resize(img, shape))
            cv2.waitKey(1)
    writer.release()
    exit(0)
