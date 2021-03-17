import sys
import time
import cv2
import imutils
import numpy as np


def stitch(element, st):
    ret = (status, stitched) = stitcher.stitch([element[0], element[1]])
    return ret

# a must be smaller than the shape
# write a into the 0,0,0 of a new shape
# so that we can write frames of different sizes to the same video
def zero_resize(a, shape):
    ret = np.zeros(shape, dtype=a.dtype)
    ret[0:a.shape[0], 0:a.shape[1], 0:a.shape[2]] = a
    return ret


if __name__ == '__main__':
    #in bytes
    start = time.perf_counter()
    video_name = sys.argv[1]
    video_len = sys.argv[2]

    left_camera = cv2.VideoCapture('example_video/left/{}_{}_l.mp4'.format(video_name, video_len))
    right_camera = cv2.VideoCapture('example_video/right/{}_{}_r.mp4'.format(video_name, video_len))
    FPS = max(left_camera.get(cv2.CAP_PROP_FPS),
                      right_camera.get(cv2.CAP_PROP_FPS))
    #print("Making imgs")
    processed = []
    stitcher = cv2.createStitcher() if imutils.is_cv3() else cv2.Stitcher_create()
    while (True):
        l_ret, l_frame = left_camera.read()
        r_ret, r_frame = right_camera.read()
        if l_ret and r_ret:
            processed.append(stitcher.stitch([l_frame, r_frame]))
        else:
            break

    #print("writing")
    # height, width, depth
    shape = (0, 0, 0)
    for (status, img) in processed:
        if status == 0:
            shape = tuple([max(current, new) for current, new in zip(shape, img.shape)])
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter("out.avi", fourcc, FPS, (shape[1], shape[0]))
    for (status, img) in processed:
        if status == 0:
            #cv2.imshow("image", zero_resize(img, shape))
            writer.write(zero_resize(img, shape))
            #cv2.waitKey(1)
    writer.release()
    print("{},{},{}".format(video_name, video_len, time.perf_counter() - start))
