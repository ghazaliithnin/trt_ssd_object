"""trt_ssd.py

This script demonstrates how to do real-time object detection with
TensorRT optimized Single-Shot Multibox Detector (SSD) engine.
"""


import time
import argparse
import trt_ssd_func
import cv2
import pycuda.autoinit  # This is needed for initializing CUDA driver


from utils.camera import add_camera_args, Camera



def parse_args():
    """Parse input arguments."""
    desc = ('Capture and display live camera video')
    parser = argparse.ArgumentParser(description=desc)
    parser = add_camera_args(parser)

    args = parser.parse_args()
    return args



def main():
    args = parse_args()
    cam = Camera(args)
    if not cam.isOpened():
        raise SystemExit('ERROR: failed to open camera!')
    while True:
        frame = cam.read()
        if frame is None:
            break
        clss = trt_ssd_func.run_trt_ssd(frame)
        print(clss)
        cv2.imshow("frame",frame)
        key = cv2.waitKey(1)
        if key == 27:  # ESC key: quit program
            break
    cam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
