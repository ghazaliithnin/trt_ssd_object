"""trt_ssd.py

This script demonstrates how to do real-time object detection with
TensorRT optimized Single-Shot Multibox Detector (SSD) engine.
"""


import time
import argparse

import cv2
import pycuda.autoinit  # This is needed for initializing CUDA driver

from utils.ssd_classes import get_cls_dict
from utils.ssd import TrtSSD
from utils.camera import add_camera_args, Camera
from utils.display import open_window, set_display, show_fps
from utils.visualization import BBoxVisualization


WINDOW_NAME = 'TrtSsdDemo'
INPUT_HW = (300, 300)
SUPPORTED_MODELS = ['ssd_mobilenet_v1_coco']


def parse_args():
    """Parse input arguments."""
    desc = ('Capture and display live camera video, while doing '
            'real-time object detection with TensorRT optimized '
            'SSD model on Jetson Nano')
    parser = argparse.ArgumentParser(description=desc)
    parser = add_camera_args(parser)
    parser.add_argument('-m', '--model', type=str,
                        default='ssd_mobilenet_v1_coco',
                        choices=SUPPORTED_MODELS)
    args = parser.parse_args()
    return args


def loop_and_detect(frame, trt_ssd, conf_th, vis):

    boxes, confs, clss = trt_ssd.detect(frame, conf_th)
    #frame = vis.draw_bboxes(frame, boxes, confs, clss)


    return clss

def run_trt_ssd(frame):
    args = parse_args()
    #cam = Camera(args)
    #if not cam.isOpened():
    #    raise SystemExit('ERROR: failed to open camera!')

    cls_dict = get_cls_dict(args.model.split('_')[-1])
    trt_ssd = TrtSSD(args.model, INPUT_HW)
    print("trt start")


    #open_window(
     #   WINDOW_NAME, 'Camera TensorRT SSD Demo',
     #   cam.img_width, cam.img_height)
    vis = BBoxVisualization(cls_dict)
    clss = loop_and_detect(frame, trt_ssd, conf_th=0.3, vis=vis)
    return clss

