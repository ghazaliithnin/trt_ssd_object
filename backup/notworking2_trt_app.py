"""trt_ssd.py

This script demonstrates how to do real-time object detection with
TensorRT optimized Single-Shot Multibox Detector (SSD) engine.
"""
from flask import Flask, render_template, Response
import cv2
app = Flask(__name__)

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
SUPPORTED_MODELS = [
    'ssd_mobilenet_v1_coco',
    'ssd_mobilenet_v1_egohands',
    'ssd_mobilenet_v2_coco',
    'ssd_mobilenet_v2_egohands',
    'ssd_inception_v2_coco',
    'ssdlite_mobilenet_v2_coco',
]


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


def loop_and_detect(cam, trt_ssd, conf_th, vis):
    """Continuously capture images from camera and do object detection.

    # Arguments
      cam: the camera instance (video source).
      trt_ssd: the TRT SSD object detector instance.
      conf_th: confidence/score threshold for object detection.
      vis: for visualization.
    """
    full_scrn = False
    fps = 0.0
    tic = time.time()
    while True:
        if cv2.getWindowProperty(WINDOW_NAME, 0) < 0:
            break
        img = cam.read()
        if img is None:
            break
        boxes, confs, clss = trt_ssd.detect(img, conf_th)
        img = vis.draw_bboxes(img, boxes, confs, clss)
        img = show_fps(img, fps)
        #cv2.imshow(WINDOW_NAME, img)
        toc = time.time()
        curr_fps = 1.0 / (toc - tic)
        # calculate an exponentially decaying average of fps number
        fps = curr_fps if fps == 0.0 else (fps*0.95 + curr_fps*0.05)
        tic = toc
        
        return img
        
        '''
        key = cv2.waitKey(1)
        if key == 27:  # ESC key: quit program
            break
        elif key == ord('F') or key == ord('f'):  # Toggle fullscreen
            full_scrn = not full_scrn
            set_display(WINDOW_NAME, full_scrn)
        '''
        
def gen_frames():

    ret, buffer = cv2.imencode('.jpg', frame)
    #print("shape",frame.shape)
    #frame = cv2.resize(frame,(640,480))
    #image, net_yolo = check_vaccine_func.load_image(frame, net_yolo)
    #ln = check_vaccine_func.get_layer_names(image, net_yolo)
    #check_vaccine_func.detect_vac(frame, net_yolo, ln)
    #time.sleep(2)

    frame = buffer.tobytes()
    yield (b'--frame\r\n'
           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result
def index():
    print("in index")
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    print("in video_feed")
    args = parse_args()
    cam = Camera(args)
    if not cam.isOpened():
        raise SystemExit('ERROR: failed to open camera!')

    cls_dict = get_cls_dict(args.model.split('_')[-1])
    trt_ssd = TrtSSD(args.model, INPUT_HW)

    open_window(
        WINDOW_NAME, 'Camera TensorRT SSD Demo',
        cam.img_width, cam.img_height)
    vis = BBoxVisualization(cls_dict)
    #loop_and_detect(cam, trt_ssd, conf_th=0.3, vis=vis)
    #frame=loop_and_detect(cam, trt_ssd, conf_th=0.3, vis=vis)
    

    #cam.release()
    #cv2.destroyAllWindows()
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    #return Response(loop_and_detect(cam, trt_ssd, conf_th=0.3, vis=vis), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)
