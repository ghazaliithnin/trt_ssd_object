
#Import necessary libraries
from flask import Flask, render_template, Response
import cv2

#Initialize the Flask app
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
import trt_ssd_func

INPUT_HW = (300, 300)
SUPPORTED_MODELS = ['ssd_mobilenet_v1_coco']

def parse_args():
    """Parse input arguments."""
    desc = ('Capture and display live camera video on browser html')
    parser = argparse.ArgumentParser(description=desc)
    parser = add_camera_args(parser)
    parser.add_argument('-m', '--model', type=str,
                        default='ssd_mobilenet_v1_coco',
                        choices=SUPPORTED_MODELS)
    args = parser.parse_args()
    return args

# Function to get coordinates of bounding boxes
def coords(boxes):
    for p1,p2,p3,p4 in boxes:
        return p1,p2,p3,p4
    
# Function to get the class ID of an object
def kelas(clss):
    for item_class in clss:
        return item_class


def gen_frames():
    
    args = parse_args()
    cam = Camera(args)

    if not cam.isOpened():
        raise SystemExit('ERROR: failed to open camera!')
    

    while True:
        frame = cam.read()
        if frame is None:
            break
        else:

            clss = trt_ssd_func.run_trt_ssd(frame)
            try:
                #p1,p2,p3,p4 = coords(boxes)
                item_class=kelas(clss)
                print(item_class)
            except Exception as e:
                print(e)
                #pass
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

@app.route('/')
def index():
    print("in index")
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    print("in video_feed")
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
