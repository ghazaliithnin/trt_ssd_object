
#Import necessary libraries
from flask import Flask, render_template, Response
import cv2

#Initialize the Flask app
app = Flask(__name__)

import time
import argparse

import cv2
import pycuda.autoinit  # This is needed for initializing CUDA driver

from utils.camera import add_camera_args, Camera
global webvideo_frame
webvideo_frame = None

def parse_args():
    """Parse input arguments."""
    desc = ('Capture and display live camera video on browser html')
    parser = argparse.ArgumentParser(description=desc)
    parser = add_camera_args(parser)

    args = parser.parse_args()
    return args
    
def loopcamera(cam):
    global webvideo_frame
    while True:
        frame = cam.read()
        if frame is None:
            break
        
        webvideo_frame=frame.copy()


def main():
    args = parse_args()
    cam = Camera(args)

    if not cam.isOpened():
        raise SystemExit('ERROR: failed to open camera!')
    loopcamera(cam)



def gen_frames():   
    ret, buffer = cv2.imencode('.jpg', webvideo_frame)
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
    main()
