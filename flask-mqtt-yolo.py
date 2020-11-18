from flask import Flask, render_template, Response
from ctypes import *
import math
import random
import os
import cv2
import numpy as np
import time
import darknet
from datetime import datetime
import json
from camera import VideoCamera
import threading
import argparse
from threading import Thread, enumerate
from queue import Queue
import paho.mqtt.client as mqtt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--flask", action="store_true",
                    help="enable flask app")
parser.add_argument('-v', '--verbose', action="store_true",
                    required=False, default=False, help='Enable verbose output')
parser.add_argument("-i", "--ip", type=str, required=False, default=os.getenv('LISTEN_IP', '0.0.0.0'),
                    help="listen ip address")
parser.add_argument("--port", type=int, required=False, default=os.getenv('LISTEN_PORT', '5000'),
                    help="ephemeral port number of the server (1024 to 65535) default 5000")
parser.add_argument('-d', '--devno', type=int, default=os.getenv('DEVNO', '-1'),
                    help='device number for camera (typically -1=find first available, 0=internal, 1=external)')
parser.add_argument('-n', '--capture-string', type=str, default=os.getenv('CAPTURE_STRING'),
                    help='Any valid VideoCapture string(IP camera connection, RTSP connection string, etc')
parser.add_argument('-c', '--confidence', type=float,
                    default=os.getenv('CONFIDENCE', '0.3'))
parser.add_argument('-p', '--publish', action="store_true")
parser.add_argument('-s', '--sleep', type=float,
                    default=os.getenv('SLEEP', '1.0'))
parser.add_argument('--protocol', type=str,
                    default=os.getenv('PROTOCOL', 'HTTP'))
parser.add_argument('-m', '--model-name', type=str, required=False,
                    default=os.getenv('MODEL_NAME', 'ssd_mobilenet_coco'), help='Name of model')
parser.add_argument('-x', '--model-version', type=str, required=False, default=os.getenv('MODEL_VERSION', ''),
                    help='Version of model. Default is to use latest version.')
parser.add_argument('-u', '--url', type=str, required=False, default=os.getenv('TRITON_URL', 'localhost:5000'),
                    help='Inference server URL. Default is localhost:8000.')
parser.add_argument('-b', '--mqtt-broker-host', type=str, required=False, default=os.getenv('MQTT_BROKER_HOST', 'fluent-bit'),
                    help='mqtt broker host')
parser.add_argument("--mqtt-broker-port", type=int, required=False, default=os.getenv('MQTT_BROKER_PORT', '1883'),
                    help="port number of the mqtt server (1024 to 65535) default 1883")
parser.add_argument('-t', '--mqtt-topic', type=str, required=False, default=os.getenv('MQTT_TOPIC', '/demo'),
                    help='mqtt broker topic')
parser.add_argument('-ann', '--armnn', action="store_true")
parser.add_argument('-db1', '--detect-car', action="store_true")
parser.add_argument('-db2', '--detect-person', action="store_true")
parser.add_argument('-db3', '--detect-bus', action="store_true")
parser.add_argument('-db4', '--detect-bicycle', action="store_true")
parser.add_argument('-db5', '--detect-motorcycle', action="store_true")
args = parser.parse_args()

outputFrame = None
outputArray = None
lock = threading.Lock()
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")
                    

def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax

def cvDrawBoxes(detections, img):
    for detection in detections:
        x, y, w, h = detection[2][0],            detection[2][1],            detection[2][2],            detection[2][3]
        xmin, ymin, xmax, ymax = convertBack(
            float(x), float(y), float(w), float(h))
        pt1 = (xmin, ymin)
        pt2 = (xmax, ymax)
        cv2.rectangle(img, pt1, pt2, (255, 255, 0), 4)
        cv2.putText(img,
                    detection[0].encode('utf-8').decode() +
                    " [" + str(round(float(detection[1]) * 100, 2)) + "]",
                    (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    [255, 255, 0], 2)
    return img
    
def detection_loop():
    global outputFrame,outputArray,lock

    configPath = "./cfg/yolov3-tiny.cfg"
    weightPath = "./yolov3-tiny.weights"
    metaPath = "./cfg/coco.data"
    netMain = darknet.load_net_custom(configPath.encode("ascii"), weightPath.encode("ascii"), 0, 1)
    metaMain = darknet.load_meta(metaPath.encode("ascii"))
    try:
        with open(metaPath) as metaFH:
            metaContents = metaFH.read()
            import re
            match = re.search("names *= *(.*)$", metaContents,re.IGNORECASE | re.MULTILINE)
            if match:
                result = match.group(1)
            else:
                result = None
            try:
                if os.path.exists(result):
                    with open(result) as namesFH:
                        namesList = namesFH.read().strip().split("\n")
                        altNames = [x.strip() for x in namesList]
            except TypeError:
                pass
    except Exception:
        pass
    out = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*"MJPG"), 10.0,(darknet.network_width(netMain), darknet.network_height(netMain)))
    darknet_image = darknet.make_image(darknet.network_width(netMain), darknet.network_height(netMain),3)
    cam = cv2.VideoCapture(0)
    cam.set(3,1280)
    cam.set(4,720)
    print("------------darknet_finished---------")
    try:
        for img,text in getframe(cam,netMain,metaMain,darknet_image,out):
            if img is not None:
                with lock:
                    outputFrame = img
                    outputArray = text
            else:
                print("--------------Thread_finished(ctl + C)-----------------")
                
    except:
        os._exit(1)
        
def getframe(cam,netMain,metaMain,darknet_image,out):
    while True:
        ret, frame_read = cam.read()
        height, width, channels = frame_read.shape[:3]
        frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)

        frame_resized = cv2.resize(frame_rgb,
                                (darknet.network_width(netMain),
                                    darknet.network_height(netMain)),
                                interpolation=cv2.INTER_LINEAR)
        darknet.copy_image_from_bytes(darknet_image,frame_resized.tobytes())
        detections = darknet.detect_image(netMain,metaMain,darknet_image, thresh=0.25)
        image = cvDrawBoxes(detections, frame_resized)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ret1, jpeg = cv2.imencode('.jpg', image)
        out.write(image)
        cv2.waitKey(1)
        jpeg = jpeg.tobytes()
        yield jpeg,detections
    cam.release()
    yield  None,None

def generate():
    global outputFrame,outputArray,lock

    while True:
        with lock:
            if outputFrame is None:
                continuei
        if outputArray!=[]:
            print(outputArray)

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + outputFrame + b'\r\n\r\n')
               
if __name__ == '__main__':
    t = threading.Thread(target=detection_loop)
    t.start()
    app.run(host='0.0.0.0',threaded=True,port=5000,debug=False)
    print("--------------finished-----------------")
