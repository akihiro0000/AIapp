#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from flask import Flask, render_template, Response
from ctypes import *
import math
import random
import os
import cv2
import numpy as np
import time
import darknet
from camera import VideoCamera
#add 
import argparse
from threading import Thread, enumerate
from queue import Queue
#add
import paho.mqtt.client as mqtt
import argparse



#add : all arg
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





app = Flask(__name__)




@app.route('/')
def index():
    return render_template('index.html')


def gen(camera):
    while True:
        #add : mqtt_text
        frame,mqtt_text = camera.get_frame()
        #add        
        mqtt_client.publish("{}/{}".format(args.mqtt_topic, label),mqtt_text)

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')




@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    #add
    mqtt_client = mqtt.Client()
    #add
    mqtt_client.connect(args.mqtt_broker_host, args.mqtt_broker_port,60)
    #add
    mqtt_client.loop_start()

    app.run(host='0.0.0.0', debug=True)
    
    #add
    mqtt_client.disconnect()
