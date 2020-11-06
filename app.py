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
import argparse
from threading import Thread, enumerate
from queue import Queue


app = Flask(__name__)




@app.route('/')
def index():
    return render_template('index.html')


def gen(camera):
    while True:
        frame = camera.get_frame()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')




@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5050, debug=True)

