#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
from ctypes import *
import math
import random
import os
import numpy as np
import time
import darknet
from threading import Thread, enumerate
from queue import Queue


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


netMain = None
metaMain = None
altNames = None


class VideoCamera(object):
    def __init__(self):
        global metaMain, netMain, altNames
        configPath = "./cfg/yolov3-tiny.cfg"
        weightPath = "./yolov3-tiny.weights"
        metaPath = "./cfg/coco.data"
        if not os.path.exists(configPath):
            raise ValueError("Invalid config path `" + os.path.abspath(configPath)+"`")
        if not os.path.exists(weightPath):
            raise ValueError("Invalid weight path `" + os.path.abspath(weightPath)+"`")
        if not os.path.exists(metaPath):
            raise ValueError("Invalid data file path `" + os.path.abspath(metaPath)+"`")
        if netMain is None:
            self.netMain = darknet.load_net_custom(configPath.encode("ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
        if metaMain is None:
            self.metaMain = darknet.load_meta(metaPath.encode("ascii"))
        if altNames is None:
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
                                self.altNames = [x.strip() for x in namesList]
                    except TypeError:
                        pass
            except Exception:
                pass

        self.out = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*"MJPG"), 10.0,(darknet.network_width(self.netMain), darknet.network_height(self.netMain)))
        print("Starting the YOLO loop...")

        # Create an image we reuse for each detect
        self.darknet_image = darknet.make_image(darknet.network_width(self.netMain), darknet.network_height(self.netMain),3)
        self.video = cv2.VideoCapture(0)
        self.video.set(3,1280)
        self.video.set(4,720)


    def __del__(self):
        self.video.release()

    def get_frame(self):
        ret, frame_read = self.video.read()
        print(ret)
        print(frame_read)
        height, width, channels = frame_read.shape[:3]
        frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb,
                                (darknet.network_width(self.netMain),
                                    darknet.network_height(self.netMain)),
                                interpolation=cv2.INTER_LINEAR)
        
        
        darknet.copy_image_from_bytes(self.darknet_image,frame_resized.tobytes())
        detections = darknet.detect_image(self.netMain, self.metaMain, self.darknet_image, thresh=0.25)
        image = cvDrawBoxes(detections, frame_resized)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ret, jpeg = cv2.imencode('.jpg', image)
        #cv2.imshow('YOLOv3-Tiny', image)
        self.out.write(image)
        cv2.waitKey(1)
        
        #add:detections
        return jpeg.tobytes(),detections

