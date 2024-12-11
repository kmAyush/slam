#!/usr/bin/env python3
import cv2
import numpy as np

from display import Display

W, H = 640, 480
disp = Display(W, H)

if __name__ == "__main__":
    cap = cv2.VideoCapture("videos/test_countryroad.mp4")
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            disp.paint(cv2.resize(frame, (W, H)))
        else:
            break
