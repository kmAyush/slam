#!/usr/bin/env python3
import cv2
import numpy as np

from display import Display
from frame import Frame, denormalize, match_frames

W, H = 1920//2, 1080//2
F = 270

disp = Display(W, H)
K = np.array([[F,0,W//2],[0,F,H//2],[0,0,1]])

frames=[]
def process_frame(img):
    img = cv2.resize(img, (W, H))
    f = Frame(img, K)
    frames.append(f)
    if len(frames) <= 1:
        return
    
    # Draw matches
    matches, Rt = match_frames(frames[-1], frames[-2])
    for pt1, pt2 in matches:
        u1, v1 = denormalize(K, pt1)
        u2, v2 = denormalize(K, pt2)

        cv2.circle(img, (u1, v1), color=(0, 255, 0), radius=3)
        cv2.line(img, (u1, v1), (u2, v2), color=(255, 0, 0))
    disp.paint(img)

if __name__ == "__main__":
    cap = cv2.VideoCapture("videos/test_countryroad.mp4")

    while cap.isOpened():

        # cap.read gives us the frame and a boolean value
        flag, frame = cap.read()

        if flag==True:
            process_frame(frame)
        else:
            break
