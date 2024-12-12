#!/usr/bin/env python3
import os
import cv2
import numpy as np

from display import Display
from frame import Frame, denormalize, match_frames, IRt
from pointmap import Map, Point

W, H = 1920//2, 1080//2
F = 270

K = np.array([[F,0,W//2],[0,F,H//2],[0,0,1]])
Kinv = np.linalg.inv(K)

disp = Display(W, H) if os.getenv("D2D") is not None else None
pointmap = Map()


def triangulate(pose1, pose2, pts1, pts2): 
    return cv2.triangulatePoints(pose1[:3], pose2[:3], pts1.T, pts2.T).T

def process_frame(img):
    img = cv2.resize(img, (W, H))
    frame = Frame(pointmap, img, K)
    print(f"{type(frame)}: {frame.id}")
    if frame.id == 0:
        return
    
    f1 = pointmap.frames[-1]
    f2 = pointmap.frames[-2]

    # Draw matches
    idx1, idx2, Rt = match_frames(f1, f2)
    f1.pose = np.dot(Rt, f2.pose)
    #print("{}, {}".format(len(idx1), len(idx2)))

    # Homogeneous 3D coordinates
    pts4d = triangulate(f1.pose, f2.pose, f1.pts[idx1], f2.pts[idx2])
    pts4d /= pts4d[:, 3:]

    # Reject points behind the camera
    good_pts4d = (np.abs(pts4d[:, 3]) > 0.005) & (pts4d[:, 2] > 0)
    
    for i, p in enumerate(pts4d):
        if not good_pts4d[i]:
            continue
        pt = Point(pointmap, p)
        pt.add_observation(f1, idx1[i])
        pt.add_observation(f2, idx2[i])



    for pt1, pt2 in zip(f1.pts[idx1], f2.pts[idx2]):
        u1, v1 = denormalize(K, pt1)
        u2, v2 = denormalize(K, pt2)

        cv2.circle(img, (u1, v1), color=(0, 255, 0), radius=3)
        cv2.line(img, (u1, v1), (u2, v2), color=(255, 0, 0))

    # 2D
    if disp is not None:
        disp.paint(img)
        
    # 3D
    pointmap.display()


if __name__ == "__main__":
    cap = cv2.VideoCapture("videos/test_countryroad.mp4")

    while cap.isOpened():

        # cap.read gives us the frame and a boolean value
        flag, frame = cap.read()

        if flag==True:
            process_frame(frame)
        else:
            break
