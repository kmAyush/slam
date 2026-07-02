#!/usr/bin/env python3
import os
import sys
import cv2
import signal
import argparse
import numpy as np

from display import Display
from frame import Frame, denormalize, match_frames, IRt
from pointmap import Map, Point

W, H = 1920//2, 1080//2
F = 270

K = np.array([[F,0,W//2],[0,F,H//2],[0,0,1]])
Kinv = np.linalg.inv(K)


def triangulate(pose1, pose2, pts1, pts2):
    return cv2.triangulatePoints(pose1[:3], pose2[:3], pts1.T, pts2.T).T

def process_frame(img, disp, pointmap):
    img = cv2.resize(img, (W, H))
    frame = Frame(pointmap, img, K)
    print(f"{type(frame)}: {frame.id}")
    if frame.id == 0:
        return

    f1 = pointmap.frames[-1]
    f2 = pointmap.frames[-2]

    idx1, idx2, Rt = match_frames(f1, f2)
    # Rt = T_{prev←curr}; inverse gives T_{curr←world}
    f1.pose = np.dot(np.linalg.inv(Rt), f2.pose)

    pts4d = triangulate(f1.pose, f2.pose, f1.pts[idx1], f2.pts[idx2])

    # Reject near-infinity points BEFORE normalising
    good_pts4d = np.abs(pts4d[:, 3]) > 0.005
    pts4d /= pts4d[:, 3:]
    good_pts4d &= (pts4d[:, 2] > 0)

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

    if disp is not None:
        disp.paint(img)

    pointmap.display()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    view_group = parser.add_mutually_exclusive_group()
    view_group.add_argument("--top-down", dest="view", action="store_const", const="top-down",
                            help="Bird's-eye view (default)")
    view_group.add_argument("--diagonal", dest="view", action="store_const", const="diagonal",
                            help="Diagonal front view")
    parser.set_defaults(view="top-down")
    args = parser.parse_args()

    disp = Display(W, H) if os.getenv("D2D") is not None else None
    pointmap = Map(view=args.view)

    # Ensure Ctrl+C kills the viewer subprocess cleanly
    def _shutdown(sig, frame):
        pointmap.viewer_proc.terminate()
        sys.exit(0)
    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    cap = cv2.VideoCapture("videos/test_countryroad.mp4")
    while cap.isOpened():
        flag, frame = cap.read()
        if flag:
            process_frame(frame, disp, pointmap)
        else:
            break

    pointmap.viewer_proc.terminate()
