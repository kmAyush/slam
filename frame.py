import cv2
import numpy as np

def extract(img):
    orb = cv2.ORB_create()
    pts = cv2.goodFeaturesToTrack(np.mean(img, axis=2).astype(np.uint8), 500, qualityLevel=0.01, minDistance=3)
    kps = [cv2.KeyPoint(x=f[0][0], y=f[0][1], _size=20) for f in pts]
    kps, des = orb.compute(img, kps)
    return np.array([(kp.pt[0], kp.pt[1]) for kp in kps]), des

def match_frames(des1, des2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(des1, des2, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    return good_matches
