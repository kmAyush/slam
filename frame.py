import cv2
import numpy as np

from skimage.measure import ransac
from skimage.transform import EssentialMatrixTransform

def extract(img):
    orb = cv2.ORB_create()
    # Detection
    pts = cv2.goodFeaturesToTrack(np.mean(img, axis=2).astype(np.uint8), 3000, qualityLevel=0.01, minDistance=3)
    
    # Extraction
    kps = [cv2.KeyPoint(x=f[0][0], y=f[0][1], size=20) for f in pts]
    kps, des = orb.compute(img, kps)
    return np.array([(kp.pt[0], kp.pt[1]) for kp in kps]), des

def match_frames(f1, f2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(f1.des, f2.des, k=2)

    # Lowe's ratio test
    kps_pairs = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            pt1=f1.kps[m.queryIdx]
            pt2=f2.kps[m.trainIdx]
            kps_pairs.append((pt1, pt2))
            
    assert len(kps_pairs) >= 8
    kps_pairs = np.array(kps_pairs)

    # Fit matrix
    model, inliers = ransac((kps_pairs[:, 0], kps_pairs[:, 1]),
                            EssentialMatrixTransform,
                            min_samples=8,
                            residual_threshold=0.005,
                            max_trials=200)
    kps_pairs = kps_pairs[inliers]
    Rt = extractRt(model.params)

    return kps_pairs, Rt

def normalize(Kinv, pts):
    return np.dot(Kinv, add_ones(pts).T).T[:, 0:2]

def denormalize(K, pt):
    ret = np.dot(K, np.array([pt[0], pt[1], 1.0]))
    return int(round(ret[0])), int(round(ret[1]))

# def extract(self, img):
#     # Detection
#     pts = cv2.goodFeaturesToTrack(np.mean(img, axis=2).astype(np.uint8), 3000, qualityLevel=0.01, minDistance=3)
    
#     # Extraction
#     kps = [cv2.KeyPoint(x=f[0][0], y=f[0][1], size=20) for f in pts]
#     kps, des = self.orb.compute(img, kps)

#     # Matching
#     ret = []
#     if self.last is not None:
#         matches = self.bf.knnMatch(des, self.last['des'], k=2)
#         for m,n in matches:
#             if m.distance < 0.75*n.distance:
#                 kp1 = kps[m.queryIdx].pt
#                 kp2 = self.last['kps'][m.trainIdx].pt
#                 ret.append((kp1, kp2))
#         #print(f"Number of raw matches: {len(matches)}")
#         #print(f"Number of filtered matches: {len(ret)}")

#     # Filter
#     Rt = None
#     if len(ret) > 0:
#         ret = np.array(ret)

#         # Normalize to make ret independent of camera intrinsics
#         ret[:, 0, :] = self.normalize(ret[:, 0, :])
#         ret[:, 1, :] = self.normalize(ret[:, 1, :])
#         #print(len(ret[:,0]), len(ret[:,1]))
#         if len(ret) < 8:
#             # Not enough matches, skip this frame
#             return None, None
#         model, inliers = ransac((ret[:, 0], ret[:, 1]),
#                                 EssentialMatrixTransform,
#                                 min_samples=8,
#                                 residual_threshold=0.005,
#                                 max_trials=200)
#         ret = ret[inliers]
#         Rt = extractRt(model.params)

#     # Update last
#     self.last = {'kps': kps, 'des': des}
#     return ret, Rt

# turn [[x,y]] -> [[x,y,1]]
def add_ones(x):
  return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)

def extractRt(E):
    W = np.mat([[0,-1,0],[1,0,0],[0,0,1]],dtype=float)
    U,d,Vt = np.linalg.svd(E)
    assert np.linalg.det(U) > 0
    if np.linalg.det(Vt) < 0:
        Vt *= -1.0
    R = np.dot(np.dot(U, W), Vt)
    if np.sum(R.diagonal()) < 0:
        R = np.dot(np.dot(U, W.T), Vt)
    t = U[:, 2]
    Rt = np.concatenate([R,t.reshape(3,1)], axis=1)
    return Rt

def extract_Rot_trans(E):
    W = np.mat([[0,-1,0],[1,0,0],[0,0,1]],dtype=float)
    U, d, Vt = np.linalg.svd(E)
    if np.linalg.det(U) < 0:
        U *= -1.0
    if np.linalg.det(Vt) < 0:
        Vt *= -1.0
    R = np.dot(np.dot(U, W), Vt)
    R2 = np.dot(np.dot(U, W.T), Vt)
    t = U[:, 2]
    Rt = np.concatenate([R,t.reshape(3,1)], axis=1)

class Frame(object):
    def __init__(self, img, K):
        self.K = K
        self.Kinv = np.linalg.inv(self.K)

        pts, self.des = extract(img)
        self.kps = normalize(self.Kinv, pts)