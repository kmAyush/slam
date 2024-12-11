import cv2
import numpy as np

from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform
class Extractor(object):
    def __init__(self, K):
        self.orb = cv2.ORB_create()
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.last = None
        self.K = K
        self.Kinv = np.linalg.inv(self.K)
    
    def normalize(self, pts):
        return np.dot(self.Kinv, add_ones(pts).T).T[:, 0:2]
    
    def denormalize(self, pt):
        ret = np.dot(self.K, np.array([pt[0], pt[1], 1.0]))
        return int(round(ret[0])), int(round(ret[1]))

    def extract(self, img):
        # Detection
        pts = cv2.goodFeaturesToTrack(np.mean(img, axis=2).astype(np.uint8), 2000, qualityLevel=0.01, minDistance=3)
        
        # Extraction
        kps = [cv2.KeyPoint(x=f[0][0], y=f[0][1], size=20) for f in pts]
        kps, des = self.orb.compute(img, kps)

        # Matching
        ret = []
        if self.last is not None:
            matches = self.bf.knnMatch(des, self.last['des'], k=2)
            for m,n in matches:
                if m.distance < 0.75*n.distance:
                    kp1 = kps[m.queryIdx].pt
                    kp2 = self.last['kps'][m.trainIdx].pt
                    ret.append((kp1, kp2))

        # Filter
        if len(ret) > 8:
            ret = np.array(ret)

            # Normalize to make ret independent of camera intrinsics
            ret[:, 0, :] = self.normalize(ret[:, 0, :])
            ret[:, 1, :] = self.normalize(ret[:, 1, :])
    
            model, inliers = ransac((ret[:, 0], ret[:, 1]),
                                    FundamentalMatrixTransform,
                                    min_samples=8,
                                    residual_threshold=1,
                                    max_trials=100)
            ret = ret[inliers]

        # Update last
        self.last = {'kps': kps, 'des': des}
        return ret

# turn [[x,y]] -> [[x,y,1]]
def add_ones(x):
  return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)