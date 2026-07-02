import time
import numpy as np
import OpenGL.GL as gl
import pangolin

from multiprocessing import Process, Queue
from frame import Frame

class Point(object):
    # A Point is a 3-D point in the world
    # Each Point is observed in multiple Frames

    def __init__(self, ptmap, loc):
        self.pt = loc
        self.frames = []
        self.idxs = []

        self.id = len(ptmap.points)
        ptmap.points.append(self)

    def add_observation(self, frame, idx):
        self.frames.append(frame)
        self.idxs.append(idx)

VIEWS = {
    "top-down": dict(eye=(0, -100, 0), look=(0, 0, 0), up=(0, 0, 1)),
    "diagonal": dict(eye=(-2, -10, -2), look=(0, 0, 0), up=(0, -1, 0)),
}

class Map(object):
    def __init__(self, view="top-down"):
        self.frames = []
        self.points = []
        self.state = None
        self.q = Queue()
        self.view = VIEWS.get(view, VIEWS["top-down"])

        self.viewer_proc = Process(target=self.viewer_thread, args=(self.q,))
        self.viewer_proc.daemon = True
        self.viewer_proc.start()
    
    def viewer_thread(self, q):
        try:
            self.viewer_init(1024, 768)
        except Exception as e:
            print(f"3D viewer disabled: {e}")
            return
        while True:
            self.viewer_refresh(q)
    
    def viewer_init(self, W, H):
        pangolin.CreateWindowAndBind('Main', W, H)
        gl.glEnable(gl.GL_DEPTH_TEST)
        
        v = self.view
        self.scam = pangolin.OpenGlRenderState(
            pangolin.ProjectionMatrix(W, H, 420, 420, W//2, H//2, 0.2, 10000),
            pangolin.ModelViewLookAt(*v["eye"], *v["look"], *v["up"])
        )
        
        self.handler = pangolin.Handler3D(self.scam)
        
        # Create Interactive View in window
        self.dcam = pangolin.CreateDisplay()
        self.dcam.SetBounds(0.0, 1.0, 0.0, 1.0, -W/H)
        self.dcam.SetHandler(self.handler)

    def viewer_refresh(self, q):
        if self.state is None or not q.empty():
            self.state = q.get()

        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glClearColor(1.0, 1.0, 1.0, 1.0)

        self.dcam.Activate(self.scam)

        poses, pts = self.state

        # Draw 3D point cloud
        if len(pts) > 0:
            gl.glPointSize(2)
            gl.glColor3f(0.0, 1.0, 0.0)
            pangolin.DrawPoints(pts[:, :3])

        # Draw camera trajectory as connected line + position dots
        if len(poses) > 0:
            # pose is world-to-camera; camera centre in world = -R^T @ t = inv(pose)[:3, 3]
            cam_positions = np.array([np.linalg.inv(p)[:3, 3] for p in poses])
            gl.glLineWidth(2)
            gl.glColor3f(0.0, 0.0, 1.0)
            pangolin.DrawLine(cam_positions)
            gl.glPointSize(5)
            pangolin.DrawPoints(cam_positions)

        pangolin.FinishFrame()
        time.sleep(0.01)

    def display(self):
        poses, pts = [], []
        
        for f in self.frames:
            poses.append(f.pose)

        for p in self.points:
            pts.append(p.pt)
        
        self.q.put((np.array(poses), np.array(pts)))
    
    