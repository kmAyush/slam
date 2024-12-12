import numpy as np
import OpenGL.GL as gl
import pypangolin as pangolin

from multiprocessing import Process, Queue
from frame import Frame

class Point(object):
    # A Point is a 3-D point in the world
    # Each Point is observed in multiple Frames

    def __init__(self, ptmap, loc):
        self.pt = loc
        self.frames = []
        self.idxs = []

        self.id = len(ptmap.frames)
        ptmap.frames.append(self)

    def add_observation(self, frame, idx):
        self.frames.append(frame)
        self.idxs.append(idx)

class Map(object):
    def __init__(self):
        #self.location = loc
        self.frames = []
        self.points = []
        self.state = None
        self.q = Queue()

        p = Process(target=self.viewer_thread, args=(self.q,))
        p.daemon = True
        p.start()
    
    def viewer_thread(self, q):
        self.viewer_init(1024, 768)
        while True:
            self.viewer_refresh(q)
    
    def viewer_init(self, W, H):
        pangolin.CreateWindowAndBind('Main', W, H)
        gl.glEnable(gl.GL_DEPTH_TEST)
        
        self.scam = pangolin.OpenGlRenderState(
            pangolin.ProjectionMatrix(W, H, 420, 420, W//2, H//2, 0.2, 10000),
            pangolin.ModelViewLookAt(-2, 2, -2, 
                                     0, 0, 0,
                                     0, -1, 0 ))
        
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
        gl.glPointSize(2)
        gl.glColor3f(0.0, 1.0, 0.0)
        
        pangolin.DrawPoints(self.points)
        
        pangolin.FinishFrame()

    def display(self):
        poses, pts = [], []
        
        for f in self.frames:
            if isinstance(f, Frame):
                poses.append(f.pose)

        for p in self.points:
            pts.append(p.pt)
        
        self.q.put((np.array(poses), np.array(pts)))
    
    