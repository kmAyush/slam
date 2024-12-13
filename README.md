# SLAM Implementation

## Technologies Used
- **OpenCV**: For image processing, feature extraction, and keypoints matching.
- **NumPy**: For numerical computations and matrix operations.
- **SciKit-Image**: Used for RANSAC and Essential Matrix estimation.
- **SDL2**: For real-time display of video frames with overlaid feature matches.

## Methodology

### Feature Detection and Description
- ORB (Oriented FAST and Rotated BRIEF) is used for detecting and describing keypoints.
- `cv2.goodFeaturesToTrack` is utilized for detecting keypoints with subpixel precision.

### Feature Matching
- Brute-force matching with Hamming distance.
- Lowe's ratio test for filtering reliable matches.

### Essential Matrix Estimation
- RANSAC is employed to estimate the Essential Matrix, filtering outliers from feature matches.

### Camera Pose Estimation
- Decompose the Essential Matrix to extract Rotation and Translation matrices.
- Normalize and denormalize points for mapping matches to image coordinates.

### Frame Processing
- A `Frame` class stores keypoints, descriptors, and associated transformations.
- Matches are visualized by drawing circles and lines on frames.

### Real-time Visualization
- Real-time frame processing is achieved using SDL2 for a seamless display.

## Key Files
- **`slam.py`**: Main script to process video frames and manage SLAM pipeline.
- **`frame.py`**: Handles feature extraction, matching, and pose estimation.
- **`display.py`**: Manages visualization using SDL2.
