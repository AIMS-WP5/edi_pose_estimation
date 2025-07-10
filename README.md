# EDI Pose Estimation

This repository includes scripts for object pose detection using RGBD cameras. 
The original object detection algorithms are taken from [here](https://github.com/aims50toolbox/pose_estimation_ros).

### Setup

It is recommended to create a virtual environment and install all dependencies there.
```
python3 -m venv /path/to/new/venv
source /path/to/new/venv/bin/activate
cd /path/to/edi_pose_estimation/
pip install -r requirements.txt
```

### Usage

(Running object pose detection creates `mobile_sam.pt` and `rtdetr-x.pt` files in the working directory.)
Before running, modify config files in the `config` directory with the appropriate file paths or transformations.

To run object detection with a RealSense camera:  
`cd src`
`python3 rs_est.py`

With Basler Blaze camera:
`python3 blaze_est.py`  
For color frames this uses "intensity" images obtained from the Blaze camera.

### Scripts used for testing estimation accuracy

Testing was done by putting a bottle in a fixed position relative to an ArUco board 
and comparing this "ground truth" position with the estimated pose.

These scripts include:
`comb_align_testing.py` (requires precise transform between color and depth camera defined in code)  
`rs_est_testing.py`  
`zivid_est_testing.py`  

### Util dir

Includes several "helper" scripts for analyzing testing output, visualizing pointclouds etc.
