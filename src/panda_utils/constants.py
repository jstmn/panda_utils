import numpy as np

from panda_utils.types import PandaLimits, WorkspaceBounds
from panda_utils.utils import cm_to_m


# Note that when color and depth are combined to create a pointcloud, the dimensions of the narrower of the two sensors
# will be choosen as the output dimension. This means we should use HD mode for both depth and color. However, note that
# at 1280 x 720, the depth camera can only record at 6 fps. At 640 x 480, the FOV is still greater than the color
# cameras when in HD mode, however the fps increases to 640 x 480. Therefore, the final parameters we'll use is
# 640 x 480 @ 30fps for depth (FOV: 75 x 62), color: 1280 x 720 @ 15 fps (FOV: 69 x 42).

DEFAULT_DEOXYS_INTERFACE_CFG = "configs/charmander.yml"


# Panda robot limits based on official specifications
PANDA_LIMITS = PandaLimits(
    # Joint position limits [rad]
    q_min=np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]),
    q_max=np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]),
    q_vel_max=np.array([2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100]),
    q_acc_max=np.array([15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0]),
    q_jerk_max=np.array([7500.0, 3750.0, 5000.0, 6250.0, 7500.0, 10000.0, 10000.0]),
    tau_max=np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0]),
    tau_dot_max=np.array([1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0]),
    translation_vel_max=np.array([1.7, 1.7, 1.7, 0.0, 0.0, 0.0, 0.0]),
    rotation_vel_max=np.array([0.0, 0.0, 0.0, 2.5, 2.5, 2.5, 0.0]),
    translation_acc_max=np.array([13.0, 13.0, 13.0, 0.0, 0.0, 0.0, 0.0]),
    rotation_acc_max=np.array([0.0, 0.0, 0.0, 25.0, 25.0, 25.0, 0.0]),
    translation_jerk_max=np.array([6500.0, 6500.0, 6500.0, 0.0, 0.0, 0.0, 0.0]),
    rotation_jerk_max=np.array([0.0, 0.0, 0.0, 12500.0, 12500.0, 12500.0, 0.0]),
)



DEPTH_MAX_M = 1.4

# See https://docs.google.com/drawings/d/1Mvzcg4S3asflx9s8VEFmOry6SM-qBRHE3Jz-pSo5bjY/edit?usp=sharing for details
WORKSPACE_BOUNDS = WorkspaceBounds(
    min_x_m=cm_to_m(-21),
    max_x_m=cm_to_m(99),
    min_y_m=cm_to_m(-38.25),
    max_y_m=cm_to_m(42.75),
    min_z_m=cm_to_m(-1.0),
    max_z_m=cm_to_m(75.0),
)

"""Intrinsics and Extrinsics of Realsense D435i
- PPX, PPY: Principal point coordinates (optical center) in pixel coordinates
- FX, FY: Focal lengths in pixel units (focal length in mm divided by pixel size)

 Intrinsic of "Depth" / 640x480 / {Z16}
  Width:      	640
  Height:     	480
  PPX:        	324.033264160156
  PPY:        	233.439270019531
  Fx:         	390.25830078125
  Fy:         	390.25830078125
  Distortion: 	Brown Conrady
  Coeffs:     	0  	0  	0  	0  	0
  FOV (deg):  	78.7 x 63.17

 Intrinsic of "Color" / 1280x720 / {YUYV/RGB8/BGR8/RGBA8/BGRA8/Y8}
  Width:      	1280
  Height:     	720
  PPX:        	641.741088867188
  PPY:        	369.680297851562
  Fx:         	912.776611328125
  Fy:         	912.644897460938
  Distortion: 	Inverse Brown Conrady
  Coeffs:     	0  	0  	0  	0  	0
  FOV (deg):  	70.07 x 43.05

Extrinsic from "Color"	  To	  "Depth" :
 Rotation Matrix:
   0.999749        -0.021599        -0.00590855
   0.021574         0.999758        -0.00426594
   0.00599926       0.0041374        0.999973

 Translation Vector: -0.0146005274727941  -0.000270371703663841  -0.000890049617737532
"""
