# panda_utils

This library contains code to do the following:
* Convert RGB and Depth frames from a realsense camera to a pointcloud
* Control the panda using the [deoxys_control](https://github.com/UT-Austin-RPL/deoxys_control) library. Notably, collision checking is performed before actions are sent to the robot. Collision checking is performed by the [Jrl2](https://github.com/jstmn/Jrl2) library. Collisions are checked between links, and primitives (cuboids, spheres) which can be placed in the scene.









## Installation

Almost always I recommend using [uv](https://docs.astral.sh/uv/) or [pixi](https://pixi.sh/latest/), but because there's so many libraries that need to be cloned locally and then installed in an editable mode, I recommend just sticking to a simple conda environment. 

```bash
sudo apt install libhidapi-hidraw0 libhidapi-libusb0
echo "DEOXYS_DIR='/PATH/TO/deoxys_control'" >> ~/.bashrc; source ~/.bashrc
echo "PYTHONPATH='$PYTHONPATH:$DEOXYS_DIR/deoxys/'" >> ~/.bashrc; source ~/.bashrc
echo "ROS_WS='/PATH/TO/ROS_WS'" >> ~/.bashrc; source ~/.bashrc
echo "PANDA_UTILS_DIR='/PATH/TO/panda_utils'" >> ~/.bashrc; source ~/.bashrc

# Create and activate a conda environment called `panda`:
conda create -n panda_utils python=3.10 -y
conda activate panda_utils
pip install matplotlib numpy open3d opencv-python termcolor h5py matplotlib rospkg ruff black
git clone https://github.com/jstmn/Jrl2 thirdparty/Jrl2
git clone https://github.com/jstmn/Jrl thirdparty/Jrl
pip install -e thirdparty/Jrl2
pip install -e thirdparty/Jrl
pip install -U -r ${DEOXYS_DIR}/deoxys/requirements.txt
pip install 'protobuf<=3.20.0'  # This will give warnings/errors, that's fine
pip install -e ${DEOXYS_DIR}/deoxys
pip install -e .
```





## Data collection

``` bash

# in every new terminal:
conda activate panda_utils; source /opt/ros/humble/setup.bash; source ~/Desktop/ros2_ws/install/setup.bash

# Terminal 1 - ZED cameras - Stop with `bash scripts/stop_zed.sh`
# Resolution is VGA (672x376 @ 30fps, RGB + depth). launch_all.sh can't override it: to change, set
# `grab_resolution` (HD2K | HD1080 | HD720 | VGA) in ~/Desktop/ros2_ws/src/zed-ros2-wrapper/zed_wrapper/config/{zed,zed2i,zedm}.yaml,
# then rebuild: (cd ~/Desktop/ros2_ws && colcon build --packages-select zed_wrapper)
CAMS="zed zed2i zedm" bash ~/Desktop/ros2_ws/zed_tools/launch_all.sh


# Terminal 2 - Teleoperation + log demonstrations [Option 1]
python scripts/log_demonstration.py \
  --output_dir /home/res1/Desktop/real2sim-toolkit/data/Oct20_2026_datacollect --description PickCube \
  --recording_rate_hz 20.0 --camera_ids zed zed2i zedm --zed_resolution VGA

# Terminal 2 - Teleoperation [Option 2]
python scripts/panda_teleop.py
```

### Output format

`log_demonstration.py` writes one HDF5 per demo, `<output_dir>/<description>__<MM-DD-YYYY__HH-MM-SS>/traj_<i>__<zed_resolution>.h5`,
in [real2sim-toolkit](https://github.com/jstmn/real2sim-toolkit)'s ManiSkill layout (from "Data formats" in its README,
checked by `r2st.utils.validate_merged_camera_group`). All datasets share one frame index; `*` marks extras beyond the toolkit's spec.

| Dataset | Shape, dtype | Notes |
| --- | --- | --- |
| `obs/qpos`, `obs/qvel`* | `(T, 9)` float64 | `panda_joint1..7`, `panda_finger_joint1/2`. `(T, 7)` with `--no-gripper` |
| `obs/sensor_data/<cam>/rgb` | `(T, H, W, 3)` uint8 | RGB, gzip 1. `<cam>` is the `--camera_ids` name |
| `obs/sensor_data/<cam>/depth` | `(T, H, W)` uint16 | mm, registered to `rgb` (unproject with the color K), gzip 1 |
| `obs/sensor_data/<cam>/timestamp_ms` | `(T,)` float64 | When the logger took sample `i`. Same for every camera |
| `obs/sensor_data/<cam>/{rgb,depth*}_timestamp_ms` | `(T,)` float64 | Capture time of that frame |
| `obs/robot_arm_ns`*, `obs/robot_gripper_ns`* | `(T,)` int64 | Franka's own clocks (ns since boot, 1 ms resolution). Only comparable with themselves |

`*_timestamp_ms` are host wall clock, Unix-epoch ms. Root attrs: `zed_resolution`, `timezone`, `start_time_local`,
`end_time_local`, `duration_s`. Without depth (`--no-depth`, `--no-depth-cameras`) a camera fails the toolkit's validator.
Also written per demo: `<stem>_timing.json`, `<cam>_camera_info_{color,depth}.json` (rectified intrinsics), and `images/`
(a preview every 10th sample).
