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
echo "ROS_WS='/PATH/TO/ROS2_WS'" >> ~/.bashrc; source ~/.bashrc

# Create the 'panda_utils' conda env:
conda create -n panda_utils python=3.10 -y
conda activate panda_utils
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
  --output_dir /home/res1/Desktop/real2sim-toolkit/data/Oct26_2026_datacollect --description PickCube_6x6x6cm \
  --recording_rate_hz 20.0 --camera_ids zed zed2i zedm --zed_resolution VGA --show-camera-feed

# Terminal 2 - Teleoperation [Option 2]
python scripts/run_teleop.py
```

### Output format

`log_demonstration.py` writes one HDF5 per demo, `<output_dir>/<description>__<MM-DD-YYYY__HH-MM-SS>/traj_<i>__<zed_resolution>.h5`,
in [real2sim-toolkit](https://github.com/jstmn/real2sim-toolkit)'s stereo layout (from "Data format" in its README).
`r2st.utils.validate_noisy_depth_camera_group` checks a camera's RGB-D datasets and `validate_stereo_depth_camera_group`
its stereo pair. All datasets share one frame index; `*` marks extras beyond the toolkit's spec.

| Dataset | Shape, dtype | Notes |
| --- | --- | --- |
| `obs/qpos`, `obs/qvel`* | `(T, 9)` float64 | `panda_joint1..7`, `panda_finger_joint1/2`. `(T, 7)` with `--no-gripper` |
| `obs/sensor_data/<cam>/timestamp_ms` | `(T,)` float64 | When the logger took sample `i`. Same for every camera. `<cam>` is the `--camera_ids` name |
| `obs/sensor_data/<cam>/left/rgb` | `(T, H, W, 3)` uint8 | Rectified left RGB, gzip 1 |
| `obs/sensor_data/<cam>/left/depth_noisy` | `(T, H, W)` uint16 | The ZED's own depth, mm, registered to `left/rgb` (unproject with the left K), gzip 1 |
| `obs/sensor_data/<cam>/right/rgb` | `(T, H, W, 3)` uint8 | Rectified right RGB, gzip 1. ZEDs only |
| `.../{left/rgb,left/depth_noisy*,right/rgb}_timestamp_ms` | `(T,)` float64 | Capture time of that frame. Left and right of a sample share one stamp |
| `obs/robot_arm_ns`*, `obs/robot_gripper_ns`* | `(T,)` int64 | Franka's own clocks (ns since boot, 1 ms resolution). Only comparable with themselves |

`*_timestamp_ms` are host wall clock, Unix-epoch ms. Root attrs: `zed_resolution`, `timezone`, `start_time_local`,
`end_time_local`, `duration_s`. Without depth (`--no-depth`, `--no-depth-cameras`) a camera fails the noisy-depth
validator; a RealSense camera (no right eye) fails the stereo one. Also written per demo: `<stem>_timing.json`,
`<cam>_camera_info.json` (rectified intrinsics keyed `left`, `depth_noisy`, `right`; `right.p[3] = -fx * baseline`), and
`images/` (a preview every 10th sample).


## Agent control

`scripts/agent_control.py` lets a Claude or OpenAI model drive the end effector from camera images. Each step it sends the
last `--history_length` observations, gets back one structured pose delta (cm / deg, plus gripper open/closed), caps
it per step, clamps it to `WORKSPACE_BOUNDS`, and tracks the target with `OSC_POSE_PRECISE` from our deoxys fork (per-tick
Levenberg-Marquardt IK steps through `JOINT_IMPEDANCE`; deoxys `OSC_POSE` barely moves the arm for small goals). There is no collision checking.
Steps execute immediately unless `--require-confirmation-before-action` is passed. Library code is in
`src/panda_utils/agent_control/`.

```bash
# Cameras running as above. Claude models need ANTHROPIC_API_KEY (or `ant auth login`); gpt-* / o<N>* models need OPENAI_API_KEY.
python scripts/agent_control.py --model claude-opus-5-5 --task-description "pick up the red cube" \
  --camera_ids zed zed2i zedm --history_length 3 --action-reference-frame ee --require-confirmation-before-action
```

`--action-reference-frame world` expresses deltas in the robot base frame, `ee` in the gripper's frame. Each run logs
the images sent, the model's actions and token usage to one timestamped folder,
`logs/agent_control/<task>__<MM-DD-YYYY__HH-MM-SS>/` (`step_000__<camera>.jpg`, `step_000.json`, ...).
