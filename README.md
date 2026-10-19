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

# Terminal 1 - ZED cameras
CAMS="zed2i zedm" bash ~/Desktop/ros2_ws/zed_tools/launch_all.sh

# Terminal 2 - Teleoperation + log demonstrations [Option 1]
python scripts/log_demonstration.py \
  --output_dir ~/Desktop/data/<OUTPUT_DIR> --description <DESCRIPTION> \
  --recording_rate_hz 20.0 --camera_ids zed2i zedm

# Terminal 2 - Teleoperation [Option 2]
python scripts/panda_teleop.py
```