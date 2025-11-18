# panda_utils

This library contains code to do the following:
* Convert RGB and Depth frames from a realsense camera to a pointcloud
* Control the panda using the [deoxys_control](https://github.com/UT-Austin-RPL/deoxys_control) library. Notably, collision checking is performed before actions are sent to the robot. Collision checking is performed by the [Jrl2](https://github.com/jstmn/Jrl2) library. Collisions are checked between links, and primitives (cuboids, spheres) which can be placed in the scene.









## Installation

Almost always I recommend using [uv](https://docs.astral.sh/uv/) or [pixi](https://pixi.sh/latest/), but because there's so many libraries that need to be cloned locally and then installed in an editable mode, I recommend just sticking to a simple conda environment. 

```bash
echo "DEOXYS_DIR='/home/roc/Desktop/deoxys_control'" >> ~/.bashrc; source ~/.bashrc
echo "PYTHONPATH='$PYTHONPATH:$DEOXYS_DIR/deoxys/'" >> ~/.bashrc; source ~/.bashrc
echo "ROS_WS='/home/roc/Desktop/panda_ros_ws'" >> ~/.bashrc; source ~/.bashrc
echo "PANDA_UTILS_DIR='/path/to/this/repo'" >> ~/.bashrc; source ~/.bashrc

# Create and activate a conda environment called `panda`:
conda create -n hardware_env python=3.10 -y
conda activate hardware_env
pip install matplotlib rospkg
git clone https://github.com/jstmn/Jrl2 thirdparty/Jrl2
git clone https://github.com/jstmn/Jrl thirdparty/Jrl
pip install -e thirdparty/Jrl2
pip install -e thirdparty/Jrl
pip install -U -r ${DEOXYS_DIR}/deoxys/requirements.txt
pip install 'protobuf<=3.20.0'  # This will give warnings/errors, that's fine
pip install -e ${DEOXYS_DIR}/deoxys
pip install -e .


# Grounded-Segment-Anythin installation (needed for automatic extrinsics calibration)
git clone https://github.com/IDEA-Research/Grounded-Segment-Anything.git thirdparty/Grounded-Segment-Anything
conda install -c nvidia cuda-toolkit=12.8
sudo apt-get install -y libxcb-xinerama0 libxcb-xinerama0-dev libxkbcommon-x11-0 libxcb1 libxcb-render0 libxcb-shape0 libxcb-xfixes0
pip install --no-build-isolation -e thirdparty/Grounded-Segment-Anything/GroundingDINO
cd thirdparty/Grounded-Segment-Anything
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

# Validate with
python grounded_sam_demo.py \
  --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
  --grounded_checkpoint groundingdino_swint_ogc.pth \
  --sam_checkpoint sam_vit_h_4b8939.pth \
  --input_image assets/demo1.jpg \
  --output_dir "outputs" \
  --box_threshold 0.3 \
  --text_threshold 0.25 \
  --text_prompt "bear" \
  --device "cuda"


# Validate with
python grounded_sam_demo.py \
  --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
  --grounded_checkpoint groundingdino_swint_ogc.pth \
  --sam_checkpoint sam_vit_h_4b8939.pth \
  --input_image ../../tests/data/extrinsics_test_data.h5__color_image0.jpg \
  --output_dir "outputs" \
  --box_threshold 0.3 \
  --text_threshold 0.25 \
  --text_prompt "franka panda white robot arm with gripper" \
  --device "cuda"

#   --text_prompt "white robot arm" \
```