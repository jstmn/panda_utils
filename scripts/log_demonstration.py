"""Teleoperated demonstration logging with ZED cameras (ROS 2).

Same UI, same HDF5 layout, same SpaceMouse teleop as the old ROS 1 logger.
Camera source is ZED.

What changed vs the ROS 1 logger:
  * ROS 1 (rospy) -> ROS 2 (rclpy). Only Humble is installed on this machine.
  * RealSense topics -> ZED topics, matching zed_tools/launch_all.sh, which launches with
    `namespace:=zed_multi camera_name:=<name>`, giving /zed_multi/<cam>/...
    NOTE: src/panda_utils/rgb_subscriber_ros2_zed.py uses the stock single-camera
    convention (/zed_<id>/zed_node/...) which does NOT match this rig's launcher.
    Override with --topic-template if your launch differs; --list-topics shows what is live.
  * ZED depth is 32FC1 in metres; converted to uint16 millimetres so the HDF5 matches
    what the RealSense pipeline produced. ZED colour is bgra8; converted to RGB.
  * Gripper is optional (--no-gripper). NOTE: this rig DOES have a Franka Hand -- verified
    2026-09-06, gripper-interface runs on the NUC and FrankaInterface.last_gripper_q reads
    0.0804 (fully open). Do NOT pass --no-gripper here: it drops the finger columns, and
    the Square task needs the grasp signal. The flag exists for a genuinely bare flange.
  * Does not import panda_utils.utils / .constants: both pull in ROS 1 (rospy) and open3d
    and are unimportable here. The few helpers needed are inlined below.

Cameras must already be running:
    bash ~/Desktop/ros2_ws/zed_tools/launch_all.sh
Robot must already be running on the NUC (see README_SETUP.md).

Usage:
    source /opt/ros/humble/setup.bash
    source ~/Desktop/ros2_ws/install/setup.bash
    source ~/anaconda3/bin/activate deoxys

    # This
    python scripts/log_demonstration.py \
        --output_dir ~/Desktop/data/zed_demos --description raise_cube \
        --recording_rate_hz 15.0 --camera_ids zed zed2i zedm --no-gripper --no-depth

    # Or this
    python scripts/log_demonstration.py \
        --output_dir ~/Desktop/data/MPC_oct15 --description LiftPegUpright \
        --recording_rate_hz 15.0 --camera_ids zed2i zedm


    # Discover live camera topics
    python scripts/log_demonstration.py --list-topics

For the aux-pose (real2sim) pipeline: depth ON, and the D435 as the scene camera, so the
frames FoundationPose labels are the frames the policy sees. See scripts/r2st/r2st_convert_demo.py.

ros2 launch realsense2_camera rs_launch.py camera_namespace:=camera camera_name:=d435 \
    align_depth.enable:=true rgb_camera.color_profile:=640x480x30 \
    depth_module.depth_profile:=640x480x30

python scripts/log_demonstration.py \
    --output_dir ~/Desktop/data/aux_demos --description lift_peg \
    --recording_rate_hz 15.0 --camera_ids d435=realsense zedm --no-gripper
"""

from __future__ import annotations

import argparse
from collections import namedtuple
import json
import select
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Callable

import cv2
import h5py
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image, JointState
from termcolor import cprint

from deoxys.experimental.motion_utils import reset_joints_to
from deoxys.franka_interface import FrankaInterface
from deoxys.utils.config_utils import get_default_controller_config
from deoxys.utils.input_utils import input2action
from deoxys.utils.io_devices import SpaceMouse

# panda_utils.constants pulls in ROS 1; fall back to the measured values for this rig's
# SpaceNavigator (046d:c626). Keep in sync with panda_utils/constants.py.
try:
    from panda_utils.constants import SPACE_MOUSE_PRODUCT_ID, SPACE_MOUSE_VENDOR_ID
except Exception:  # noqa: BLE001 - any import error means the ROS 1 chain is unavailable
    SPACE_MOUSE_VENDOR_ID = 1133   # 0x046D
    SPACE_MOUSE_PRODUCT_ID = 50726  # 0xC626

CONTROLLER_TYPE = "OSC_POSE"

RESET_JOINT_POSITIONS = [0.09162008, -0.19826458, -0.01990020, -2.47322699, -0.01307074, 2.30396583, 0.84809397]




# Topic layouts, one per camera vendor. ZED and realsense2_camera agree on message types and
# on nothing else: the names differ, and so does which topic carries colour-registered depth.
#
#   ZED        zed_tools/launch_all.sh -> namespace:=zed_multi, camera_name:=<zed|zed2i|zedm>.
#              `depth/depth_registered` is 32FC1 metres, already on the left-rgb grid.
#   RealSense  ros2 launch realsense2_camera rs_launch.py camera_namespace:=camera \
#                  camera_name:=d435 align_depth.enable:=true -> /camera/d435/...
#              `aligned_depth_to_color/image_raw` is 16UC1 millimetres on the colour grid.
#              The raw `depth/image_rect_raw` is on the DEPTH sensor's grid and must not be
#              used here: real2sim-toolkit's merged layout, and decode_depth below, both
#              assume depth and colour share rays. align_depth.enable:=true is not optional.
#
# decode_depth handles both encodings already, so a layout is the only thing that varies.
CameraLayout = namedtuple("CameraLayout", "template color depth color_info depth_info")

LAYOUTS = {
    "zed": CameraLayout(
        template="/zed_multi/{cam}",
        color="rgb/image_rect_color",
        depth="depth/depth_registered",
        color_info="rgb/camera_info",
        depth_info="depth/camera_info",
    ),
    "realsense": CameraLayout(
        template="/camera/{cam}",
        color="color/image_raw",
        depth="aligned_depth_to_color/image_raw",
        color_info="color/camera_info",
        depth_info="aligned_depth_to_color/camera_info",
    ),
}
DEFAULT_LAYOUT = "zed"
DEFAULT_TOPIC_TEMPLATE = LAYOUTS[DEFAULT_LAYOUT].template


def parse_camera_spec(spec: str) -> tuple[str, str]:
    """'zed2i' -> ('zed2i', 'zed');  'd435=realsense' -> ('d435', 'realsense')."""
    cam_id, _, layout = spec.partition("=")
    layout = layout or DEFAULT_LAYOUT
    if layout not in LAYOUTS:
        raise ValueError(f"unknown layout {layout!r} in {spec!r}; known: {sorted(LAYOUTS)}")
    return cam_id, layout

JOINT_NAMES = [f"panda_joint{i}" for i in range(1, 8)]
GRIPPER_JOINT_NAMES = ["panda_finger_joint1", "panda_finger_joint2"]
JOINT_PUBLISH_HZ = 50.0
TELEOP_HZ = 50.0


def make_qos(best_effort: bool) -> QoSProfile:
    """QoS for ZED image topics.

    The wrapper defaults to RELIABLE. A RELIABLE subscriber never receives from a
    BEST_EFFORT publisher, which shows up as "no data" with no error at all - hence the
    --qos-best-effort escape hatch.
    """
    return QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT if best_effort else ReliabilityPolicy.RELIABLE,
        history=HistoryPolicy.KEEP_LAST,
        depth=10,
        durability=DurabilityPolicy.VOLATILE,
    )


def decode_color(msg: Image) -> np.ndarray:
    """ZED colour Image -> HxWx3 uint8 RGB."""
    buf = np.frombuffer(msg.data, dtype=np.uint8)
    if msg.encoding in ("bgra8", "rgba8"):
        img = buf.reshape(msg.height, msg.width, 4)
        img = img[:, :, [2, 1, 0]] if msg.encoding == "bgra8" else img[:, :, :3]
    elif msg.encoding in ("bgr8", "rgb8"):
        img = buf.reshape(msg.height, msg.width, 3)
        if msg.encoding == "bgr8":
            img = img[:, :, ::-1]
    else:
        raise ValueError(f"unsupported colour encoding: {msg.encoding}")
    return np.ascontiguousarray(img)


def decode_depth(msg: Image) -> np.ndarray:
    """ZED depth Image -> HxW uint16 millimetres.

    ZED publishes 32FC1 metres by default; OpenNI mode publishes 16UC1 millimetres.
    Both are normalised to uint16 mm so the HDF5 matches the RealSense pipeline.
    """
    if msg.encoding in ("32FC1", "TYPE_32FC1"):
        depth_m = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)
        # NaN/inf mark invalid returns in ZED depth; store as 0 rather than wrapping.
        depth_m = np.nan_to_num(depth_m, nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(depth_m * 1000.0, 0, np.iinfo(np.uint16).max).astype(np.uint16)
    if msg.encoding in ("16UC1", "mono16"):
        return np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width).copy()
    raise ValueError(f"unsupported depth encoding: {msg.encoding}")


def camera_info_to_dict(msg: CameraInfo) -> dict:
    """Serialise CameraInfo without the ROS 1 to_public_dict helper."""
    return {
        "height": msg.height,
        "width": msg.width,
        "distortion_model": msg.distortion_model,
        "d": list(msg.d),
        "k": list(msg.k),
        "r": list(msg.r),
        "p": list(msg.p),
        "binning_x": msg.binning_x,
        "binning_y": msg.binning_y,
        "roi": {
            "x_offset": msg.roi.x_offset,
            "y_offset": msg.roi.y_offset,
            "height": msg.roi.height,
            "width": msg.roi.width,
            "do_rectify": msg.roi.do_rectify,
        },
    }


def save_combined_rgbs(images: list[np.ndarray], camera_ids: list[str], out_path: Path) -> None:
    """Write a labelled side-by-side montage of one timestep's RGB frames.

    Local reimplementation of panda_utils.utils.save_combined_rgbs (that module imports rospy).
    """
    if not images:
        return
    target_h = min(img.shape[0] for img in images)
    panels = []
    for img, cam_id in zip(images, camera_ids):
        scale = target_h / img.shape[0]
        panel = cv2.resize(img, (int(img.shape[1] * scale), target_h))
        panel = cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)  # cv2.imwrite expects BGR
        cv2.putText(panel, cam_id, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
        panels.append(panel)
    cv2.imwrite(str(out_path), np.hstack(panels))


class ZedDataCollector(Node):
    """Subscribes to ZED colour (and optionally depth) and records samples with joint states."""

    def __init__(self, output_dir: Path, description: str, camera_ids: list[str],
                 robot_interface: FrankaInterface, topic_template: str,
                 has_gripper: bool, best_effort: bool, record_depth: bool = True,
                 camera_layouts: dict[str, str] | None = None,
                 no_depth_cameras: set[str] | None = None):
        super().__init__("log_demonstration")
        for cam_id in camera_ids:
            assert "_" not in cam_id, f"Camera id cannot contain underscore: {cam_id}"

        self.output_dir = output_dir / f"{description}__{datetime.now().strftime('%m-%d_%H-%M-%S')}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._camera_ids = camera_ids
        self._robot = robot_interface
        self._has_gripper = has_gripper
        # Depth is per-camera, not global. The wrist ZED is an RGB policy input only --
        # nothing in the aux pipeline unprojects it -- and its 1280x720 uint16 depth is by
        # far the largest thing in the file (~1.5 GB of a 2.5 GB demo). Dropping just that
        # channel takes a 60-demo session from ~180 GB to ~60 GB without touching the D435
        # depth the labels actually depend on.
        skip = set(no_depth_cameras or ())
        unknown = skip - set(camera_ids)
        assert not unknown, f"--no-depth-cameras names cameras that are not recorded: {sorted(unknown)}"
        self._depth_cams = set() if not record_depth else set(camera_ids) - skip
        self._is_recording = False

        self.joint_q: list[np.ndarray] = []
        self.joint_dq: list[np.ndarray] = []
        self.camera_data: dict[str, dict] = {}
        for cam_id in camera_ids:
            self.camera_data[cam_id] = {
                "depth_images": [], "color_images": [],
                "depth_camera_info": None, "color_camera_info": None,
                "_latest_depth": None, "_latest_color": None,
            }

        qos = make_qos(best_effort)
        self._subscribers = []
        camera_layouts = camera_layouts or {}
        for cam_id in camera_ids:
            layout = LAYOUTS[camera_layouts.get(cam_id, DEFAULT_LAYOUT)]
            # --topic-template still overrides, but only for the layout it was written for;
            # applying a ZED namespace to a RealSense camera would subscribe to nothing and
            # report it as "no frames", which is the failure this whole table exists to avoid.
            template = (topic_template if camera_layouts.get(cam_id, DEFAULT_LAYOUT) == DEFAULT_LAYOUT
                        else layout.template)
            ns = template.format(cam=cam_id).rstrip("/")
            self._subscribers += [
                self.create_subscription(Image, f"{ns}/{layout.color}",
                                         lambda m, c=cam_id: self._color_cb(m, c), qos),
                self.create_subscription(CameraInfo, f"{ns}/{layout.color_info}",
                                         lambda m, c=cam_id: self._color_info_cb(m, c), qos),
            ]
            if cam_id in self._depth_cams:
                self._subscribers += [
                    self.create_subscription(Image, f"{ns}/{layout.depth}",
                                             lambda m, c=cam_id: self._depth_cb(m, c), qos),
                    self.create_subscription(CameraInfo, f"{ns}/{layout.depth_info}",
                                             lambda m, c=cam_id: self._depth_info_cb(m, c), qos),
                ]
                cprint(f"   subscribed: {ns}/{{{layout.color}, {layout.depth}}}", "white")
            else:
                cprint(f"   subscribed: {ns}/{layout.color}  (depth off)", "white")

    # -- callbacks ---------------------------------------------------------

    def _color_cb(self, msg: Image, cam_id: str) -> None:
        try:
            self.camera_data[cam_id]["_latest_color"] = decode_color(msg)
        except ValueError as exc:
            self.get_logger().warn(f"{cam_id}: {exc}")

    def _depth_cb(self, msg: Image, cam_id: str) -> None:
        try:
            self.camera_data[cam_id]["_latest_depth"] = decode_depth(msg)
        except ValueError as exc:
            self.get_logger().warn(f"{cam_id}: {exc}")

    def _color_info_cb(self, msg: CameraInfo, cam_id: str) -> None:
        if self.camera_data[cam_id]["color_camera_info"] is None:
            self.camera_data[cam_id]["color_camera_info"] = camera_info_to_dict(msg)

    def _depth_info_cb(self, msg: CameraInfo, cam_id: str) -> None:
        if self.camera_data[cam_id]["depth_camera_info"] is None:
            self.camera_data[cam_id]["depth_camera_info"] = camera_info_to_dict(msg)

    # -- recording ---------------------------------------------------------

    @property
    def is_recording(self) -> bool:
        return self._is_recording

    def _camera_ready(self, cam_id: str) -> bool:
        if self.camera_data[cam_id]["_latest_color"] is None:
            return False
        if cam_id in self._depth_cams and self.camera_data[cam_id]["_latest_depth"] is None:
            return False
        return True

    def wait_for_frames(self, timeout_s: float = 20.0) -> bool:
        """Block until every camera has delivered colour (and depth, if enabled)."""
        need = ", ".join(f"{c}:{'colour+depth' if c in self._depth_cams else 'colour'}"
                         for c in self._camera_ids)
        cprint(f"⏳ Waiting for first {need} frames from: {self._camera_ids}", "cyan")
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            missing = [c for c in self._camera_ids if not self._camera_ready(c)]
            if not missing:
                cprint("✅ All cameras streaming", "green")
                return True
            time.sleep(0.2)

        missing = [c for c in self._camera_ids if not self._camera_ready(c)]
        cprint(f"❌ No frames from: {missing} after {timeout_s:.0f}s", "red")
        cprint("   Check the cameras are up:  bash ~/Desktop/ros2_ws/zed_tools/verify.sh", "red")
        cprint("   Check topic names:         python scripts/log_demonstration.py --list-topics", "red")
        cprint("   If topics exist but no data arrives, retry with --qos-best-effort", "red")
        return False

    def start_recording(self) -> None:
        self._is_recording = True
        self.joint_q, self.joint_dq = [], []
        for cam_id in self._camera_ids:
            self.camera_data[cam_id]["depth_images"] = []
            self.camera_data[cam_id]["color_images"] = []
        cprint("🎬 Started recording", "green")

    def stop_recording(self) -> None:
        self._is_recording = False
        cprint("⏹️  Stopped recording", "yellow")

    def record_sample(self) -> tuple[list[str], bool]:
        """Record one synchronised sample. Returns (cameras captured, joint state captured).

        All-or-nothing: joint state and every camera are committed together, or nothing is.
        A partial commit would leave joint_states_q longer than the image datasets, silently
        misaligning observations against actions in the saved HDF5.
        """
        if not self._is_recording:
            return [], False

        # Read joints straight off the deoxys interface rather than round-tripping through
        # ROS as the ROS 1 logger did - fewer moving parts and no publisher/subscriber race.
        arm_q = self._robot.last_q
        if arm_q is None:
            return [], False
        q = list(arm_q)
        dq = list(self._robot.last_dq)
        if self._has_gripper:
            if self._robot.last_gripper_q is None:
                return [], False
            gripper_angle = float(self._robot.last_gripper_q) / 2.0
            q += [gripper_angle, gripper_angle]
            dq += [0.0, 0.0]

        # Snapshot every camera before committing anything.
        frames = {}
        for cam_id in self._camera_ids:
            color = self.camera_data[cam_id]["_latest_color"]
            if color is None:
                continue
            if cam_id in self._depth_cams:
                depth = self.camera_data[cam_id]["_latest_depth"]
                if depth is None:
                    continue
                frames[cam_id] = (color, depth)
            else:
                frames[cam_id] = (color, None)

        if len(frames) < len(self._camera_ids):
            return list(frames), False  # caller reports the missing cameras and stops

        self.joint_q.append(np.asarray(q, dtype=np.float64))
        self.joint_dq.append(np.asarray(dq, dtype=np.float64))
        for cam_id, (color, depth) in frames.items():
            self.camera_data[cam_id]["color_images"].append(color)
            if cam_id in self._depth_cams and depth is not None:
                self.camera_data[cam_id]["depth_images"].append(depth)

        return list(frames), True

    # -- saving ------------------------------------------------------------

    def save(self, traj_name: str) -> None:
        if len(self.joint_q) == 0:
            cprint("⚠️  0 joint states logged, nothing saved.", "red")
            return

        h5_path = self.output_dir / f"{traj_name}.h5"
        with h5py.File(h5_path, "a") as f:
            q = np.array(self.joint_q)
            dq = np.array(self.joint_dq)
            f.create_dataset("joint_states_q", data=q)
            f.create_dataset("joint_states_dq", data=dq)

            if self._has_gripper:
                gripper_q = q[:, -1]
                if gripper_q.max() - gripper_q.min() < 0.01:
                    cprint(f"ERROR: GRIPPER DOESN'T SEEM TO BE MOVING. "
                           f"RANGE: {gripper_q.max() - gripper_q.min()}", "red")
                    cprint("If the gripper was moving, restart the franka via https://172.16.0.2/desk/", "red")

            for cam_id in self._camera_ids:
                if len(self.camera_data[cam_id]["color_images"]) == 0:
                    continue
                f.create_dataset(f"{cam_id}__color_image",
                                 data=np.array(self.camera_data[cam_id]["color_images"]))
                if cam_id in self._depth_cams and len(self.camera_data[cam_id]["depth_images"]) > 0:
                    f.create_dataset(f"{cam_id}__depth_image",
                                     data=np.array(self.camera_data[cam_id]["depth_images"]))

        for cam_id in self._camera_ids:
            kinds = [("color", "color_camera_info")]
            if cam_id in self._depth_cams:
                kinds.append(("depth", "depth_camera_info"))
            for kind, key in kinds:
                if self.camera_data[cam_id][key] is not None:
                    with open(self.output_dir / f"{cam_id}_camera_info_{kind}.json", "w") as fh:
                        json.dump(self.camera_data[cam_id][key], fh, indent=2)

        # Sample montages, one per timestep, for eyeballing the demo afterwards.
        img_dir = self.output_dir / "images"
        img_dir.mkdir(exist_ok=True)
        cprint(f"\n📸 Saving sample images to {img_dir}...", "cyan")

        available = {c: self.camera_data[c]["color_images"] for c in self._camera_ids
                     if len(self.camera_data[c]["color_images"]) > 0}
        for cam_id in self._camera_ids:
            n_c = len(self.camera_data[cam_id]["color_images"])
            if cam_id in self._depth_cams:
                n_d = len(self.camera_data[cam_id]["depth_images"])
                cprint(f"   Camera {cam_id}: {n_d} depth, {n_c} color images collected", "white")
            else:
                cprint(f"   Camera {cam_id}: {n_c} color images collected (depth off)", "white")

        if available:
            num_samples = min(len(v) for v in available.values())
            ids = [c for c in self._camera_ids if c in available]
            for i in range(num_samples):
                save_combined_rgbs([available[c][i] for c in ids], ids,
                                   img_dir / f"combined_rgb__{traj_name}__{i:03d}.png")
            cprint(f"   Saved {num_samples} combined RGB images as PNG", "white")

        cprint(f"💾 Saved {len(self.joint_q)} samples to {h5_path}", "green")


def joint_publish_thread_target(node: Node, robot: FrankaInterface, has_gripper: bool,
                                should_run: Callable[[], bool]) -> None:
    """Publish /panda/joint_states for rviz and any other consumer."""
    pub = node.create_publisher(JointState, "/panda/joint_states", 10)
    names = JOINT_NAMES + (GRIPPER_JOINT_NAMES if has_gripper else [])
    period = 1.0 / JOINT_PUBLISH_HZ

    while rclpy.ok() and should_run():
        time.sleep(period)
        arm_q = robot.last_q
        if arm_q is None:
            continue
        position = list(arm_q)
        velocity = list(robot.last_dq)
        if has_gripper:
            if robot.last_gripper_q is None:
                continue
            gripper_angle = float(robot.last_gripper_q) / 2.0
            position += [gripper_angle, gripper_angle]
            velocity += [0.0, 0.0]

        msg = JointState()
        msg.header.stamp = node.get_clock().now().to_msg()
        msg.name = names
        msg.position = position
        msg.velocity = velocity
        msg.effort = [0.0] * len(names)
        pub.publish(msg)


def teleop_control_thread_target(robot: FrankaInterface, device: SpaceMouse,
                                 should_continue: Callable[[], bool],
                                 should_run: Callable[[], bool]) -> None:
    controller_cfg = get_default_controller_config(controller_type=CONTROLLER_TYPE)
    period = 1.0 / TELEOP_HZ
    cprint("🎮 Teleop active - use SpaceMouse to control robot", "cyan")

    n_errors = 0
    while rclpy.ok() and should_run():
        time.sleep(period)
        if not should_continue():
            continue
        try:
            action, _grasp = input2action(device=device, controller_type=CONTROLLER_TYPE)
            # input2action returns None whenever the SpaceMouse's reset flag is set, i.e.
            # the RIGHT button was pressed. Passing that straight through is what produced
            # "too many indices for array: array is 0-dimensional": deoxys wraps it with
            # np.array(None), which is a 0-d object array, then does action[0:3] *= ...
            # (franka_interface.py:284). Skipping the tick is the correct response -- a
            # reset request is not a motion command.
            if action is None:
                continue
            robot.control(controller_type=CONTROLLER_TYPE, action=action, controller_cfg=controller_cfg)
        except Exception as exc:  # noqa: BLE001 - never let one bad tick end teleop
            n_errors += 1
            # Breaking here used to strand the arm for the rest of the session: a single
            # transient fault killed the thread and nothing moved again. Report and carry
            # on, and only give up if it is clearly not transient.
            if n_errors <= 3 or n_errors % 100 == 0:
                cprint(f"Teleop error ({n_errors}): {exc}", "red")
                if n_errors == 1:
                    traceback.print_exc()
            if n_errors >= 500:
                cprint("Teleop: 500 consecutive errors, stopping teleop thread.", "red")
                break
        else:
            n_errors = 0


def list_topics() -> int:
    """Print live camera-ish topics so topic-name mismatches are obvious."""
    rclpy.init(args=None)
    node = rclpy.create_node("log_demonstration_topic_lister")
    time.sleep(2.0)  # discovery needs a moment
    names = sorted(n for n, _ in node.get_topic_names_and_types())
    cam = [n for n in names if any(k in n.lower() for k in ("zed", "image", "depth", "camera_info"))]
    cprint("\nLive camera-related topics:", "cyan")
    if cam:
        for n in cam:
            print(f"   {n}")
    else:
        cprint("   (none found - are the cameras running?)", "yellow")
        cprint("   bash ~/Desktop/ros2_ws/zed_tools/launch_all.sh", "yellow")
    cprint("\nExpected colour topic per layout (--camera_ids <id>=<layout>):", "cyan")
    for name, layout in sorted(LAYOUTS.items()):
        marker = "  [default]" if name == DEFAULT_LAYOUT else ""
        print(f"   {name:<10} {layout.template.format(cam='<id>')}/{layout.color}{marker}")
    cprint("Override with --topic-template if yours differ.\n", "cyan")
    node.destroy_node()
    rclpy.shutdown()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--description", type=str, default="")
    parser.add_argument("--recording_rate_hz", type=float, default=15.0)
    parser.add_argument("--camera_ids", type=str, nargs="*", default=None,
                        help="Camera names as launched, optionally '<name>=<layout>' with layout "
                             f"in {sorted(LAYOUTS)} (default {DEFAULT_LAYOUT}). "
                             "e.g. zedm d435=realsense")
    parser.add_argument("--topic-template", type=str, default=DEFAULT_TOPIC_TEMPLATE,
                        help=f"namespace template, '{{cam}}' substituted (default: {DEFAULT_TOPIC_TEMPLATE})")
    parser.add_argument("--interface-cfg", type=str, default=None)
    parser.add_argument("--vendor-id", type=int, default=SPACE_MOUSE_VENDOR_ID)
    parser.add_argument("--product-id", type=int, default=SPACE_MOUSE_PRODUCT_ID)
    parser.add_argument("--reset_joints", type=str, default="true")
    parser.add_argument("--run_teleop", type=str, default="true")
    parser.add_argument("--no-gripper", action="store_true",
                        help="rig has no gripper: do not wait for or record gripper state")
    parser.add_argument("--no-depth-cameras", type=str, nargs="*", default=None,
                        help="cameras to record WITHOUT depth, e.g. `--no-depth-cameras zedm`. "
                             "The wrist ZED is an RGB policy input only, and its 720p depth is "
                             "most of the file size; the D435 depth the aux labels need is kept.")
    parser.add_argument("--no-depth", action="store_true",
                        help="do not subscribe to or record depth (colour + joints only)")
    parser.add_argument("--qos-best-effort", action="store_true",
                        help="use BEST_EFFORT QoS if topics exist but no frames arrive")
    parser.add_argument("--list-topics", action="store_true", help="list live topics and exit")
    args = parser.parse_args()

    if args.list_topics:
        return list_topics()
    if not args.output_dir:
        parser.error("--output_dir is required (or use --list-topics)")

    args.run_teleop = str(args.run_teleop).lower() == "true"
    args.reset_joints = str(args.reset_joints).lower() == "true"
    has_gripper = not args.no_gripper
    record_depth = not args.no_depth
    no_depth_cameras = set(args.no_depth_cameras or ())

    try:
        parsed = [parse_camera_spec(spec) for spec in (args.camera_ids or [])]
    except ValueError as exc:
        parser.error(str(exc))
    camera_ids = [cam_id for cam_id, _ in parsed]
    camera_layouts = dict(parsed)
    if not camera_ids:
        cprint("No camera ids were provided! Only joint states will be recorded", "yellow")

    project_dir = Path(__file__).parent.parent
    interface_cfg = args.interface_cfg or f"{project_dir}/configs/charmander.yml"

    try:
        device = SpaceMouse(vendor_id=args.vendor_id, product_id=args.product_id)
        device.start_control()
    except OSError as exc:
        cprint(f"❌ Failed to initialize SpaceMouse: {exc}", "red")
        cprint("   Check the HID driver bind and udev rule - see README_SETUP.md", "red")
        return 1

    cprint("\n" + "=" * 60, "cyan")
    cprint("🤖 Demo Collection with Teleop (ZED / ROS 2)", "cyan", attrs=["bold"])
    cprint("=" * 60, "cyan")
    cprint(f"Output   : {args.output_dir}", "white")
    cprint(f"Cameras  : {camera_ids or '(none)'}", "white")
    if not record_depth:
        cprint("Depth    : DISABLED for every camera (--no-depth)", "white")
    else:
        per_cam = ", ".join(f"{c}:{'off' if c in no_depth_cameras else 'on'}" for c in camera_ids)
        cprint(f"Depth    : {per_cam}", "white")
    cprint(f"Gripper  : {'enabled' if has_gripper else 'DISABLED (--no-gripper)'}", "white")
    cprint("=" * 60 + "\n", "cyan")

    rclpy.init(args=None)

    cprint("🔗 Initializing robot interface...", "cyan")
    robot_interface = FrankaInterface(interface_cfg, use_visualizer=False)
    robot_interface._state_buffer = []

    deadline = time.time() + 10.0
    while time.time() < deadline:
        if robot_interface.last_q is not None and (not has_gripper or robot_interface.last_gripper_q is not None):
            break
        time.sleep(0.1)
    else:
        cprint("❌ Deoxys is not streaming robot state. Is franka-interface running on the NUC?", "red")
        robot_interface.close()
        rclpy.shutdown()
        return 1
    cprint("✅ Robot interface ready", "green")

    collector = ZedDataCollector(Path(args.output_dir).expanduser(), args.description, camera_ids,
                                 robot_interface, args.topic_template, has_gripper,
                                 args.qos_best_effort, record_depth=record_depth,
                                 no_depth_cameras=no_depth_cameras,
                                 camera_layouts=camera_layouts)

    running = True
    should_continue_flag = True

    def spin_target() -> None:
        while running and rclpy.ok():
            rclpy.spin_once(collector, timeout_sec=0.05)

    threading.Thread(target=spin_target, daemon=True).start()

    if camera_ids and not collector.wait_for_frames():
        running = False
        robot_interface.close()
        collector.destroy_node()
        rclpy.shutdown()
        return 1

    threading.Thread(target=joint_publish_thread_target,
                     args=(collector, robot_interface, has_gripper, lambda: running),
                     daemon=True).start()

    if args.run_teleop:
        threading.Thread(target=teleop_control_thread_target,
                         args=(robot_interface, device, lambda: should_continue_flag, lambda: running),
                         daemon=True).start()

    demo_index = 0
    period = 1.0 / args.recording_rate_hz
    exit_code = 0

    try:
        while rclpy.ok():
            cprint(f"\n{'='*60}", "cyan")
            cprint(f"📝 DEMO #{demo_index}", "cyan", attrs=["bold"])
            cprint(f"{'='*60}", "cyan")

            if args.reset_joints:
                cprint("🔄 Resetting robot...", "yellow")
                should_continue_flag = False
                time.sleep(0.1)
                reset_joints_to(robot_interface, RESET_JOINT_POSITIONS)
                should_continue_flag = True

            input("\n▶️  Press ENTER to START recording demo...")
            collector.start_recording()
            print("\n⏸️  Press ENTER to STOP recording demo...")

            while True:
                if select.select([sys.stdin], [], [], 0.0)[0] == [sys.stdin]:
                    if sys.stdin.read(1) == "\n":
                        collector.stop_recording()
                        break

                time.sleep(period)
                found_cams, joint_recorded = collector.record_sample()
                missing = [c for c in camera_ids if c not in found_cams]
                if missing:
                    kind = "RGBD" if collector._depth_cams else "RGB"
                    cprint(f"❌ {kind} data missing from: {missing}. Stopping.", "red")
                    collector.stop_recording()
                    exit_code = 1
                    break
                if not joint_recorded:
                    cprint("❌ Joint states missing. Stopping.", "red")
                    collector.stop_recording()
                    exit_code = 1
                    break

            cprint(f"💾 Saving data to {collector.output_dir}", "white")
            collector.save(traj_name=f"traj_{demo_index}")
            cprint(f"✅ Demo #{demo_index} complete!", "green")

            if exit_code:
                break
            if input("\n❓ Collect another demo? (y/n): ").strip().lower() != "y":
                break
            demo_index += 1
    except KeyboardInterrupt:
        cprint("\n⏹️  Interrupted", "yellow")
    finally:
        cprint(f"\n🏁 Collection complete! Total demos: {demo_index + 1}", "green", attrs=["bold"])
        running = False
        should_continue_flag = False
        time.sleep(0.2)
        try:
            robot_interface.close()
        except Exception:  # noqa: BLE001 - never mask the original failure during cleanup
            pass
        collector.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
