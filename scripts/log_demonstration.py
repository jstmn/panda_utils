"""Teleoperated demonstration logging with ZED cameras (ROS 2).

Same UI and SpaceMouse teleop as the old ROS 1 logger. Camera source is ZED. Each demo is
written in real2sim-toolkit's ManiSkill-style `obs/` stereo layout (see ZedDataCollector.save):
every ZED is recorded as a rectified left/right RGB pair plus the ZED's own depth.

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
        --recording_rate_hz 15.0 --camera_ids zed zed2i zedm --zed_resolution VGA --no-gripper --no-depth

    # Or this
    python scripts/log_demonstration.py \
        --output_dir ~/Desktop/data/MPC_oct15 --description LiftPegUpright \
        --recording_rate_hz 15.0 --camera_ids zed2i zedm --zed_resolution VGA


    # Discover live camera topics
    python scripts/log_demonstration.py --list-topics

For the aux-pose (real2sim) pipeline: depth ON, and the D435 as the scene camera, so the
frames FoundationPose labels are the frames the policy sees. See scripts/r2st/r2st_convert_demo.py
(which still reads the older flat layout, `<cam>__color_image` etc. at the file root).

ros2 launch realsense2_camera rs_launch.py camera_namespace:=camera camera_name:=d435 \
    align_depth.enable:=true rgb_camera.color_profile:=640x480x30 \
    depth_module.depth_profile:=640x480x30

python scripts/log_demonstration.py \
    --output_dir ~/Desktop/data/aux_demos --description lift_peg \
    --recording_rate_hz 15.0 --camera_ids d435=realsense zedm --zed_resolution VGA --no-gripper
"""

from __future__ import annotations

import argparse
import json
import os
import select
import sys
import threading
import time
import traceback
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Callable
from zoneinfo import ZoneInfo

import cv2
import h5py
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image, JointState
from termcolor import cprint

from deoxys.experimental.motion_utils import reset_joints_to
from deoxys.franka_interface import FrankaInterface
from deoxys.utils.config_utils import get_default_controller_config
from deoxys.utils.input_utils import input2action
from deoxys.utils.io_devices import SpaceMouse

from panda_utils.ros2_cameras import (
    DEFAULT_LAYOUT, LAYOUTS, decode_color, decode_depth, make_qos, parse_camera_spec, stamp_to_ns,
)

# panda_utils.constants pulls in ROS 1; fall back to the measured values for this rig's
# SpaceNavigator (046d:c626). Keep in sync with panda_utils/constants.py.
try:
    from panda_utils.constants import SPACE_MOUSE_PRODUCT_ID, SPACE_MOUSE_VENDOR_ID
except Exception:  # noqa: BLE001 - any import error means the ROS 1 chain is unavailable
    SPACE_MOUSE_VENDOR_ID = 1133   # 0x046D
    SPACE_MOUSE_PRODUCT_ID = 50726  # 0xC626

CONTROLLER_TYPE = "OSC_POSE"

RESET_JOINT_POSITIONS = [0.09162008, -0.19826458, -0.01990020, -2.47322699, -0.01307074, 2.30396583, 0.84809397]

# Timing. One timestamp per sample per stream, aligned with obs/qpos. Two kinds of clock, and
# only timestamps on the same clock can be compared:
#   obs/sensor_data/<cam>/**timestamp_ms               host wall clock, float64 ms since the Unix epoch
#   obs/robot_arm_ns, obs/robot_gripper_ns             Franka's own clocks, int64 ns since they booted
# Internally everything is kept as int64 ns (time.time_ns(), ROS header stamps) until save().
# Human-readable start/end times are written in LOCAL_TZ, which handles PST/PDT itself.
LOCAL_TZ = ZoneInfo("America/Los_Angeles")
HOST_CLOCK = "host wall clock, ns since Unix epoch"
ROBOT_ARM_CLOCK = ("Franka controller (libfranka RobotState::time), ns since controller boot, "
                   "1 ms resolution. Only comparable with itself.")
ROBOT_GRIPPER_CLOCK = ("Franka gripper (libfranka GripperState::time), ns since gripper boot, "
                       "1 ms resolution. Only comparable with itself.")




DEFAULT_TOPIC_TEMPLATE = LAYOUTS[DEFAULT_LAYOUT].template


# Image size (height, width) the USB ZEDs (ZED, ZED 2i, ZED Mini) publish for each
# `general.grab_resolution`, set in ros2_ws/src/zed-ros2-wrapper/zed_wrapper/config/<model>.yaml.
# Assumes `pub_resolution: 'NATIVE'` (common_stereo.yaml); 'CUSTOM' downscales and fails the check.
ZED_RESOLUTIONS = {
    "HD2K": (1242, 2208),
    "HD1080": (1080, 1920),
    "HD720": (720, 1280),
    "VGA": (376, 672),
}

JOINT_NAMES = [f"panda_joint{i}" for i in range(1, 8)]
GRIPPER_JOINT_NAMES = ["panda_finger_joint1", "panda_finger_joint2"]
JOINT_PUBLISH_HZ = 50.0
MONTAGE_EVERY = 10  # save the preview montage PNG for every Nth sample only
# Recent left / right frames kept per stereo camera, to pair them by header stamp (~0.3 s at 30 fps)
STEREO_PAIR_BUFFER = 10
TELEOP_HZ = 50.0


def ns_to_local_str(ns: int) -> str:
    """1792534323123456789 -> '2026-10-20T15:12:03.123-07:00 PDT'."""
    t = datetime.fromtimestamp(ns // 1_000_000_000, LOCAL_TZ).replace(microsecond=(ns // 1000) % 1_000_000)
    return f"{t.isoformat(timespec='milliseconds')} {t.tzname()}"


def stream_stats(stamps_ns: np.ndarray, sample_ns: np.ndarray | None = None) -> dict:
    """Rate / gap / repeat summary of one stream's per-sample timestamps.

    A repeat is the same message recorded on two consecutive samples (the logger takes the
    latest frame each tick and so outruns a slow stream). Age is how old the recorded message
    was when it was sampled, so is only computed for streams on the host clock.
    """
    gaps_ms = np.diff(stamps_ns) / 1e6
    span_s = (stamps_ns[-1] - stamps_ns[0]) / 1e9
    n_unique = len(np.unique(stamps_ns))
    stats = {
        "n": int(len(stamps_ns)),
        "rate_hz": round((n_unique - 1) / span_s, 2) if span_s > 0 else None,
        "max_gap_ms": round(float(gaps_ms.max()), 1) if len(gaps_ms) else None,
        "repeats": int((gaps_ms == 0).sum()),
    }
    if sample_ns is not None:
        age_ms = (sample_ns - stamps_ns) / 1e6
        stats["mean_age_ms"] = round(float(age_ms.mean()), 1)
        stats["max_age_ms"] = round(float(age_ms.max()), 1)
    return stats


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


FEED_WINDOW = "camera feed"


def combined_bgr(images: list[np.ndarray], camera_ids: list[str]) -> np.ndarray | None:
    """Labelled side-by-side BGR montage. `images` are RGB, one per camera."""
    if not images:
        return None
    target_h = min(img.shape[0] for img in images)
    panels = []
    for img, cam_id in zip(images, camera_ids):
        scale = target_h / img.shape[0]
        panel = cv2.resize(img, (int(img.shape[1] * scale), target_h))
        panel = cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)
        cv2.putText(panel, cam_id, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
        panels.append(panel)
    return np.hstack(panels)


def save_combined_rgbs(images: list[np.ndarray], camera_ids: list[str], out_path: Path) -> None:
    """Write a labelled side-by-side montage of one timestep's RGB frames.

    Local reimplementation of panda_utils.utils.save_combined_rgbs (that module imports rospy).
    """
    montage = combined_bgr(images, camera_ids)
    if montage is not None:
        cv2.imwrite(str(out_path), montage)


class ZedDataCollector(Node):
    """Subscribes to ZED colour (left), right, and optionally depth, and records samples with joint states."""

    def __init__(self, output_dir: Path, description: str, camera_ids: list[str],
                 robot_interface: FrankaInterface, topic_template: str,
                 has_gripper: bool, best_effort: bool, zed_resolution: str, record_depth: bool = True,
                 camera_layouts: dict[str, str] | None = None,
                 no_depth_cameras: set[str] | None = None):
        super().__init__("log_demonstration")
        for cam_id in camera_ids:
            assert "_" not in cam_id, f"Camera id cannot contain underscore: {cam_id}"

        self.output_dir = output_dir / f"{description}__{datetime.now().strftime('%m-%d-%Y__%H-%M-%S')}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._camera_ids = camera_ids
        self._robot = robot_interface
        self._has_gripper = has_gripper
        self._zed_resolution = zed_resolution
        self._zed_shape = ZED_RESOLUTIONS[zed_resolution]
        self._zed_cams = {c for c in camera_ids if (camera_layouts or {}).get(c, DEFAULT_LAYOUT) == "zed"}
        # Depth is per-camera, not global. The wrist ZED is an RGB policy input only --
        # nothing in the aux pipeline unprojects it -- and its uint16 depth is by far the
        # largest thing in the file. Dropping just that channel shrinks a session to roughly
        # a third without touching the D435 depth the labels actually depend on.
        skip = set(no_depth_cameras or ())
        unknown = skip - set(camera_ids)
        assert not unknown, f"--no-depth-cameras names cameras that are not recorded: {sorted(unknown)}"
        self._depth_cams = set() if not record_depth else set(camera_ids) - skip
        # Every camera whose layout has a right eye (the ZEDs) is recorded as a stereo pair.
        self._stereo_cams = {c for c in camera_ids
                             if LAYOUTS[(camera_layouts or {}).get(c, DEFAULT_LAYOUT)].right is not None}
        self._is_recording = False

        self.joint_q: list[np.ndarray] = []
        self.joint_dq: list[np.ndarray] = []
        self.sample_time_ns: list[int] = []
        self.robot_arm_ns: list[int] = []
        self.robot_gripper_ns: list[int] = []
        self.camera_data: dict[str, dict] = {}
        for cam_id in camera_ids:
            self.camera_data[cam_id] = {
                "depth_images": [], "color_images": [], "right_images": [],
                "depth_stamps_ns": [], "color_stamps_ns": [], "right_stamps_ns": [],
                "depth_camera_info": None, "color_camera_info": None, "right_camera_info": None,
                # (stamp_ns, image), replaced as one object so a frame and its stamp never mix
                "_latest_depth": None, "_latest_color": None,
                # recent (stamp_ns, image) of each eye, so record_sample can pick a matching pair
                "_recent_color": deque(maxlen=STEREO_PAIR_BUFFER),
                "_recent_right": deque(maxlen=STEREO_PAIR_BUFFER),
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
            if cam_id in self._stereo_cams:
                self._subscribers += [
                    self.create_subscription(Image, f"{ns}/{layout.right}",
                                             lambda m, c=cam_id: self._right_cb(m, c), qos),
                    self.create_subscription(CameraInfo, f"{ns}/{layout.right_info}",
                                             lambda m, c=cam_id: self._right_info_cb(m, c), qos),
                ]
            topics = [layout.color] + ([layout.right] if cam_id in self._stereo_cams else []) \
                + ([layout.depth] if cam_id in self._depth_cams else [])
            cprint(f"   subscribed: {ns}/{{{', '.join(topics)}}}", "white")

    # -- callbacks ---------------------------------------------------------

    def _color_cb(self, msg: Image, cam_id: str) -> None:
        try:
            frame = (stamp_to_ns(msg), decode_color(msg))
        except ValueError as exc:
            self.get_logger().warn(f"{cam_id}: {exc}")
            return
        self.camera_data[cam_id]["_latest_color"] = frame
        self.camera_data[cam_id]["_recent_color"].append(frame)

    def _right_cb(self, msg: Image, cam_id: str) -> None:
        try:
            self.camera_data[cam_id]["_recent_right"].append((stamp_to_ns(msg), decode_color(msg)))
        except ValueError as exc:
            self.get_logger().warn(f"{cam_id} right: {exc}")

    def _depth_cb(self, msg: Image, cam_id: str) -> None:
        try:
            self.camera_data[cam_id]["_latest_depth"] = (stamp_to_ns(msg), decode_depth(msg))
        except ValueError as exc:
            self.get_logger().warn(f"{cam_id}: {exc}")

    def _color_info_cb(self, msg: CameraInfo, cam_id: str) -> None:
        if self.camera_data[cam_id]["color_camera_info"] is None:
            self.camera_data[cam_id]["color_camera_info"] = camera_info_to_dict(msg)

    def _depth_info_cb(self, msg: CameraInfo, cam_id: str) -> None:
        if self.camera_data[cam_id]["depth_camera_info"] is None:
            self.camera_data[cam_id]["depth_camera_info"] = camera_info_to_dict(msg)

    def _right_info_cb(self, msg: CameraInfo, cam_id: str) -> None:
        if self.camera_data[cam_id]["right_camera_info"] is None:
            self.camera_data[cam_id]["right_camera_info"] = camera_info_to_dict(msg)

    # -- recording ---------------------------------------------------------

    @property
    def is_recording(self) -> bool:
        return self._is_recording

    def _camera_ready(self, cam_id: str) -> bool:
        if self.camera_data[cam_id]["_latest_color"] is None:
            return False
        if cam_id in self._depth_cams and self.camera_data[cam_id]["_latest_depth"] is None:
            return False
        if cam_id in self._stereo_cams and self._stereo_pair(cam_id) is None:
            return False
        return True

    def _stereo_pair(self, cam_id: str) -> tuple[tuple[int, np.ndarray], tuple[int, np.ndarray]] | None:
        """Newest (left, right) frames sharing a header stamp, i.e. from the same grab, or None.

        The two eyes arrive as separate messages, so the newest left frame's right partner may
        not be here yet. Pairing by stamp, not "latest of each", keeps a sample from mixing
        two grabs, which would corrupt stereo matching.
        """
        data = self.camera_data[cam_id]
        rights = dict(list(data["_recent_right"]))  # list() snapshots the deque against the spin thread
        for stamp, left in reversed(list(data["_recent_color"])):
            if stamp in rights:
                return (stamp, left), (stamp, rights[stamp])
        return None

    def _check_dims(self, cam_id: str, color: np.ndarray, depth: np.ndarray | None,
                    right: np.ndarray | None = None) -> None:
        """Assert a ZED camera's frames are the size --zed_resolution says they are."""
        if cam_id not in self._zed_cams:
            return
        h, w = self._zed_shape
        for kind, img in (("color", color), ("depth", depth), ("right", right)):
            if img is not None:
                assert img.shape[:2] == (h, w), (
                    f"{cam_id} {kind} image is {img.shape[1]}x{img.shape[0]}, but --zed_resolution "
                    f"{self._zed_resolution} is {w}x{h}. Check grab_resolution / pub_resolution in "
                    "~/Desktop/ros2_ws/src/zed-ros2-wrapper/zed_wrapper/config/")

    def wait_for_frames(self, timeout_s: float = 20.0) -> bool:
        """Block until every camera has delivered colour, a stereo pair (ZED) and depth (if enabled)."""
        def streams(c: str) -> str:
            on = ["left"] + (["right"] if c in self._stereo_cams else []) + (["depth"] if c in self._depth_cams else [])
            return "+".join(on)
        need = ", ".join(f"{c}:{streams(c)}" for c in self._camera_ids)
        cprint(f"⏳ Waiting for first {need} frames from: {self._camera_ids}", "cyan")
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            missing = [c for c in self._camera_ids if not self._camera_ready(c)]
            if not missing:
                try:
                    for c in self._camera_ids:
                        depth = self.camera_data[c]["_latest_depth"]
                        pair = self._stereo_pair(c) if c in self._stereo_cams else None
                        self._check_dims(c, self.camera_data[c]["_latest_color"][1],
                                         None if depth is None else depth[1],
                                         None if pair is None else pair[1][1])
                except AssertionError as exc:
                    cprint(f"❌ {exc}", "red")
                    return False
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
        self.sample_time_ns, self.robot_arm_ns, self.robot_gripper_ns = [], [], []
        for cam_id in self._camera_ids:
            for key in ("depth_images", "color_images", "right_images",
                        "depth_stamps_ns", "color_stamps_ns", "right_stamps_ns"):
                self.camera_data[cam_id][key] = []
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

        sample_ns = time.time_ns()

        # Read joints straight off the deoxys interface rather than round-tripping through
        # ROS as the ROS 1 logger did - fewer moving parts and no publisher/subscriber race.
        # q, dq and the arm clock all come from ONE state message: last_q and last_dq each
        # re-read the newest message, so one arriving in between would mix two states.
        state = self._robot.last_state
        if state is None:
            return [], False
        q = list(state.q)
        dq = list(state.dq)
        arm_ns = state.time.toMSec * 1_000_000
        gripper_ns = None
        if self._has_gripper:
            if self._robot.gripper_state_buffer_size == 0:
                return [], False
            gripper = self._robot._gripper_state_buffer[-1]
            gripper_angle = float(gripper.width) / 2.0
            gripper_ns = gripper.time.toMSec * 1_000_000
            q += [gripper_angle, gripper_angle]
            dq += [0.0, 0.0]

        # Snapshot every camera before committing anything.
        frames = {}
        for cam_id in self._camera_ids:
            right = None
            if cam_id in self._stereo_cams:
                pair = self._stereo_pair(cam_id)
                if pair is None:
                    continue
                color, right = pair
            else:
                color = self.camera_data[cam_id]["_latest_color"]
                if color is None:
                    continue
            if cam_id in self._depth_cams:
                depth = self.camera_data[cam_id]["_latest_depth"]
                if depth is None:
                    continue
            else:
                depth = None
            self._check_dims(cam_id, color[1], None if depth is None else depth[1],
                             None if right is None else right[1])
            frames[cam_id] = (color, depth, right)

        if len(frames) < len(self._camera_ids):
            return list(frames), False  # caller reports the missing cameras and stops

        self.joint_q.append(np.asarray(q, dtype=np.float64))
        self.joint_dq.append(np.asarray(dq, dtype=np.float64))
        self.sample_time_ns.append(sample_ns)
        self.robot_arm_ns.append(arm_ns)
        if gripper_ns is not None:
            self.robot_gripper_ns.append(gripper_ns)
        for cam_id, (color, depth, right) in frames.items():
            data = self.camera_data[cam_id]
            for kind, frame in (("color", color), ("depth", depth), ("right", right)):
                if frame is not None:
                    data[f"{kind}_stamps_ns"].append(frame[0])
                    data[f"{kind}_images"].append(frame[1])

        return list(frames), True

    def show_camera_feed(self) -> None:
        """Open (or refresh) one window with every camera's latest RGB frame side by side."""
        images, ids = [], []
        for cam_id in self._camera_ids:
            latest = self.camera_data[cam_id]["_latest_color"]
            if latest is None:
                continue
            images.append(latest[1])
            ids.append(cam_id)
        montage = combined_bgr(images, ids)
        if montage is None:
            return
        cv2.imshow(FEED_WINDOW, montage)
        cv2.waitKey(1)

    # -- saving ------------------------------------------------------------

    def save(self, traj_name: str) -> None:
        if len(self.joint_q) == 0:
            cprint("⚠️  0 joint states logged, nothing saved.", "red")
            return

        # e.g. traj_0__VGA.h5. Readers glob traj_*.h5 and parse the index with
        # stem.split("_")[1] or r"traj_(\d+)", both of which still work with the suffix.
        h5_path = self.output_dir / f"{traj_name}__{self._zed_resolution}.h5"
        # real2sim-toolkit's stereo layout ("Data format" in its README, checked by
        # r2st.utils.validate_noisy_depth_camera_group / validate_stereo_depth_camera_group).
        # Every dataset shares one frame index.
        #   obs/qpos, obs/qvel                                    (T,9) float64, (T,7) with --no-gripper
        #   obs/robot_arm_ns, obs/robot_gripper_ns                int64 on Franka's own clocks, `clock` attr
        #   obs/sensor_data/<cam>/timestamp_ms                    (T,) float64, sample tick (sample_time_ns / 1e6)
        #   obs/sensor_data/<cam>/left/rgb                        (T,H,W,3) uint8 rectified left RGB, gzip 1
        #   obs/sensor_data/<cam>/left/depth_noisy                (T,H,W) uint16 mm, the ZED's own depth,
        #                                                         registered to left/rgb, gzip 1
        #   obs/sensor_data/<cam>/right/rgb                       (T,H,W,3) uint8 rectified right RGB, gzip 1
        #   obs/sensor_data/<cam>/{left/rgb,left/depth_noisy,right/rgb}_timestamp_ms
        #                                                         (T,) float64, capture stamp of that frame.
        #                                                         left and right of a sample share one stamp.
        # All *timestamp_ms are host wall clock, ms since the Unix epoch. They carry no `clock`
        # attr because the toolkit expects plain float arrays there.
        with h5py.File(h5_path, "a") as f:
            q = np.array(self.joint_q)
            dq = np.array(self.joint_dq)
            f.create_dataset("obs/qpos", data=q)
            f.create_dataset("obs/qvel", data=dq)
            f.attrs["zed_resolution"] = self._zed_resolution

            # name -> (timestamps, clock, timestamps to measure age against)
            sample_ns = np.array(self.sample_time_ns, dtype=np.int64)
            streams = {"sample_time": (sample_ns, HOST_CLOCK, None),
                       "robot_arm": (np.array(self.robot_arm_ns, dtype=np.int64), ROBOT_ARM_CLOCK, None)}
            if self._has_gripper:
                streams["robot_gripper"] = (np.array(self.robot_gripper_ns, dtype=np.int64),
                                            ROBOT_GRIPPER_CLOCK, None)
            for name in ("robot_arm", "robot_gripper"):
                if name in streams:
                    stamps, clock, _ = streams[name]
                    f.create_dataset(f"obs/{name}_ns", data=stamps).attrs["clock"] = clock
            f.attrs["timezone"] = str(LOCAL_TZ)
            f.attrs["start_time_local"] = ns_to_local_str(int(sample_ns[0]))
            f.attrs["end_time_local"] = ns_to_local_str(int(sample_ns[-1]))
            f.attrs["duration_s"] = float(sample_ns[-1] - sample_ns[0]) / 1e9

            if self._has_gripper:
                gripper_q = q[:, -1]
                if gripper_q.max() - gripper_q.min() < 0.01:
                    cprint(f"ERROR: GRIPPER DOESN'T SEEM TO BE MOVING. "
                           f"RANGE: {gripper_q.max() - gripper_q.min()}", "red")
                    cprint("If the gripper was moving, restart the franka via https://172.16.0.2/desk/", "red")

            cprint("   compressing camera frames...", "white")
            for cam_id in self._camera_ids:
                data = self.camera_data[cam_id]
                if len(data["color_images"]) == 0:
                    continue
                group = f.create_group(f"obs/sensor_data/{cam_id}")
                group.create_dataset("timestamp_ms", data=sample_ns / 1e6)
                kinds = [("color", "left/rgb")]
                if cam_id in self._depth_cams and len(data["depth_images"]) > 0:
                    kinds.append(("depth", "left/depth_noisy"))
                if cam_id in self._stereo_cams and len(data["right_images"]) > 0:
                    kinds.append(("right", "right/rgb"))
                for kind, name in kinds:
                    group.create_dataset(name, data=np.array(data[f"{kind}_images"]),
                                         compression="gzip", compression_opts=1)
                    stamps = np.array(data[f"{kind}_stamps_ns"], dtype=np.int64)
                    group.create_dataset(f"{name}_timestamp_ms", data=stamps / 1e6)
                    streams[f"{cam_id}/{name}"] = (stamps, HOST_CLOCK, sample_ns)

        timing = {
            "timezone": str(LOCAL_TZ),
            "start_time_local": ns_to_local_str(self.sample_time_ns[0]),
            "end_time_local": ns_to_local_str(self.sample_time_ns[-1]),
            "duration_s": round((self.sample_time_ns[-1] - self.sample_time_ns[0]) / 1e9, 3),
            "streams": {name: {"clock": clock, **stream_stats(stamps, age_ref)}
                        for name, (stamps, clock, age_ref) in streams.items()},
        }
        with open(self.output_dir / f"{h5_path.stem}_timing.json", "w") as fh:
            json.dump(timing, fh, indent=2)
        cprint(f"\n⏱  {timing['start_time_local']} -> {timing['end_time_local']} "
               f"({timing['duration_s']:.1f} s)", "cyan")
        for name, st in timing["streams"].items():
            line = f"   {name:<24} {st['rate_hz']} Hz, max gap {st['max_gap_ms']} ms, {st['repeats']} repeats"
            if "mean_age_ms" in st:
                line += f", age {st['mean_age_ms']}/{st['max_age_ms']} ms (mean/max)"
            cprint(line, "white")

        # One <cam>_camera_info.json per camera, keyed like the HDF5 streams.
        for cam_id in self._camera_ids:
            kinds = [("left", "color_camera_info")]
            if cam_id in self._depth_cams:
                kinds.append(("depth_noisy", "depth_camera_info"))
            if cam_id in self._stereo_cams:
                # right P[3] = -fx * baseline (ROS stereo convention), so the baseline is in here
                kinds.append(("right", "right_camera_info"))
            infos = {kind: self.camera_data[cam_id][key] for kind, key in kinds
                     if self.camera_data[cam_id][key] is not None}
            with open(self.output_dir / f"{cam_id}_camera_info.json", "w") as fh:
                json.dump(infos, fh, indent=2)

        # Sample montages, every MONTAGE_EVERY-th timestep, for eyeballing the demo afterwards.
        # Filenames keep the sample index, so a PNG still maps to its row in the HDF5.
        img_dir = self.output_dir / "images"
        img_dir.mkdir(exist_ok=True)
        cprint(f"\n📸 Saving sample images to {img_dir}...", "cyan")

        available = {c: self.camera_data[c]["color_images"] for c in self._camera_ids
                     if len(self.camera_data[c]["color_images"]) > 0}
        for cam_id in self._camera_ids:
            n_c = len(self.camera_data[cam_id]["color_images"])
            counts = [f"{n_c} left"]
            if cam_id in self._stereo_cams:
                counts.append(f"{len(self.camera_data[cam_id]['right_images'])} right")
            if cam_id in self._depth_cams:
                counts.append(f"{len(self.camera_data[cam_id]['depth_images'])} depth")
            else:
                counts.append("depth off")
            cprint(f"   Camera {cam_id}: {', '.join(counts)} images collected", "white")

        if available:
            num_samples = min(len(v) for v in available.values())
            ids = [c for c in self._camera_ids if c in available]
            saved = range(0, num_samples, MONTAGE_EVERY)
            for i in saved:
                save_combined_rgbs([available[c][i] for c in ids], ids,
                                   img_dir / f"combined_rgb__{traj_name}__{i:03d}.png")
            cprint(f"   Saved {len(saved)} combined RGB images as PNG (every {MONTAGE_EVERY}th of {num_samples})",
                   "white")

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


def hold_position(robot: FrankaInterface) -> None:
    """Zero-delta termination. The NUC otherwise keeps interpolating the last teleop goal."""
    gripper = float(robot.last_gripper_action)
    if gripper == 0.0:
        gripper = -1.0  # untouched; 0 would command a close
    try:
        robot.control(
            controller_type=CONTROLLER_TYPE,
            action=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, gripper],
            controller_cfg=get_default_controller_config(controller_type=CONTROLLER_TYPE),
            termination=True,
        )
    except Exception as exc:  # noqa: BLE001 - exiting must not raise
        cprint(f"Could not hold the arm on exit: {exc}", "yellow")


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


def call_while_refreshing(fn: Callable[[], None], refresh: Callable[[], None] | None) -> None:
    """Run `fn` here, or on a worker while `refresh` keeps the camera window alive.

    save() gzip-compresses every frame on the caller. Doing that on the OpenCV thread
    stops the Qt window from processing events, the desktop treats it as not responding,
    and Force Quit delivers SIGKILL — which leaves a half-written HDF5. The worker does
    the write; this thread only calls imshow / waitKey.
    """
    if refresh is None:
        fn()
        return
    errors: list[Exception] = []

    def _run() -> None:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 - re-raised on the GUI thread
            errors.append(exc)

    worker = threading.Thread(target=_run, name="save-demo")
    worker.start()
    try:
        while worker.is_alive():
            refresh()
            worker.join(0.03)
    finally:
        worker.join()
    if errors:
        raise errors[0]


def read_line(prompt: str, refresh: Callable[[], None] | None = None, poll_s: float = 0.03) -> str:
    """Read one stdin line. With `refresh`, keep calling it until Enter so a window stays live.

    Canonical-mode stdin only becomes readable once Enter is pressed, so this still waits for
    a full line; it just does not block inside input(), which would freeze cv2.imshow.
    """
    if refresh is None:
        return input(prompt)
    print(prompt, end="", flush=True)
    chars: list[str] = []
    while True:
        refresh()
        if select.select([sys.stdin], [], [], poll_s)[0] != [sys.stdin]:
            continue
        ch = sys.stdin.read(1)
        if ch in ("\n", ""):
            return "".join(chars)
        chars.append(ch)


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
    parser.add_argument("--zed_resolution", type=str, choices=sorted(ZED_RESOLUTIONS),
                        help="grab_resolution the ZED cameras were launched with (required). Every ZED "
                             "frame is checked against it, and it is saved as the HDF5 attr "
                             "'zed_resolution'.")
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
                             "The wrist ZED is an RGB policy input only, and its depth is "
                             "most of the file size; the D435 depth the aux labels need is kept.")
    parser.add_argument("--no-depth", action="store_true",
                        help="do not subscribe to or record depth (colour + joints only)")
    parser.add_argument("--qos-best-effort", action="store_true",
                        help="use BEST_EFFORT QoS if topics exist but no frames arrive")
    parser.add_argument("--list-topics", action="store_true", help="list live topics and exit")
    parser.add_argument("--show-camera-feed", action="store_true",
                        help="open one window with every camera's latest RGB frame joined side by side")
    args = parser.parse_args()

    if args.list_topics:
        return list_topics()
    if not args.output_dir:
        parser.error("--output_dir is required (or use --list-topics)")
    if not args.zed_resolution:
        parser.error(f"--zed_resolution is required, one of {sorted(ZED_RESOLUTIONS)}")

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
    h, w = ZED_RESOLUTIONS[args.zed_resolution]
    cprint(f"ZED res  : {args.zed_resolution} ({w}x{h})", "white")
    if not record_depth:
        cprint("Depth    : DISABLED for every camera (--no-depth)", "white")
    else:
        per_cam = ", ".join(f"{c}:{'off' if c in no_depth_cameras else 'on'}" for c in camera_ids)
        cprint(f"Depth    : {per_cam}", "white")
    depthless = camera_ids if not record_depth else [c for c in camera_ids if c in no_depth_cameras]
    if depthless:
        cprint(f"⚠️  No depth for {depthless}: these obs/sensor_data groups will fail "
               "r2st.utils.validate_noisy_depth_camera_group.", "yellow")
    monocular = [c for c in camera_ids if LAYOUTS[camera_layouts[c]].right is None]
    if monocular:
        cprint(f"⚠️  No right eye for {monocular} (layout has none): these obs/sensor_data groups will "
               "fail r2st.utils.validate_stereo_depth_camera_group.", "yellow")
    cprint(f"Gripper  : {'enabled' if has_gripper else 'DISABLED (--no-gripper)'}", "white")
    if args.show_camera_feed:
        cprint(f"Feed     : live window '{FEED_WINDOW}'", "white")
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
                                 args.qos_best_effort, args.zed_resolution, record_depth=record_depth,
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
    refresh = collector.show_camera_feed if args.show_camera_feed else None

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

            read_line("\n▶️  Press ENTER to START recording demo...", refresh)
            collector.start_recording()
            print("\n⏸️  Press ENTER to STOP recording demo...")

            while True:
                if refresh is not None:
                    refresh()
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
            call_while_refreshing(lambda: collector.save(traj_name=f"traj_{demo_index}"), refresh)
            cprint(f"✅ Demo #{demo_index} complete!", "green")

            if exit_code:
                break
            if read_line("\n❓ Collect another demo? (y/n): ", refresh).strip().lower() != "y":
                break
            demo_index += 1
    except KeyboardInterrupt:
        cprint("\n⏹️  Interrupted", "yellow")
    finally:
        cprint(f"\n🏁 Collection complete! Total demos: {demo_index + 1}", "green", attrs=["bold"])
        # Stop teleop before the hold, and wait out any control() already in flight,
        # so a leftover SpaceMouse command cannot land after the termination.
        should_continue_flag = False
        if args.run_teleop:
            time.sleep(0.2)
            hold_position(robot_interface)
        running = False
        # The demo is already on disk. cv2.destroyAllWindows (Qt) and rclpy.shutdown
        # block after this line, so the shell would not return and the window would stay up.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(exit_code)


if __name__ == "__main__":
    sys.exit(main())
