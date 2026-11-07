#!/usr/bin/env python3
"""SpaceMouse teleop demos recorded for robosimstudio real2sim: telemetry + front and wrist video.

robosimstudio's live real2sim page (robosimstudio/examples/live) takes reference CLIPS and stills
(`./ctl new NAME --ref FILE`, `./ctl ref frames CLIP --at ...`), and a twin's routine drives the Panda with
`sim.step(action, mode="joint_position")`: 7 joint targets and a gripper value, 0 open .. 1 shut. So every
episode is written twice over:

    <output-dir>/<task>__YYYYmmdd_HHMMSS/
        episode_000.h5                 everything (layout below)
        episode_000_front.mp4          the clips, for `ctl new --ref` / `ctl ref frames`
        episode_000_wrist.mp4
        episode_000_front_first.png    first frames, for `ctl compare --ref`
        episode_000_wrist_first.png

HDF5 layout (T samples at --rate-hz, every dataset's first axis is T):
    attrs                 task, robot, rate_hz, start_time_iso, controller_type, reset_joint_positions,
                          interface_cfg (yaml text), gripper_convention, format_version, n_samples
    time/t                s since the episode started (monotonic)
    time/wall             unix seconds
    robot/joint_pos       (T,7) rad        robot/joint_vel (T,7)   robot/joint_pos_desired (T,7)
    robot/gripper_width   (T,)  m          Franka Hand: 0.08 open, 0 shut
    robot/ee_pose         (T,4,4)          O_T_EE in the robot base frame
    robot/ee_pos (T,3) m, robot/ee_quat (T,4) xyzw
    action/osc_delta      (T,6)            the SpaceMouse OSC_POSE action last sent (before deoxys' scaling)
    action/gripper        (T,)             -1 open / +1 close, as sent
    action/t_cmd          (T,)             when it was sent, on time/t's clock (NaN before the first)
    sim/qpos              (T,8)            7 joints + gripper fraction (0 open .. 1 shut): robosimstudio's
                                           joint_position action for one Panda
    cameras/<cam>/rgb     (T,H,W,3) uint8  latest frame at each sample
    cameras/<cam>/t_frame (T,)             when that frame arrived, on time/t's clock
    cameras/<cam>/frame_new (T,) bool      the frame changed since the previous sample
    cameras/wrist/depth   (T,H,W) uint16 mm, registered to the wrist rgb (unless --no-wrist-depth)
    cameras/<cam> attrs   model, source, width, height; wrist: camera_info (JSON: k, d, p of the rgb image)

Cameras: front = the Logitech C310 on USB, read here over V4L2; wrist = the ZED-M on the arm, read from the
ROS 2 topics docker/zed_cameras.sh publishes. Start only that camera:
    ZED_CAMERAS="zedm:zedm:15137730" docker/zed_cameras.sh
The NUC must be running auto_arm.sh and auto_gripper.sh.

Usage (from zeyu_panda_utils):
    docker/run_local.sh python3 scripts/record_robosim_demo.py --task fold_towel
    # check the plumbing without moving the arm: no teleop, no homing, 5 s
    docker/run_local.sh python3 scripts/record_robosim_demo.py --task test --no-teleop --no-home --duration 5

ENTER starts an episode and ENTER stops it; Ctrl-C stops and keeps it, then exits. The SpaceMouse's pitch is
ignored by default, so tilting the puck forward/back does not rotate the gripper; roll and yaw still work.
`--lock-rotation` picks other rotations to ignore (`--lock-rotation roll yaw`), or none (the flag alone).
"""

from __future__ import annotations

import argparse
import json
import queue
import select
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path

import cv2
import h5py
import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from sensor_msgs.msg import CameraInfo, Image
from termcolor import cprint

from deoxys.experimental.motion_utils import reset_joints_to
from deoxys.franka_interface import FrankaInterface
from deoxys.utils.config_utils import get_default_controller_config
from deoxys.utils.input_utils import input2action
from deoxys.utils.io_devices import SpaceMouse

from log_demonstration2 import (
    CONTROLLER_TYPE,
    RESET_JOINT_POSITIONS,
    SPACE_MOUSE_PRODUCT_ID,
    SPACE_MOUSE_VENDOR_ID,
    camera_info_to_dict,
    decode_color,
    decode_depth,
    make_qos,
)

FORMAT_VERSION = 1
FRANKA_HAND_MAX_WIDTH_M = 0.08
TELEOP_HZ = 50.0
DEFAULT_FRONT_DEVICE = "/dev/v4l/by-id/usb-046d_081b_C1531760-video-index0"  # the Logitech C310
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "robosim"  # <workspace>/data/robosim
ZED_TOPICS = {
    "rgb": "/zed_multi/{cam}/rgb/image_rect_color",
    "rgb_info": "/zed_multi/{cam}/rgb/camera_info",
    "depth": "/zed_multi/{cam}/depth/depth_registered",  # 32FC1 metres, already on the rgb grid
}


class RecordingError(RuntimeError):
    """A sensor stopped delivering mid-episode."""


def mat2quat_xyzw(r: np.ndarray) -> np.ndarray:
    """Rotation matrix -> unit quaternion (x, y, z, w), w >= 0. deoxys' transform_utils.mat2quat does the same
    but is numba-compiled through eigh, which needs scipy, and the panda-deoxys image has none."""
    m00, m11, m22 = r[0, 0], r[1, 1], r[2, 2]
    trace = m00 + m11 + m22
    if trace > 0.0:
        s = 2.0 * np.sqrt(trace + 1.0)
        q = [(r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s, 0.25 * s]
    elif m00 > m11 and m00 > m22:
        s = 2.0 * np.sqrt(1.0 + m00 - m11 - m22)
        q = [0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s, (r[2, 1] - r[1, 2]) / s]
    elif m11 > m22:
        s = 2.0 * np.sqrt(1.0 + m11 - m00 - m22)
        q = [(r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s, (r[0, 2] - r[2, 0]) / s]
    else:
        s = 2.0 * np.sqrt(1.0 + m22 - m00 - m11)
        q = [(r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s, (r[1, 0] - r[0, 1]) / s]
    q = np.asarray(q, dtype=np.float64)
    q /= np.linalg.norm(q)
    return -q if q[3] < 0.0 else q


# -- cameras ---------------------------------------------------------------


class V4L2Camera:
    """A USB (UVC) webcam read on a background thread; reopens the device if it disappears.

    From zeyu-Policy-deploy-Franka/backend/cameras.py, plus a frame counter so samples can say whether a frame
    is new, and the two frame-rate fixes below (buffer count, exposure_dynamic_framerate). YUYV, not MJPG: the C310's MJPG frames make libjpeg print "Corrupt JPEG data" for every frame, and
    640x480 YUYV at 30 fps fits USB 2.
    """

    # VIDIOC_G/S_CTRL = _IOWR('V', 27/28, struct v4l2_control {u32 id; s32 value}), V4L2_CID_EXPOSURE_AUTO_PRIORITY.
    _VIDIOC_G_CTRL = 0xC008561B
    _VIDIOC_S_CTRL = 0xC008561C
    _CID_EXPOSURE_AUTO_PRIORITY = 0x009A0903

    def __init__(self, device: str, width: int, height: int, fps: int = 30, fourcc: str = "YUYV",
                 fixed_fps: bool = True):
        self.device, self.width, self.height, self.fps, self.fourcc = device, width, height, fps, fourcc
        self.fixed_fps = fixed_fps
        self._original_priority: int | None = None  # put back on close, so other users of the webcam are unaffected
        self.error: str | None = None
        self._lock = threading.Lock()
        self._frame: tuple[np.ndarray, float, int] | None = None
        self._seq = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"v4l2-{device}", daemon=True)
        self._thread.start()

    def _open(self):
        cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        if not cap.isOpened():
            raise OSError(f"cannot open {self.device}")
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        # 2, not 1: with a single buffer the driver has nowhere to put a frame while we hold one and drops every
        # other frame (measured on the C310: 15 fps with 1, 29 with 2 or 4). The reader drains continuously, so the
        # newest frame is still at most one frame old.
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        if self.fixed_fps:
            self._set_exposure_auto_priority(False)
        return cap

    def _set_exposure_auto_priority(self, on: bool, remember: bool = True) -> None:
        """With exposure_dynamic_framerate on (the C310 ships with it on), auto exposure lengthens the exposure past
        one frame in a dim room and the camera drops to 7-15 fps, so most samples repeat a frame. Off caps the
        exposure at 1/fps and keeps the rate. OpenCV has no property for it, hence the ioctl; a control set on a
        second descriptor applies to the open stream."""
        import fcntl
        import os
        import struct

        fd = os.open(self.device, os.O_RDWR)
        try:
            if remember and self._original_priority is None:
                got = fcntl.ioctl(fd, self._VIDIOC_G_CTRL, struct.pack("Ii", self._CID_EXPOSURE_AUTO_PRIORITY, 0))
                self._original_priority = struct.unpack("Ii", got)[1]
            fcntl.ioctl(fd, self._VIDIOC_S_CTRL, struct.pack("Ii", self._CID_EXPOSURE_AUTO_PRIORITY, int(on)))
        except OSError as exc:
            cprint(f"⚠️  {self.device}: could not set exposure_dynamic_framerate={int(on)} ({exc}); "
                   "the frame rate may drop in dim light", "yellow")
        finally:
            os.close(fd)

    def _run(self) -> None:
        cap = None
        while not self._stop.is_set():
            try:
                if cap is None:
                    cap = self._open()
                    self.error = None
                ok, bgr = cap.read()
                if not ok:
                    raise OSError(f"no frame from {self.device}")
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                with self._lock:
                    self._seq += 1
                    self._frame = (rgb, time.monotonic(), self._seq)
            except Exception as exc:  # noqa: BLE001 - unplugged, busy, ...: report and retry
                self.error = str(exc)
                if cap is not None:
                    cap.release()
                    cap = None
                self._stop.wait(2.0)
        if cap is not None:
            cap.release()

    def latest(self) -> tuple[np.ndarray, float, int] | None:
        """(rgb, monotonic arrival time, frame counter), or None before the first frame."""
        with self._lock:
            return self._frame

    def attrs(self) -> dict:
        return {"model": "Logitech C310", "source": self.device, "fourcc": self.fourcc, "fps": self.fps,
                "exposure_dynamic_framerate": int(not self.fixed_fps)}

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)
        if self._original_priority is not None:
            try:
                self._set_exposure_auto_priority(bool(self._original_priority), remember=False)
            except OSError:  # unplugged meanwhile: nothing to restore
                pass


class ZedRosCamera:
    """Latest rgb (and registered depth) of one ZED from the zed_cameras.sh topics, spun on its own thread."""

    def __init__(self, cam: str, record_depth: bool):
        self.cam, self.record_depth = cam, record_depth
        self.error: str | None = None
        self.camera_info: dict | None = None
        self._lock = threading.Lock()
        self._rgb: tuple[np.ndarray, float, int] | None = None
        self._depth: tuple[np.ndarray, float] | None = None
        self._seq = 0
        # BEST_EFFORT receives from both BEST_EFFORT and RELIABLE publishers; RELIABLE only from RELIABLE.
        qos = make_qos(best_effort=True)
        self._node = rclpy.create_node(f"record_robosim_demo_{cam}")
        self._subs = [
            self._node.create_subscription(Image, ZED_TOPICS["rgb"].format(cam=cam), self._on_rgb, qos),
            self._node.create_subscription(CameraInfo, ZED_TOPICS["rgb_info"].format(cam=cam), self._on_info, qos),
        ]
        if record_depth:
            self._subs.append(
                self._node.create_subscription(Image, ZED_TOPICS["depth"].format(cam=cam), self._on_depth, qos))
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._thread = threading.Thread(target=self._executor.spin, name=f"ros-{cam}", daemon=True)
        self._thread.start()

    def _on_rgb(self, msg: Image) -> None:
        try:
            rgb = decode_color(msg)
        except ValueError as exc:
            self.error = str(exc)
            return
        with self._lock:
            self._seq += 1
            self._rgb = (rgb, time.monotonic(), self._seq)

    def _on_depth(self, msg: Image) -> None:
        try:
            depth = decode_depth(msg)
        except ValueError as exc:
            self.error = str(exc)
            return
        with self._lock:
            self._depth = (depth, time.monotonic())

    def _on_info(self, msg: CameraInfo) -> None:
        if self.camera_info is None:
            self.camera_info = camera_info_to_dict(msg)

    def latest(self) -> tuple[np.ndarray, float, int] | None:
        with self._lock:
            return self._rgb

    def latest_depth(self) -> tuple[np.ndarray, float] | None:
        with self._lock:
            return self._depth

    def ready(self) -> bool:
        return (self.latest() is not None and self.camera_info is not None
                and (not self.record_depth or self.latest_depth() is not None))

    def attrs(self) -> dict:
        return {"model": "ZED-M" if self.cam == "zedm" else self.cam,
                "source": ZED_TOPICS["rgb"].format(cam=self.cam),
                "depth_source": ZED_TOPICS["depth"].format(cam=self.cam) if self.record_depth else "",
                "camera_info": json.dumps(self.camera_info)}

    def close(self) -> None:
        self._executor.shutdown()
        self._thread.join(timeout=2)
        self._node.destroy_node()


# -- teleop ----------------------------------------------------------------


class Teleop:
    """SpaceMouse -> OSC_POSE on a thread, as log_demonstration2.teleop_control_thread_target, but remembering
    the last action sent so it can be recorded. `control_lock` is held around every robot.control call, so homing
    (which takes it for its whole move) never interleaves commands with teleop.

    lock_rotation names SpaceMouse rotations to ignore, which makes the arm easier to steer. input2action builds
    action[3:6] (OSC's ax, ay, az: rotation about the base x, y, z axes) from the puck's (pitch, roll, -yaw), so each
    name zeroes one action index (LOCKABLE_ROTATIONS)."""

    LOCKABLE_ROTATIONS = {"pitch": 3, "roll": 4, "yaw": 5}

    def __init__(self, robot: FrankaInterface, device: SpaceMouse | None, t0: float,
                 lock_rotation: tuple[str, ...] = ("pitch",)):
        self._robot, self._device, self._t0 = robot, device, t0
        self._locked_indices = [self.LOCKABLE_ROTATIONS[name] for name in lock_rotation]
        self.control_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._last: tuple[np.ndarray, float, float] = (np.zeros(6), float("nan"), float("nan"))
        self._stop = threading.Event()
        self._thread = None
        if device is not None:
            self._thread = threading.Thread(target=self._run, name="teleop", daemon=True)
            self._thread.start()

    def _run(self) -> None:
        controller_cfg = get_default_controller_config(controller_type=CONTROLLER_TYPE)
        n_errors = 0
        while not self._stop.wait(1.0 / TELEOP_HZ):
            try:
                action, _grasp = input2action(device=self._device, controller_type=CONTROLLER_TYPE)
                if action is None:  # the RIGHT button: deoxys' reset flag, not a motion command
                    continue
                action[self._locked_indices] = 0.0  # recorded as sent, so action/osc_delta shows the zeros too
                with self.control_lock:
                    self._robot.control(controller_type=CONTROLLER_TYPE, action=action,
                                        controller_cfg=controller_cfg)
                with self._state_lock:
                    self._last = (np.asarray(action[:6], dtype=np.float64), float(action[6]),
                                  time.monotonic() - self._t0)
            except Exception as exc:  # noqa: BLE001 - one bad tick must not end teleop
                n_errors += 1
                if n_errors <= 3 or n_errors % 100 == 0:
                    cprint(f"\nTeleop error ({n_errors}): {exc}", "red")
                    if n_errors == 1:
                        traceback.print_exc()
                if n_errors >= 500:
                    cprint("Teleop: 500 consecutive errors, stopping teleop.", "red")
                    return
            else:
                n_errors = 0

    def last(self) -> tuple[np.ndarray, float, float]:
        """(osc_delta (6,), gripper -1/+1 or NaN, t_cmd or NaN)."""
        with self._state_lock:
            return self._last

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)


# -- writing ---------------------------------------------------------------


class EpisodeWriter:
    """Appends samples to one episode's HDF5 and MP4s on a background thread, so compression and disk never hold
    up the sampling clock. Datasets are made on the first sample (its shapes define them) and grown in blocks."""

    BLOCK = 256

    def __init__(self, stem: Path, rate_hz: float, file_attrs: dict, camera_attrs: dict[str, dict]):
        self.stem = stem
        self.paths = [stem.with_suffix(".h5")]
        self._rate_hz = rate_hz
        self._camera_attrs = camera_attrs
        self._h5 = h5py.File(self.paths[0], "w")
        for key, value in file_attrs.items():
            self._h5.attrs[key] = value
        self._datasets: dict[str, h5py.Dataset] = {}
        self._videos: dict[str, cv2.VideoWriter] = {}
        self.n = 0
        self.error: BaseException | None = None
        self._queue: queue.Queue = queue.Queue()
        self._thread = threading.Thread(target=self._run, name="episode-writer", daemon=True)
        self._thread.start()

    def put(self, sample: dict[str, np.ndarray]) -> None:
        if self.error is not None:
            raise RecordingError(f"writing {self.paths[0].name} failed: {self.error}")
        self._queue.put(sample)

    def backlog(self) -> int:
        return self._queue.qsize()

    def _run(self) -> None:
        while True:
            sample = self._queue.get()
            if sample is None:
                return
            if self.error is not None:
                continue  # drain; the main thread is told on its next put
            try:
                self._write(sample)
            except BaseException as exc:  # noqa: BLE001 - reported to the main thread
                self.error = exc

    def _write(self, sample: dict[str, np.ndarray]) -> None:
        if self.n == 0:
            self._create(sample)
        for key, value in sample.items():
            ds = self._datasets[key]
            if self.n >= ds.shape[0]:
                ds.resize(ds.shape[0] + self.BLOCK, axis=0)
            ds[self.n] = value
        for cam, video in self._videos.items():
            video.write(cv2.cvtColor(sample[f"cameras/{cam}/rgb"], cv2.COLOR_RGB2BGR))
        self.n += 1

    def _create(self, sample: dict[str, np.ndarray]) -> None:
        for key, value in sample.items():
            value = np.asarray(value)
            image = value.ndim >= 2 and value.dtype in (np.uint8, np.uint16)
            self._datasets[key] = self._h5.create_dataset(
                key, shape=(self.BLOCK,) + value.shape, maxshape=(None,) + value.shape, dtype=value.dtype,
                chunks=(1,) + value.shape if image else (self.BLOCK,) + value.shape,
                compression="lzf" if image else None)
        for cam, attrs in self._camera_attrs.items():
            group = self._h5[f"cameras/{cam}"]
            rgb = sample[f"cameras/{cam}/rgb"]
            for key, value in {**attrs, "height": rgb.shape[0], "width": rgb.shape[1]}.items():
                group.attrs[key] = value
            mp4 = self.stem.parent / f"{self.stem.name}_{cam}.mp4"
            png = self.stem.parent / f"{self.stem.name}_{cam}_first.png"
            # mp4v: this image's OpenCV has no H.264 encoder; robosimstudio reads clips with ffmpeg, which decodes it.
            video = cv2.VideoWriter(str(mp4), cv2.VideoWriter_fourcc(*"mp4v"), self._rate_hz,
                                    (rgb.shape[1], rgb.shape[0]))
            if not video.isOpened():
                raise OSError(f"cannot open a video writer for {mp4}")
            self._videos[cam] = video
            cv2.imwrite(str(png), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            self.paths += [mp4, png]

    def close(self) -> int:
        """Finish writing and close the files; returns the number of samples written."""
        self._queue.put(None)
        self._thread.join()
        for video in self._videos.values():
            video.release()
        for ds in self._datasets.values():
            ds.resize(self.n, axis=0)
        self._h5.attrs["n_samples"] = self.n
        self._h5.close()
        if self.error is not None:
            cprint(f"❌ Writing failed after {self.n} samples: {self.error}", "red")
        return self.n

    def delete(self) -> None:
        for path in self.paths:
            path.unlink(missing_ok=True)


# -- sampling --------------------------------------------------------------


class Sampler:
    """Builds one sample: robot state + last teleop action + every camera's latest frame."""

    def __init__(self, robot: FrankaInterface, teleop: Teleop, cameras: dict, t0: float, max_frame_age: float):
        self._robot, self._teleop, self._cameras = robot, teleop, cameras
        self._t0, self._max_age = t0, max_frame_age
        self._last_seq: dict[str, int] = {}
        self._last_state = None
        self._state_changed_at = time.monotonic()

    def reset(self, t0: float) -> None:
        self._t0 = t0
        self._last_seq.clear()

    def sample(self) -> dict[str, np.ndarray]:
        now = time.monotonic()
        state = self._robot.last_state
        width = self._robot.last_gripper_q
        if state is None or width is None:
            raise RecordingError("no robot state (is deoxys running on the NUC?)")
        if state is not self._last_state:
            self._last_state, self._state_changed_at = state, now
        elif now - self._state_changed_at > self._max_age:
            raise RecordingError(f"robot state is {now - self._state_changed_at:.1f} s old")

        q = np.array(state.q, dtype=np.float64)
        ee_pose = np.array(state.O_T_EE, dtype=np.float64).reshape(4, 4).T
        width = float(width)
        osc_delta, gripper_cmd, t_cmd = self._teleop.last()
        out = {
            "time/t": np.float64(now - self._t0),
            "time/wall": np.float64(time.time()),
            "robot/joint_pos": q,
            "robot/joint_vel": np.array(state.dq, dtype=np.float64),
            "robot/joint_pos_desired": np.array(state.q_d, dtype=np.float64),
            "robot/gripper_width": np.float64(width),
            "robot/ee_pose": ee_pose,
            "robot/ee_pos": ee_pose[:3, 3].copy(),
            "robot/ee_quat": mat2quat_xyzw(ee_pose[:3, :3]),
            "action/osc_delta": osc_delta,
            "action/gripper": np.float64(gripper_cmd),
            "action/t_cmd": np.float64(t_cmd),
            "sim/qpos": np.append(q, np.clip(1.0 - width / FRANKA_HAND_MAX_WIDTH_M, 0.0, 1.0)),
        }
        for name, cam in self._cameras.items():
            latest = cam.latest()
            if latest is None:
                raise RecordingError(f"no frame from the {name} camera" + (f" ({cam.error})" if cam.error else ""))
            rgb, t_frame, seq = latest
            if now - t_frame > self._max_age:
                raise RecordingError(f"the {name} camera's last frame is {now - t_frame:.1f} s old"
                                     + (f" ({cam.error})" if cam.error else ""))
            out[f"cameras/{name}/rgb"] = rgb
            out[f"cameras/{name}/t_frame"] = np.float64(t_frame - self._t0)
            out[f"cameras/{name}/frame_new"] = np.bool_(seq != self._last_seq.get(name))
            self._last_seq[name] = seq
            if getattr(cam, "record_depth", False):
                depth = cam.latest_depth()
                if depth is None or now - depth[1] > self._max_age:
                    raise RecordingError(f"no recent depth from the {name} camera")
                out[f"cameras/{name}/depth"] = depth[0]
        return out


def enter_pressed() -> bool:
    """True if a line was typed (non-blocking). Without a terminal (stdin is /dev/null, always 'readable' at EOF)
    there is nothing to press, so only --duration or Ctrl-C ends an episode."""
    if sys.stdin.isatty() and select.select([sys.stdin], [], [], 0.0)[0]:
        sys.stdin.readline()
        return True
    return False


def record_episode(stem: Path, sampler: Sampler, cameras: dict, rate_hz: float, file_attrs: dict,
                   duration: float | None) -> tuple[EpisodeWriter, str]:
    """Sample at rate_hz until ENTER, --duration or Ctrl-C. Returns the closed writer and why it stopped."""
    writer = EpisodeWriter(stem, rate_hz, file_attrs, {name: cam.attrs() for name, cam in cameras.items()})
    period = 1.0 / rate_hz
    t0 = time.monotonic()
    sampler.reset(t0)
    next_t, late, last_status, n_at_status = t0, 0, t0, 0
    reason = "enter"
    cprint(f"🎬 Recording {stem.name} - press ENTER to stop", "green")
    try:
        while True:
            if enter_pressed():
                break
            now = time.monotonic()
            if duration is not None and now - t0 >= duration:
                reason = "duration"
                break
            if now < next_t:
                time.sleep(next_t - now)
            next_t += period
            if time.monotonic() - next_t > period:  # fell more than a sample behind: skip ahead, don't burst
                late += 1
                next_t = time.monotonic() + period
            sample = sampler.sample()
            writer.put(sample)

            now = time.monotonic()
            if now - last_status >= 1.0:
                hz = (writer.n + writer.backlog() - n_at_status) / (now - last_status)
                ages = "  ".join(f"{name} {1000 * (now - t0 - sample[f'cameras/{name}/t_frame']):3.0f} ms"
                                 for name in cameras)
                print(f"\r   {sample['time/t']:6.1f} s  {writer.n + writer.backlog():5d} samples  {hz:4.1f} Hz  "
                      f"| {ages} | gripper {1000 * sample['robot/gripper_width']:4.1f} mm "
                      f"| write queue {writer.backlog():3d}   ", end="", flush=True)
                last_status, n_at_status = now, writer.n + writer.backlog()
    except RecordingError as exc:
        print()
        cprint(f"❌ {exc}. Stopping this episode.", "red")
        reason = "error"
    except KeyboardInterrupt:
        reason = "interrupt"
    except BaseException:
        writer.close()  # leave readable files behind whatever went wrong
        raise
    print()
    n = writer.close()
    if late:
        cprint(f"⚠️  The sampling loop fell behind {late} times (the clip still has one frame per sample).", "yellow")
    cprint(f"⏹️  Stopped ({reason}): {n} samples, {n / rate_hz:.1f} s", "yellow")
    return writer, reason


def wait_for_sensors(robot: FrankaInterface, cameras: dict, timeout_s: float = 20.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        robot_ok = robot.last_state is not None and robot.last_gripper_q is not None
        cams_missing = [name for name, cam in cameras.items()
                        if not (cam.ready() if hasattr(cam, "ready") else cam.latest() is not None)]
        if robot_ok and not cams_missing:
            return True
        time.sleep(0.2)
    if not robot_ok:
        cprint("❌ No arm/gripper state from deoxys. Are auto_arm.sh and auto_gripper.sh running on the NUC?", "red")
    for name in cams_missing:
        cam = cameras[name]
        if isinstance(cam, ZedRosCamera):
            cprint(f"❌ {name}: nothing on {ZED_TOPICS['rgb'].format(cam=cam.cam)}"
                   + (" (or its camera_info / depth)" if cam.record_depth else " (or its camera_info)")
                   + '. Start it: ZED_CAMERAS="zedm:zedm:15137730" docker/zed_cameras.sh', "red")
        else:
            cprint(f"❌ {name}: no frames from {cam.device} ({cam.error or 'no error yet'}). "
                   "Is the webcam plugged in and passed into the container (docker/run_local.sh)?", "red")
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", required=True, help="name of the task/scene, used for the session folder")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--rate-hz", type=float, default=30.0, help="samples (and clip frames) per second")
    parser.add_argument("--front-device", default=DEFAULT_FRONT_DEVICE, help="V4L2 device of the front webcam, "
                        "or 'none'")
    parser.add_argument("--front-size", default="640x480", help="WxH; 640x480 is the C310's largest YUYV size")
    parser.add_argument("--front-allow-low-fps", action="store_true",
                        help="leave the webcam's exposure_dynamic_framerate on: brighter in a dim room, but it "
                             "drops to 7-15 fps and most samples repeat a frame")
    parser.add_argument("--wrist-camera", default="zedm", help="ZED camera name under /zed_multi, or 'none'")
    parser.add_argument("--no-wrist-depth", action="store_true", help="do not record the wrist ZED's depth")
    parser.add_argument("--max-frame-age", type=float, default=0.5,
                        help="stop an episode if a camera or the robot is silent this long (s)")
    parser.add_argument("--no-teleop", action="store_true", help="record without commanding the arm")
    parser.add_argument("--lock-rotation", nargs="*", choices=sorted(Teleop.LOCKABLE_ROTATIONS), default=["pitch"],
                        help="SpaceMouse rotations to ignore (default: pitch); give the flag alone to lock none, "
                             "e.g. `--lock-rotation pitch yaw` or `--lock-rotation`")
    parser.add_argument("--no-home", action="store_true", help="do not move to the reset pose before each episode")
    parser.add_argument("--duration", type=float, default=None, help="stop each episode after this many seconds")
    parser.add_argument("--interface-cfg", type=Path,
                        default=Path(__file__).resolve().parent.parent / "configs" / "charmander.yml")
    parser.add_argument("--vendor-id", type=int, default=SPACE_MOUSE_VENDOR_ID)
    parser.add_argument("--product-id", type=int, default=SPACE_MOUSE_PRODUCT_ID)
    args = parser.parse_args()

    front_w, front_h = (int(v) for v in args.front_size.lower().split("x"))
    home = not args.no_home
    if home and args.no_teleop:
        cprint("Note: --no-teleop still homes the arm before each episode; add --no-home to keep it still.", "yellow")

    device = None
    if not args.no_teleop:
        try:
            device = SpaceMouse(vendor_id=args.vendor_id, product_id=args.product_id)
            device.start_control()
        except OSError as exc:
            cprint(f"❌ Cannot open the SpaceMouse: {exc} (see README_SETUP.md)", "red")
            return 1

    rclpy.init(args=None)
    cameras: dict = {}
    if args.front_device.lower() != "none":
        cameras["front"] = V4L2Camera(args.front_device, front_w, front_h, fixed_fps=not args.front_allow_low_fps)
    if args.wrist_camera.lower() != "none":
        cameras["wrist"] = ZedRosCamera(args.wrist_camera, record_depth=not args.no_wrist_depth)

    robot = FrankaInterface(str(args.interface_cfg), use_visualizer=False)
    t_origin = time.monotonic()
    teleop = Teleop(robot, device, t_origin, lock_rotation=tuple(args.lock_rotation))
    sampler = Sampler(robot, teleop, cameras, t_origin, args.max_frame_age)

    session = args.output_dir / f"{args.task}__{datetime.now():%Y%m%d_%H%M%S}"
    camera_list = ", ".join(f"{name} ({cam.attrs()['source']})" for name, cam in cameras.items()) or "none"
    cprint("\n" + "=" * 64, "cyan")
    cprint("🤖 robosimstudio demo recorder", "cyan", attrs=["bold"])
    cprint(f"   task     {args.task}\n   output   {session}\n   rate     {args.rate_hz:g} Hz\n"
           f"   cameras  {camera_list}\n"
           f"   depth    {'wrist' if 'wrist' in cameras and not args.no_wrist_depth else 'off'}\n"
           f"   teleop   {'off (--no-teleop)' if args.no_teleop else 'SpaceMouse'}"
           f"{'' if args.no_teleop else ', locked: ' + (' '.join(args.lock_rotation) or 'none')}",
           "white")
    cprint("=" * 64, "cyan")

    exit_code = 0
    try:
        cprint("⏳ Waiting for the robot and cameras...", "cyan")
        if not wait_for_sensors(robot, cameras):
            return 1
        cprint("✅ Robot and cameras streaming", "green")
        session.mkdir(parents=True, exist_ok=True)
        file_attrs = {
            "task": args.task, "robot": "panda", "rate_hz": args.rate_hz, "controller_type": CONTROLLER_TYPE,
            "reset_joint_positions": np.asarray(RESET_JOINT_POSITIONS), "interface_cfg": args.interface_cfg.read_text(),
            "gripper_convention": "robot/gripper_width in m (0.08 open); action/gripper -1 open, +1 close; "
                                  "sim/qpos[:, 7] 0 open .. 1 shut (robosimstudio joint_position)",
            "teleop": "none" if args.no_teleop else "spacemouse", "teleop_locked_rotations": " ".join(args.lock_rotation),
            "format_version": FORMAT_VERSION,
        }

        episode = 0
        while True:
            cprint(f"\n{'=' * 64}\n📝 Episode {episode}\n{'=' * 64}", "cyan")
            if home:
                cprint("🔄 Homing (gripper opens)...", "yellow")
                with teleop.control_lock:
                    reset_joints_to(robot, RESET_JOINT_POSITIONS, gripper_open=True)
            if args.duration is None:
                input("▶️  Press ENTER to START recording...")
            stem = session / f"episode_{episode:03d}"
            writer, reason = record_episode(stem, sampler, cameras, args.rate_hz,
                                            {**file_attrs, "start_time_iso": datetime.now().isoformat()},
                                            args.duration)
            if writer.n == 0:
                writer.delete()
                cprint("Nothing recorded; no files kept.", "yellow")
            elif reason not in ("interrupt", "duration") and \
                    input("❓ Keep this episode? [Y/n]: ").strip().lower().startswith("n"):
                writer.delete()
                cprint("🗑️  Deleted.", "yellow")
                continue  # same episode number again
            else:
                cprint(f"💾 Saved {writer.n} samples: {', '.join(p.name for p in writer.paths)}", "green")
            if reason == "error":
                exit_code = 1
            if reason in ("interrupt", "duration") or \
                    not input("❓ Record another episode? [y/N]: ").strip().lower().startswith("y"):
                break
            episode += 1
    except KeyboardInterrupt:
        cprint("\n⏹️  Interrupted", "yellow")
    finally:
        teleop.close()
        for cam in cameras.values():
            cam.close()
        try:
            robot.close()
        except Exception:  # noqa: BLE001 - never mask the original failure during cleanup
            pass
        rclpy.try_shutdown()
        if session.exists():
            cprint(f"\n🏁 Session folder: {session}", "green", attrs=["bold"])
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
