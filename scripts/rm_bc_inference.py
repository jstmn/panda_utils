"""robomimic BC-Transformer-GMM policy inference on the real Franka (ROS 2).

Structural port of scripts/act_clip_inference2.py -- same Rate helper, same camera
subscriber shape, same "inference loop fills action_holder, control thread consumes it"
split, same camera-staleness guard, same ENTER-to-reset, same teardown that refuses to
call a rollout clean if the control thread died. Only the policy and the controller
differ.

WHAT CHANGED vs act_clip_inference2.py
  * Policy: ACT-CLIP -> a robomimic checkpoint (`LeeHakHo/square_real30_ckpt`), loaded
    with `policy_from_checkpoint`. It is a BC_Transformer_GMM, so it needs a CONTEXT
    WINDOW, not a single frame: obs tensors are (1, context_length, ...) and the window
    is padded at episode start by repeating the first frame, exactly as robomimic's
    FrameStackWrapper does during training.
  * Controller: JOINT_IMPEDANCE (7 joint targets) -> OSC_POSE (6 Cartesian deltas +
    grasp), because that is the action space the dataset was built in.
  * Observation: the 18-dim joint state -> what the checkpoint's shape_metadata asks for,
    which is two 84x84 images plus robot0_eef_pos / robot0_eef_quat /
    robot0_gripper_qpos.
  * Cameras: two, with DIFFERENT topic layouts -- the D435 publishes
    /camera/d435/color/image_raw while the ZED Mini publishes
    /zed_multi/zedm/rgb/image_rect_color -- so the single COLOR_SUFFIX of the ACT script
    becomes the per-camera LAYOUTS table that log_demonstration2.py already uses.

TWO UNIT CONVERSIONS THAT MUST BE RIGHT
  1. The policy emits NORMALISED actions. `RolloutPolicy.__call__` would undo that, but it
     does not maintain the transformer's context window, so this script calls
     `algo.get_action` directly and applies the checkpoint's own
     `action_normalization_stats` itself:  metric = normalised * scale + offset.
     Skipping this makes the predictions ~160x too large -- measured, not guessed.
  2. deoxys multiplies before sending: `action[0:3] *= action_scale.translation` (0.05),
     `action[3:6] *= action_scale.rotation` (1.0), with `is_delta=True`. So a commanded
     metric delta d needs `action[0:3] = d / 0.05`. Rotation passes through as radians.
  3. ROTATION FRAME. The dataset's rotation action is a GRIPPER-frame axis-angle --
     r2st_to_robomimic builds it as `axis_angle(R[t].T @ R[t+1])`, i.e. R_next = R_cur @ R_d.
     deoxys applies it the other way round: osc_impedance.cpp:88 computes
     `goal_quat = AxisAngle(a).toRotationMatrix() * current_quat`, a LEFT multiply, which is
     a BASE-frame rotation. Sending the gripper-frame vector unchanged therefore executes a
     different rotation than the one demonstrated, for any orientation but identity.
     Since R_cur @ AxisAngle(v) @ R_cur.T == AxisAngle(R_cur @ v), the conversion is just
     rotating the axis-angle vector by the current end-effector orientation.

RATE MATTERS. The dataset's actions are per-frame FK deltas at 15 Hz, so the loop must
run at 15 Hz. Running faster does not merely add latency, it scales every commanded
motion by the rate ratio.

SAFETY
  Default is a DRY RUN: the policy runs on live cameras and every action is printed, but
  nothing is sent to the robot. Pass --execute to actually move it. Deltas are clamped
  per step, the end-effector is required to stay inside a workspace box, and the rollout
  stops at --max-steps. Keep a hand on the E-stop the first time.

Cameras and robot must already be running:
    ros2 launch realsense2_camera rs_launch.py camera_namespace:=camera camera_name:=d435 \
        align_depth.enable:=true rgb_camera.color_profile:=1280x720x30 \
        depth_module.depth_profile:=848x480x30
    CAMS=zedm bash ~/Desktop/ros2_ws/zed_tools/launch_all.sh
    (and the two NUC binaries -- see README_SETUP.md)

Usage:
    source /opt/ros/humble/setup.bash
    source ~/Desktop/ros2_ws/install/setup.bash
    source ~/anaconda3/bin/activate rmimic

    # look first: no motion
    python scripts/rm_bc_inference.py --checkpoint <path-or-hf>

    # then move
    python scripts/rm_bc_inference.py --checkpoint <path-or-hf> --execute
"""

from __future__ import annotations

import argparse
import select
import sys
import threading
import time
from collections import namedtuple

import numpy as np
import rclpy
import torch
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from termcolor import cprint

from deoxys.experimental.motion_utils import reset_joints_to
from deoxys.franka_interface import FrankaInterface
from deoxys.utils.config_utils import get_default_controller_config

np.set_printoptions(precision=4, suppress=True, linewidth=200)

# panda_utils.deoxys_controller.RESET_JOINT_POSITIONS, as act_clip_inference2.py inlines it
# RESET_JOINT_POSITIONS = [
#     #0.09162008, -0.19826458, -0.01990020, -2.47322699, -0.01307074, 2.30396583, 0.84809397,
#     0.55254022, 0.80662328, -0.43850005, -1.49100197, -1.17349812, 1.40865538, 2.8047689,
# ]

RESET_JOINT_POSITIONS = [
     0.51350000,  0.65620000,  0.06250000, -1.66850000, -1.21050000,  1.18390000,  1.44950000,
]

# Joint 6 is the wrist pitch. Offsetting only this one joint of the reset pose swings the zedm
# eye-in-hand view while the other six joints -- and so the EE position -- stay where they were;
# the hand pivots about the wrist rather than translating. Sign is not assumed here: + adds to the
# joint angle, and which way that tips the camera is for the operator to confirm with a small value.
# Bounds are the Panda's own limits for this joint.
WRIST_PITCH_JOINT = 5
WRIST_PITCH_LIMITS_RAD = (-0.0175, 3.7525)
# The same wrist motion expressed as a rotation of the hand, for --pregrasp-wrist-deg, which has to
# go out through OSC as a gripper-frame axis-angle rather than as a joint target. Measured by FK at
# the reset pose: joint 6 turns about [0.063, 0.998, 0.0] in the hand frame, i.e. the hand's +y.
WRIST_PITCH_AXIS_HAND = np.array([0.0, 1.0, 0.0])

CONTROLLER_TYPE = "OSC_POSE"
HF_REPO = "LeeHakHo/square_real100_bs128_ckpt"  # diffusion, 100 real demos, batch 128, lr 1.4e-4
# Arms here: baseline, aux_eef_frame -- epochs 100..1000 in steps of 100. The --hf-arm/--hf-epoch
# defaults (baseline, 100) exist in this repo, so the script runs unchanged; aux_world_frame and
# aux_obj_eef_frame do NOT, and asking for them prints the available list instead of a 404.
# This repo was still being uploaded on 2026-09-12 -- 900 and 1000 landed hours after 100..800, so
# re-check the epoch list on the Hub rather than trusting this comment if a run looks truncated.
# This run trains on ALL 100 demos with experiment.validate off, so there is no validation curve to
# pick an epoch by -- select on training loss or on offline replay, not on val.
# The batch-64 study (90/10 split, four arms, epochs 10..1000, val curve) is square_real100_ckpt.
# GMM set is square_real30_ckpt; the 60-demo repo is square_real60_r30val_ckpt.

# Which ROS topic each camera publishes colour on. The D435 and the ZED wrapper disagree,
# so this mirrors log_demonstration2.LAYOUTS rather than assuming one suffix for both.
CameraLayout = namedtuple("CameraLayout", "template color")
LAYOUTS = {
    "realsense": CameraLayout(template="/camera/{cam}", color="color/image_raw"),
    "zed": CameraLayout(template="/zed_multi/{cam}", color="rgb/image_rect_color"),
}

# camera id -> (layout, the obs key the checkpoint expects)
CAMERAS = {
    "d435": ("realsense", "agentview_image"),
    "zedm": ("zed", "robot0_eye_in_hand_image"),
}

LOW_DIM_KEYS = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]


class Rate:
    """rospy.Rate replacement -- see act_clip_inference2.Rate for why not rclpy.Rate."""

    def __init__(self, hz: float):
        self._period = 1.0 / float(hz)
        self._next = time.perf_counter() + self._period

    def sleep(self) -> None:
        now = time.perf_counter()
        delay = self._next - now
        if delay > 0.0:
            time.sleep(delay)
            self._next += self._period
        else:
            self._next = now + self._period


def make_qos(best_effort: bool) -> QoSProfile:
    return QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT if best_effort else ReliabilityPolicy.RELIABLE,
        history=HistoryPolicy.KEEP_LAST, depth=10, durability=DurabilityPolicy.VOLATILE)


def decode_color(msg: Image) -> np.ndarray:
    """Colour Image -> HxWx3 uint8 RGB, matching log_demonstration2.decode_color."""
    buf = np.frombuffer(msg.data, dtype=np.uint8)
    if msg.encoding in ("bgra8", "rgba8"):
        img = buf.reshape(msg.height, msg.width, 4)
        img = img[:, :, [2, 1, 0]] if msg.encoding == "bgra8" else img[:, :, :3]
    elif msg.encoding in ("bgr8", "rgb8"):
        img = buf.reshape(msg.height, msg.width, 3)
        if msg.encoding == "bgr8":
            img = img[:, :, ::-1]
    else:
        raise ValueError(f"unsupported encoding {msg.encoding!r}")
    return np.ascontiguousarray(img)


def to_policy_image(rgb: np.ndarray, size: int = 84) -> np.ndarray:
    """HxWx3 uint8 RGB -> (3,size,size) float32 in [0,1].

    Centre-crop then resize, identical to r2st_to_robomimic._square_resize. Getting this
    wrong is silent: the policy still returns confident actions for a differently framed
    image than it was trained on.
    """
    import cv2

    h, w = rgb.shape[:2]
    s = min(h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    img = cv2.resize(rgb[y0:y0 + s, x0:x0 + s], (size, size), interpolation=cv2.INTER_AREA)
    return img.transpose(2, 0, 1).astype(np.float32) / 255.0


# Measured on this rig 2026-09-08: commanded OSC_POSE goal offset (per 15 Hz step) vs the
# displacement the arm actually achieves. Kp.translation is 150 N/m, so a 2 mm offset pulls
# with 0.3 N and the arm does not move at all; nothing happens below ~5 mm and the response
# only becomes roughly proportional past 20 mm.
#
#   offset/step   2mm   5mm   10mm   20mm   40mm
#   achieved      0.00  0.02  0.78   4.01   9.35   (mm per step)
#
# The dataset's actions are ACHIEVED displacements (FK differences), not goal offsets: the
# demonstrations were driven by a human holding the SpaceMouse far off-centre, i.e. offsets
# 5-25x larger than the motion they produced. Replaying achieved displacements as offsets
# therefore commands a force that cannot move the arm. A constant --action-gain does not fix
# it either: multiplying a 0.5 mm intent by 7 gives 3.5 mm, still inside the dead band, which
# is exactly the fine-positioning the policy needs near the nut.
_TRANSFER_ACHIEVED_MM = [0.0, 0.02, 0.78, 4.01, 9.35]
_TRANSFER_OFFSET_MM = [0.0, 5.0, 10.0, 20.0, 40.0]


# Rotation's version of the translation table below, measured with scripts/r2st_rot_transfer.py on
# 2026-09-13 from the reset pose at 14.4 Hz with deoxys' stock gains (Kp 150/250, LINEAR_POSE
# time_fraction 0.3), 8 amplitudes x 3 repeats per axis. Achieved rad/step, hand frame:
#
#   commanded   0.010     0.015     0.020     0.025     0.030     0.040     0.050
#   about y    -0.00001  -0.00000  +0.00000  +0.00014  +0.00127  +0.00294  +0.00464
#   about z    +0.00001  +0.00006  +0.00016  +0.00070  +0.00163  +0.00376  +0.00646
#   about x    -0.00000  -0.00000  -0.00000  -0.00001  -0.00002  -0.00000  +0.00004
#
# Two things fall out of that. Each axis has its own deadband -- y does nothing until ~0.022,
# z until ~0.012 -- so one shared curve cannot serve both; y was the axis the old three-point
# table was measured on, which is why z came back under-driven and y over-driven in rollouts.
#
# And the hand does not turn about its own x axis at all at this pose: every x amplitude reads
# zero to within noise, while the SAME sweep turns the hand about z instead (-0.00049 at 0.015
# rising to -0.00593 at 0.050). Joint 7 gives hand z and joint 6 gives hand y, but hand x needs
# the whole arm to cooperate, and inside one 70 ms step it does not. So x is passed through
# untouched here: inflating it would buy no x rotation and inject z rotation that nothing asked
# for. A policy that wants hand-x rotation cannot have it through this controller.
#
# The tables map desired -> the offset that produces it, so they are the measurement inverted.
# The first entry is the deadband edge rather than 0: below it every offset achieves nothing, and
# the edge is the smallest command that is about to move, so a tiny request clears the deadband
# instead of being swallowed by it.
_ROT_TABLE = {
    1: ([0.0, 0.00014, 0.00127, 0.00294, 0.00464],
        [0.022, 0.025, 0.030, 0.040, 0.050]),
    2: ([0.0, 0.00006, 0.00016, 0.00070, 0.00163, 0.00376, 0.00646],
        [0.012, 0.015, 0.020, 0.025, 0.030, 0.040, 0.050]),
}


# A position goal turns the hand whether or not anything asked it to: the arm pivots about joint 6,
# 23 cm behind the fingertips. Measured 2026-09-13 with r2st_rot_transfer.py --mode trans --axis z
# (3 repeats, 15 steps, 5 settle steps dropped), zero-amplitude control subtracted:
#
#   z goal offset   0.002     0.005     0.008     0.012     0.015
#   induced y      +0.00016  +0.00086  +0.00267  +0.00413  +0.00376
#
# That is the same size as what the policy commands (0.001-0.005 rad/step), which is why it can
# swallow the demonstrated wrist lift whole: a rollout descending with ~4 mm z offsets carried a
# -0.0007 rad/step bias on the wrist, -23 deg over the run, against a commanded +16 deg.
#
# Only the z axis is measured, and only out to 15 mm, where the trend turns over -- probably the
# trial travelling 46 mm and changing configuration underneath itself. Past 12 mm the value is
# held flat rather than followed down.
_COUPLING_Z_OFFSET = [0.0, 0.002, 0.005, 0.008, 0.012]
_COUPLING_Z_INDUCED = [0.0, 0.00016, 0.00086, 0.00267, 0.00413]
_COUPLING_Z_MAX = 0.012


# The table was measured holding one offset for 15 steps; a rollout changes its goal every step and
# never reaches that steady state. Applied at face value to /tmp/rot_peraxis.npz the table predicts
# a mean bias of -0.00188 rad/step where the rollout actually carried -0.00070, so it over-predicts
# by 2.7x and would correct more than the error. This gain is that ratio, and nothing more
# principled: re-derive it whenever the table is re-measured.
_COUPLING_GAIN = 0.37


def coupling_induced_rot(pos_offset: np.ndarray, gain: float = _COUPLING_GAIN) -> np.ndarray:
    """Hand-frame rotation the translation goal will produce on its own, to be subtracted."""
    oz = float(pos_offset[2])
    if abs(oz) < 1e-9:
        return np.zeros(3)
    v = float(np.interp(min(abs(oz), _COUPLING_Z_MAX), _COUPLING_Z_OFFSET, _COUPLING_Z_INDUCED))
    return np.array([0.0, gain * np.sign(oz) * v, 0.0])


def rot_achieved_to_offset(d_rad: np.ndarray, cap_rad: float) -> np.ndarray:
    """Rotation the policy asks for -> goal offset that actually produces it, per axis.

    Per axis and not on the vector's magnitude: the deadbands differ by a factor of two, so a
    magnitude-based curve inflates whichever axis it was not measured on by the wrong amount.
    Treating the three components independently is only exact for small rotations, which is all
    a 14.4 Hz policy ever commands.
    """
    out = np.array(d_rad, dtype=float).copy()   # x passes through: no authority, see above
    for i, (ach, off) in _ROT_TABLE.items():
        mag = abs(float(d_rad[i]))
        if mag < 1e-9:
            out[i] = 0.0
            continue
        if mag > ach[-1]:
            slope = (off[-1] - off[-2]) / (ach[-1] - ach[-2])
            o = off[-1] + (mag - ach[-1]) * slope
        else:
            o = float(np.interp(mag, ach, off))
        out[i] = np.sign(d_rad[i]) * min(o, cap_rad)
    return out


def achieved_to_offset(d_m: np.ndarray, cap_m: float) -> np.ndarray:
    """Metric displacement the policy asks for -> goal offset that actually produces it."""
    mag = float(np.linalg.norm(d_m))
    if mag < 1e-9:
        return np.zeros(3)
    want_mm = mag * 1000.0
    if want_mm > _TRANSFER_ACHIEVED_MM[-1]:
        # past the measured range, continue with the slope of the last segment
        slope = ((_TRANSFER_OFFSET_MM[-1] - _TRANSFER_OFFSET_MM[-2])
                 / (_TRANSFER_ACHIEVED_MM[-1] - _TRANSFER_ACHIEVED_MM[-2]))
        off_mm = _TRANSFER_OFFSET_MM[-1] + (want_mm - _TRANSFER_ACHIEVED_MM[-1]) * slope
    else:
        off_mm = float(np.interp(want_mm, _TRANSFER_ACHIEVED_MM, _TRANSFER_OFFSET_MM))
    off_m = min(off_mm / 1000.0, cap_m)
    return d_m / mag * off_m


class RGBSubscriber(Node):
    """latest_rgb(cam_id) -> most recent frame or None, as in act_clip_inference2."""

    def __init__(self, camera_ids: list[str], best_effort: bool):
        super().__init__("rm_bc_inference_cameras")
        self._camera_ids = list(camera_ids)
        self._latest: dict[str, np.ndarray | None] = {c: None for c in camera_ids}
        self._lock = threading.Lock()
        qos = make_qos(best_effort)
        self._subs = []
        for cam_id in camera_ids:
            layout = LAYOUTS[CAMERAS[cam_id][0]]
            topic = f"{layout.template.format(cam=cam_id).rstrip('/')}/{layout.color}"
            self._subs.append(self.create_subscription(
                Image, topic, lambda m, c=cam_id: self._cb(m, c), qos))
            cprint(f"   subscribed: {topic}  -> {CAMERAS[cam_id][1]}", "white")

    def _cb(self, msg: Image, cam_id: str) -> None:
        try:
            img = decode_color(msg)
        except ValueError as exc:
            self.get_logger().warn(f"{cam_id}: {exc}")
            return
        with self._lock:
            self._latest[cam_id] = img

    def latest_rgb(self, cam_id: str) -> np.ndarray | None:
        with self._lock:
            return self._latest[cam_id]

    def wait_for_frames(self, timeout_s: float = 20.0) -> bool:
        cprint(f"⏳ Waiting for first colour frames from: {self._camera_ids}", "cyan")
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if not [c for c in self._camera_ids if self.latest_rgb(c) is None]:
                cprint("✅ All cameras streaming", "green")
                return True
            time.sleep(0.2)
        missing = [c for c in self._camera_ids if self.latest_rgb(c) is None]
        cprint(f"❌ No frames from: {missing} after {timeout_s:.0f}s", "red")
        cprint("   If topics exist but no data arrives, retry with --qos-best-effort", "red")
        return False


def wait_for_deoxys_ready(robot: FrankaInterface, timeout_s: float = 10.0) -> bool:
    cprint("⏳ Waiting for deoxys state ...", "cyan")
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if robot.last_q is not None and robot.last_gripper_q is not None:
            cprint("✅ Robot interface ready", "green")
            return True
        time.sleep(0.1)
    cprint("❌ No joint/gripper state from deoxys -- are the NUC binaries running?", "red")
    return False


def enter_is_pressed() -> bool:
    if select.select([sys.stdin], [], [], 0.0)[0] == [sys.stdin]:
        return sys.stdin.read(1) == "\n"
    return False


def np_array_to_hash(arr: np.ndarray) -> int:
    return int(arr[::37, ::37].sum())


class Policy:
    """robomimic checkpoint + the context window and unnormalisation robomimic's
    RolloutPolicy does not do for a transformer."""

    def __init__(self, ckpt_path: str, device: str = "cuda"):
        from robomimic.utils.file_utils import policy_from_checkpoint

        rp, ck = policy_from_checkpoint(ckpt_path=ckpt_path, device=device, verbose=False)
        self.algo = rp.policy
        self.device = device
        self.algo_name = ck["algo_name"]
        # How many past frames the network wants stacked on the time axis. A BC transformer
        # takes its whole context window; a diffusion policy takes only
        # horizon.observation_horizon (2 here) and predicts a chunk of future actions from
        # it. Reading the wrong field silently feeds the net the wrong tensor shape.
        acfg = self.algo.global_config.algo
        if self.algo_name == "diffusion_policy":
            self.ctx = int(acfg.horizon.observation_horizon)
            self.action_horizon = int(acfg.horizon.action_horizon)
        else:
            self.ctx = int(acfg.transformer.context_length)
            self.action_horizon = 1
        stats = rp.action_normalization_stats["actions"]
        self.offset = np.asarray(stats["offset"], dtype=np.float64).reshape(-1)
        self.scale = np.asarray(stats["scale"], dtype=np.float64).reshape(-1)
        self.epoch = ck["variable_state"]["epoch"]
        self.obs_keys = list(ck["shape_metadata"]["all_obs_keys"])
        self._hist: list[dict[str, np.ndarray]] = []
        self.algo.set_eval()
        self.reset()
        cprint(f"✅ policy: {type(self.algo).__name__} ({self.algo_name})  epoch {self.epoch}  "
               f"obs context {self.ctx}  action horizon {self.action_horizon}", "green")

    def reset(self) -> None:
        self._hist.clear()
        # DiffusionPolicyUNet buffers a chunk of predicted actions and pops one per call.
        # Without this the queue survives across episodes and the first actions of a new
        # rollout are leftovers planned from the previous scene.
        if hasattr(self.algo, "reset"):
            self.algo.reset()

    def __call__(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        # pad_frame_stack: before the window is full, repeat the first observation, which
        # is what the training loader did for the first frames of every demo.
        self._hist.append(obs)
        if len(self._hist) > self.ctx:
            self._hist.pop(0)
        window = [self._hist[0]] * (self.ctx - len(self._hist)) + self._hist
        od = {k: torch.from_numpy(np.stack([w[k] for w in window])).unsqueeze(0).to(self.device)
              for k in self.obs_keys}
        with torch.no_grad():
            ac = self.algo.get_action(obs_dict=od)
        ac = ac.detach().cpu().numpy()[0].astype(np.float64)
        return ac * self.scale + self.offset          # normalised -> metric


def reset_pose_with_wrist_pitch(deg: float) -> list[float]:
    """RESET_JOINT_POSITIONS with the wrist-pitch joint offset by `deg` degrees."""
    pose = list(RESET_JOINT_POSITIONS)
    q = pose[WRIST_PITCH_JOINT] + float(np.radians(deg))
    lo, hi = WRIST_PITCH_LIMITS_RAD
    if not lo <= q <= hi:
        span = (np.degrees(lo - pose[WRIST_PITCH_JOINT]), np.degrees(hi - pose[WRIST_PITCH_JOINT]))
        raise ValueError(
            f"--wrist-pitch-deg {deg:+.1f} puts joint {WRIST_PITCH_JOINT + 1} at {q:.4f} rad, outside "
            f"the Panda's [{lo}, {hi}]; from this reset pose the usable range is "
            f"{span[0]:+.1f} .. {span[1]:+.1f} deg")
    pose[WRIST_PITCH_JOINT] = q
    return pose


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=str, default=None,
                   help="local .pth, or omit and use --hf-arm/--hf-epoch to pull from the Hub")
    p.add_argument("--hf-repo", default=HF_REPO,
                   help=f"HF model repo holding <arm>/model_epoch_N.pth (default {HF_REPO})")
    p.add_argument("--hf-arm", default="baseline", #aux_world_frame
                   help="checkpoint subfolder: baseline / aux_world_frame / aux_eef_frame / "
                        "aux_obj_eef_frame for the diffusion repo, baseline / aux_obj_eef_frame "
                        "for the GMM one, <task>/<arm> for the multi-task repos (e.g. "
                        "stackcube_real60/baseline)")
    p.add_argument("--hf-epoch", type=int, default=100)
    p.add_argument("--grasp-threshold", type=float, default=0.0,
                   help="close the gripper when the policy's grasp output exceeds this (default 0.0, the "
                        "neutral cut for a +1 closed / -1 open label). The output saturates, so a real "
                        "grasp reads +1.00 and a real release -1.00; measured on hardware, lowering this "
                        "to -0.95 bought nothing and fired one spurious close on a -0.90 during the "
                        "approach. Lower it only to probe a checkpoint that never crosses zero")
    p.add_argument("--grasp-debounce", "--grasp_debounce", type=int, default=0,
                   help="require the grasp sign to hold for this many policy steps before the gripper is "
                        "actually commanded to flip (default 0, off). Every demonstration flips exactly "
                        "twice and its closed phase runs at least 172 frames, so any value up to ~170 "
                        "removes only flips the data never contained; measured on hardware, a rollout "
                        "flipped 8 times including one- and four-step blips that moved the fingers "
                        "mid-grasp. Note this cannot help when the output swings fully from +1 to -1")
    p.add_argument("--auto-grasp-on-stall", "--auto_grasp_on_stall", action="store_true",
                   help="close the gripper when the arm STALLS low, whatever the policy says. The demos "
                        "move 2.2-2.6 mm per step while approaching, and a stalled rollout moved 2 mm in "
                        "700 steps, so a stall is unambiguous. Engineering override, not the policy")
    p.add_argument("--stall-window", type=int, default=15,
                   help="how many steps of near-zero motion count as a stall (default 15, ~1 s)")
    p.add_argument("--stall-move-mm", type=float, default=2.0,
                   help="total mm the eef may move across that window and still count as stalled (default 2)")
    p.add_argument("--stall-z-max", type=float, default=0.03,
                   help="only auto-grasp below this height, m (default 0.03). Keeps the ~20 idle frames at "
                        "the start pose, z 0.265, from triggering it")
    p.add_argument("--auto-release-on-stall", "--auto_release_on_stall", action="store_true",
                   help="the counterpart to --auto-grasp-on-stall: once the gripper has been "
                        "closed for --release-min-closed steps, open it when the arm stalls "
                        "again. A policy that will not let go strands the episode -- the drawer "
                        "demos release the handle before going for the arrow")
    p.add_argument("--release-window", type=int, default=15,
                   help="steps the arm must stay within --release-move-mm to count as stalled "
                        "while holding (default 15, as --stall-window)")
    p.add_argument("--release-move-mm", type=float, default=2.0,
                   help="movement over --release-window below which the arm counts as stalled "
                        "while holding, mm (default 2.0)")
    p.add_argument("--release-min-closed", type=int, default=45,
                   help="steps the gripper must have been closed before a release can fire "
                        "(default 45, ~3 s at 14.4 Hz). Stops it opening during the grasp itself, "
                        "which always begins with the arm stationary")
    p.add_argument("--auto-release-hold", type=int, default=60,
                   help="steps to hold the auto-release open before handing the gripper back to "
                        "the policy (default 60, as --auto-grasp-hold)")
    p.add_argument("--auto-grasp-hold", type=int, default=60,
                   help="steps to hold the auto-grasp closed before handing the gripper back to the policy "
                        "(default 60, ~4 s)")
    p.add_argument("--camera_ids", nargs="*", default=["d435", "zedm"])
    p.add_argument("--interface-cfg", type=str, default=None)
    p.add_argument("--controller-cfg", "--controller_cfg", type=str, default=None,
                   help="OSC controller yaml to use instead of the deoxys default. "
                        "configs/osc-pose-tuned.yml is a copy of that default, kept here so a gain "
                        "sweep cannot change what every other script on this machine runs")
    p.add_argument("--kp-rotation", "--kp_rotation", type=float, default=None,
                   help="override Kp.rotation (deoxys default 250). Raise it if the hand does not "
                        "hold the orientation the policy asks for")
    p.add_argument("--kp-translation", "--kp_translation", type=float, default=None,
                   help="override Kp.translation (deoxys default 150)")
    p.add_argument("--interp-time-fraction", "--interp_time_fraction", type=float, default=None,
                   help="override traj_interpolator_cfg.time_fraction (deoxys default 0.3): the "
                        "fraction of the control step spent moving toward the commanded offset "
                        "before the next one replaces it")
    p.add_argument("--rate", type=float, default=15.0,
                   help="policy step rate. The dataset's actions are per-frame deltas at whatever rate "
                        "it was recorded at, measured at 14.4 Hz on these sessions, so pass --rate 14.4 "
                        "to match it")
    p.add_argument("--control-rate", type=float, default=None,
                   help="how often the CONTROLLER is handed the current action, Hz. Default: same as "
                        "--rate, i.e. one OSC command per policy step. The demonstrations were driven at "
                        "TELEOP_HZ = 50 Hz (log_demonstration2.py), so each recorded 14.4 Hz frame carried "
                        "~3.5 commands; is_delta recomputes the goal from the CURRENT pose on every send, "
                        "so --control-rate 50 reproduces that drive without touching the policy rate. The "
                        "arm pushes harder this way -- --max-offset and the workspace box still apply")
    p.add_argument("--dump-obs", "--dump_obs", type=str, nargs="?", default=None, const="/tmp/rollout.npz",
                   help="write every observation the policy actually saw to this .npz (84px images as uint8, "
                        "eef pose, gripper, the action taken). A bare --dump-obs writes /tmp/rollout.npz. Off by "
                        "default. A single-frame comparison against the training set cannot show WHEN a rollout "
                        "leaves the training distribution; this can")
    p.add_argument("--action-horizon", "--action_horizon", type=int, default=None,
                   help="how many of the 16 predicted actions to execute before looking at the cameras again. "
                        "The checkpoint's own value is 8, which at 14.4 Hz means 0.55 s of open-loop motion "
                        "between observations; the diffusion refill costs ~29 ms, so even 1 fits the step "
                        "budget. Default: leave the trained value alone")
    p.add_argument("--device", default="cuda")
    p.add_argument("--execute", action="store_true",
                   help="actually command the robot. Without it this is a dry run.")
    p.add_argument("--max-steps", type=int, default=1500, help="stop after N policy steps (~27 s at 15 Hz)")
    p.add_argument("--action-gain", type=float, default=0.5,
                   help="multiply the policy's metric delta before clamping. The honest way to "
                        "speed the arm up: the loop stays at the trained 15 Hz, so the policy "
                        "still sees the scene change at the rate it was trained on -- only the "
                        "commanded step grows. Raising --rate instead would do that AND put the "
                        "observations out of distribution. Above ~2 expect overshoot, because BC "
                        "has no feedback term to correct one.")
    p.add_argument("--transfer-curve", action="store_true", default=True,
                   help="map the policy's intended displacement through the measured "
                        "offset->motion curve instead of a flat gain (default on)")
    p.add_argument("--no-transfer-curve", dest="transfer_curve", action="store_false")
    p.add_argument("--max-offset", type=float, default=0.06,
                   help="cap on the commanded goal offset, m (safety, applies after the curve)")
    p.add_argument("--max-dpos", type=float, default=0.015,
                   help="per-step clamp on the INTENDED displacement, m")
    p.add_argument("--max-drot", type=float, default=0.05, help="per-step rotation clamp, rad")
    p.add_argument("--rot-transfer-curve", "--rot_transfer_curve", action="store_true",
                   help="inflate the rotation command the way --transfer-curve inflates "
                        "translation, using the measured table above. Off by default: the table "
                        "rests on one sweep of three points, so it is opt-in until a policy run "
                        "shows it helps")
    p.add_argument("--rot-coupling-comp", "--rot_coupling_comp", action="store_true",
                   help="subtract the rotation the translation goal drags in, using the measured "
                        "coupling table, before the rotation curve runs. Needs --transfer-curve "
                        "on to mean anything, since it reads the inflated goal")
    p.add_argument("--rot-coupling-gain", "--rot_coupling_gain", type=float, default=_COUPLING_GAIN,
                   help=f"how much of the predicted coupling to remove (default {_COUPLING_GAIN}, "
                        "the ratio between the bias a rollout actually carried and the bias the "
                        "table predicts for it). 1.0 applies the table as measured and over-corrects")
    p.add_argument("--max-rot-offset", "--max_rot_offset", type=float, default=0.15,
                   help="cap on the inflated rotation offset, rad per step (default 0.15)")
    p.add_argument("--ws-min", type=float, nargs=3, default=[0.20, -0.40, -0.05],
                   help="workspace box lower corner in the robot base frame, m")
    p.add_argument("--ws-max", type=float, nargs=3, default=[0.80, 0.40, 0.60])
    p.add_argument("--no-reset", action="store_true", help="skip the move to RESET_JOINT_POSITIONS")
    p.add_argument("--wrist-pitch-deg", "--wrist_pitch_deg", type=float, default=0.0,
                   help="offset the reset pose's wrist-pitch joint (joint 6) by this many degrees, "
                        "moving the eye-in-hand view without moving the other six joints. + adds to "
                        "the joint angle; confirm which way that tips the hand with a small value "
                        "first. 0 (default) keeps the pose this script has always used")
    p.add_argument("--pregrasp-wrist-deg", "--pregrasp_wrist_deg", type=float, default=0.0,
                   help="tilt the wrist by this many degrees right before the fingers close, on the "
                        "rising edge of the grasp command (the policy's own, the debounced one, or "
                        "the --auto-grasp-on-stall override, whichever fires). Same direction "
                        "convention as --wrist-pitch-deg. 0 (default) disables it")
    p.add_argument("--pregrasp-wrist-steps", "--pregrasp_wrist_steps", type=int, default=5,
                   help="spread --pregrasp-wrist-deg over this many inference steps (default 5, "
                        "~0.35 s at 14.4 Hz). The policy's action is dropped for those steps so the "
                        "tilt is not fighting a translation, and the gripper stays open")
    p.add_argument("--qos-best-effort", action="store_true")
    cli = p.parse_args()

    ckpt = cli.checkpoint
    if ckpt is None:
        from huggingface_hub import HfApi, hf_hub_download

        # Arm names differ per repo (the GMM set has aux_obj_eef_frame, the diffusion set
        # has aux_world_frame / aux_eef_frame), and asking for the wrong one otherwise
        # surfaces as a 60-line HTTP 404 traceback. Check first and say what is there.
        want = f"{cli.hf_arm}/model_epoch_{cli.hf_epoch}.pth"
        files = [s.rfilename for s in HfApi().repo_info(cli.hf_repo, repo_type="model").siblings]
        if want not in files:
            # An arm is the whole path in front of the file name, not the first component: the
            # multi-task repos nest it one level deeper (mimicgen_aux_real_checkpoints holds
            # <task>/<arm>/model_epoch_N.pth), and printing just the task there names something
            # --hf-arm will not accept.
            arms = sorted({f.rsplit("/", 1)[0] for f in files if f.endswith(".pth") and "/" in f})
            eps = sorted({int(f.split("_")[-1][:-4]) for f in files
                          if f.endswith(".pth") and f.rsplit("/", 1)[0] == cli.hf_arm})
            cprint(f"{cli.hf_repo} has no {want}", "red")
            cprint(f"  arms in this repo : {arms}", "yellow")
            if eps:
                cprint(f"  epochs for {cli.hf_arm}: {eps}", "yellow")
            else:
                cprint(f"  '{cli.hf_arm}' is not one of them -- pass --hf-arm from the list above,", "yellow")
                cprint("  or --hf-repo to point at the repo that does have it.", "yellow")
            return 1
        ckpt = hf_hub_download(cli.hf_repo, want, repo_type="model")
        cprint(f"checkpoint: {cli.hf_repo} :: {want}", "cyan")

    for c in cli.camera_ids:
        assert c in CAMERAS, f"unknown camera {c!r}; known: {sorted(CAMERAS)}"

    try:
        reset_pose = reset_pose_with_wrist_pitch(cli.wrist_pitch_deg)
    except ValueError as exc:
        cprint(f"❌ {exc}", "red")
        return 1
    if cli.wrist_pitch_deg:
        cprint(f"wrist pitch: joint {WRIST_PITCH_JOINT + 1} "
               f"{RESET_JOINT_POSITIONS[WRIST_PITCH_JOINT]:.4f} -> {reset_pose[WRIST_PITCH_JOINT]:.4f} rad "
               f"({cli.wrist_pitch_deg:+.1f} deg)", "cyan")
        if cli.no_reset:
            cprint("  --no-reset is set, so the arm will not be moved there this run.", "yellow")

    if cli.pregrasp_wrist_deg:
        if cli.pregrasp_wrist_steps < 1:
            cprint("❌ --pregrasp-wrist-steps must be at least 1", "red")
            return 1
        # The override is written straight into the action, past the --max-drot clamp, so the
        # per-step size has to be checked here instead.
        per_step = abs(float(np.radians(cli.pregrasp_wrist_deg))) / cli.pregrasp_wrist_steps
        if per_step > cli.max_drot:
            need = int(np.ceil(abs(float(np.radians(cli.pregrasp_wrist_deg))) / cli.max_drot))
            cprint(f"❌ --pregrasp-wrist-deg {cli.pregrasp_wrist_deg:+.1f} over "
                   f"{cli.pregrasp_wrist_steps} steps is {per_step:.4f} rad/step, past --max-drot "
                   f"({cli.max_drot}); use --pregrasp-wrist-steps {need} or more", "red")
            return 1
        cprint(f"pre-grasp wrist tilt: {cli.pregrasp_wrist_deg:+.1f} deg over "
               f"{cli.pregrasp_wrist_steps} steps ({per_step:.4f} rad/step) when the grasp fires", "cyan")

    policy = Policy(ckpt, device=cli.device)
    if cli.action_horizon is not None:
        # DiffusionPolicyUNet reads override_action_horizon on every get_action call, so this needs no
        # retraining: it only changes how much of each predicted chunk is executed before re-planning.
        policy.algo.override_action_horizon = int(cli.action_horizon)
        cprint(f"action horizon: {policy.action_horizon} -> {cli.action_horizon} "
               f"(re-plan every {cli.action_horizon} step{'s' if cli.action_horizon > 1 else ''}, "
               f"{cli.action_horizon / cli.rate * 1000:.0f} ms open-loop)", "yellow")
    ws_min, ws_max = np.asarray(cli.ws_min), np.asarray(cli.ws_max)

    cprint("=" * 78, "white")
    cprint(f"{'EXECUTE - the robot WILL move' if cli.execute else 'DRY RUN - nothing is sent to the robot'}",
           "red" if cli.execute else "yellow")
    cprint(f"transfer curve: {'ON (measured offset->motion inverse)' if cli.transfer_curve else 'OFF (raw delta)'}"
           f"   max offset {cli.max_offset*1000:.0f} mm", "white")
    cprint(f"rotation curve: {'ON (measured per axis, opt-in)' if cli.rot_transfer_curve else 'OFF (raw radians)'}"
           f"   max rot offset {cli.max_rot_offset:.3f} rad"
           f"   coupling comp {f'ON (gain {cli.rot_coupling_gain})' if cli.rot_coupling_comp else 'OFF'}", "white")
    if cli.auto_release_on_stall:
        cprint(f"auto-release: after {cli.release_min_closed} closed steps, on "
               f"<{cli.release_move_mm} mm over {cli.release_window} steps; holds open "
               f"{cli.auto_release_hold} steps", "white")
    cprint(f"rate {cli.rate} Hz   control {cli.control_rate or cli.rate} Hz   gain {cli.action_gain}   "
           f"max_steps {cli.max_steps}   clamps {cli.max_dpos*1000:.0f} mm / {cli.max_drot:.3f} rad   "
           f"grasp threshold {cli.grasp_threshold:+.2f}", "white")
    if cli.action_gain != 1.0:
        cprint(f"   NOTE: commanded motion is {cli.action_gain}x the policy's own output", "yellow")
        if cli.transfer_curve:
            # The curve already converts an intended displacement into the offset that
            # produces it, so a gain on top scales the motion a second time -- the policy's
            # 2 mm becomes 4 mm of intent, which the curve then asks 20 mm for.
            cprint("   WARNING: --action-gain is applied BEFORE the transfer curve, so the two "
                   "compound.", "red")
            cprint(f"            The arm will move about {cli.action_gain}x demonstration speed. "
                   "Use --action-gain 1 with the curve.", "red")
    cprint(f"workspace box  min {ws_min}  max {ws_max}", "white")
    cprint("=" * 78, "white")

    rclpy.init(args=None)
    cams = RGBSubscriber(cli.camera_ids, cli.qos_best_effort)

    SHUTDOWN = False
    spin_stop = threading.Event()

    def spin_target() -> None:
        while rclpy.ok() and not spin_stop.is_set():
            rclpy.spin_once(cams, timeout_sec=0.05)

    spin_thread = threading.Thread(target=spin_target, daemon=True)
    spin_thread.start()

    if not cams.wait_for_frames():
        spin_stop.set(); cams.destroy_node(); rclpy.shutdown(); return 1

    cfg = cli.interface_cfg or str((__import__("pathlib").Path(__file__).resolve().parents[1] / "configs" / "charmander.yml"))
    robot = FrankaInterface(cfg, use_visualizer=False)
    if not wait_for_deoxys_ready(robot):
        spin_stop.set(); robot.close(); cams.destroy_node(); rclpy.shutdown(); return 1

    if cli.execute and not cli.no_reset:
        cprint("Resetting to the start configuration ...", "yellow")
        reset_joints_to(robot, reset_pose, gripper_open=True)

    if cli.controller_cfg:
        from deoxys.utils.yaml_config import YamlConfig
        controller_cfg = YamlConfig(cli.controller_cfg).as_easydict()
        if controller_cfg.get("controller_type") != CONTROLLER_TYPE:
            cprint(f"❌ {cli.controller_cfg} is a {controller_cfg.get('controller_type')} config, "
                   f"but this script drives {CONTROLLER_TYPE}", "red")
            spin_stop.set(); robot.close(); cams.destroy_node(); rclpy.shutdown(); return 1
        cprint(f"controller cfg: {cli.controller_cfg}", "cyan")
    else:
        controller_cfg = get_default_controller_config(controller_type=CONTROLLER_TYPE)
    # Overrides go on top of whichever file was loaded, so the run's own log records the gains it
    # ran with instead of leaving them to whatever the yaml happened to hold at the time.
    for val, path in ((cli.kp_rotation, ("Kp", "rotation")),
                      (cli.kp_translation, ("Kp", "translation")),
                      (cli.interp_time_fraction, ("traj_interpolator_cfg", "time_fraction"))):
        if val is not None:
            was = controller_cfg[path[0]][path[1]]
            controller_cfg[path[0]][path[1]] = val
            cprint(f"  {'.'.join(path)}: {was} -> {val}", "cyan")
    tr_scale = float(controller_cfg.action_scale.translation)
    rot_scale = float(controller_cfg.action_scale.rotation)
    cprint(f"OSC_POSE action_scale: translation {tr_scale}  rotation {rot_scale}  is_delta {controller_cfg.is_delta}", "white")
    cprint(f"OSC_POSE Kp: translation {controller_cfg.Kp.translation}  rotation {controller_cfg.Kp.rotation}"
           f"   interpolator time_fraction {controller_cfg.traj_interpolator_cfg.time_fraction}", "white")

    action_holder: list[np.ndarray | None] = [None]
    # The control thread needs the orientation the action was computed at, to map the
    # gripper-frame rotation into the base frame deoxys expects.
    eef_rot_holder: list[np.ndarray | None] = [None]
    control_failed = [False]
    # One Rate per thread. A shared Rate is not thread-safe in the way that matters here:
    # both callers advance the same `_next`, so each loop wakes at half the requested rate
    # and the arm crawls at half speed. act_clip_inference2.py keeps CONTROL_RATE and
    # INFERENCE_RATE separate for exactly this reason.
    CONTROL_RATE = Rate(cli.control_rate or cli.rate)
    INFERENCE_RATE = Rate(cli.rate)

    def control_loop() -> None:
        nonlocal SHUTDOWN
        while not SHUTDOWN and rclpy.ok():
            CONTROL_RATE.sleep()
            a = action_holder[0]
            if a is None:
                continue
            try:
                if cli.execute:
                    # Gripper-frame axis-angle -> base frame, which is what deoxys applies.
                    R_cur = eef_rot_holder[0]
                    dpos = achieved_to_offset(a[0:3], cli.max_offset) if cli.transfer_curve else a[0:3]
                    # dpos first: the rotation the position goal drags in has to come off the
                    # policy's request before the rotation curve inverts what is left.
                    want_rot = a[3:6]
                    if cli.rot_coupling_comp:
                        want_rot = want_rot - coupling_induced_rot(dpos, cli.rot_coupling_gain)
                    drot = (rot_achieved_to_offset(want_rot, cli.max_rot_offset)
                            if cli.rot_transfer_curve else want_rot)
                    drot = drot if R_cur is None else (R_cur @ drot)
                    # deoxys scales what it is given, so divide the metric delta out first
                    cmd = np.concatenate([dpos / tr_scale, drot / rot_scale, [a[6]]])
                    robot.control(controller_type=CONTROLLER_TYPE, action=cmd, controller_cfg=controller_cfg)
                    robot.gripper_control(float(a[6]))
            except Exception as exc:  # noqa: BLE001
                cprint(f"[rm_bc] control_loop exception: {exc!r}", "red")
                control_failed[0] = True
                SHUTDOWN = True
                break

    control_thread = threading.Thread(target=control_loop, daemon=True)
    control_thread.start()

    obs_dump: list[dict] | None = [] if cli.dump_obs else None
    # Debounce state: the gripper command actually in force, and how many consecutive steps the
    # policy has been asking for the opposite. Lists so the inference loop can mutate them.
    grasp_state = [False]
    grasp_pending = [0]
    # Auto-grasp-on-stall state: recent eef positions, and the step the override fired.
    eef_hist: list[np.ndarray] = []
    auto_closed_at: list[int | None] = [None]
    # Auto-release state: its own position history (the release window need not match the grasp
    # one), how long the gripper has been closed, and the step the override fired.
    rel_hist: list[np.ndarray] = []
    closed_since: list[int | None] = [None]
    auto_opened_at: list[int | None] = [None]
    # Pre-grasp tilt state: whether the grasp was already commanded last step (so the tilt fires on
    # the edge, not on every step it stays closed) and how many tilt steps are left to play out.
    was_close = [False]
    pregrasp_left = [0]
    hashes = {c: 0 for c in cli.camera_ids}
    stale = {c: 0 for c in cli.camera_ids}
    exit_code = 0
    step = 0
    cprint("Running policy 🏁   (ENTER = reset,  Ctrl-C = stop)", "yellow")

    try:
        while rclpy.ok() and not SHUTDOWN and step < cli.max_steps:
            rgbs = {c: cams.latest_rgb(c) for c in cli.camera_ids}
            if any(v is None for v in rgbs.values()):
                INFERENCE_RATE.sleep(); continue
            for c, img in rgbs.items():
                h = np_array_to_hash(img)
                stale[c] = stale[c] + 1 if h == hashes[c] else 0
                hashes[c] = h
                if stale[c] == 10:
                    raise RuntimeError(f"Camera {c} hasn't changed for 10 frames")

            eef = robot.last_eef_pose
            q_g = robot.last_gripper_q
            if eef is None or q_g is None:
                INFERENCE_RATE.sleep(); continue
            eef = np.asarray(eef)
            pos = eef[:3, 3].astype(np.float32)
            if np.any(pos < ws_min) or np.any(pos > ws_max):
                raise RuntimeError(f"EEF {pos} left the workspace box -- stopping")

            from deoxys.utils import transform_utils
            quat = np.asarray(transform_utils.mat2quat(eef[:3, :3])).reshape(-1).astype(np.float32)
            f = float(q_g) / 2.0

            obs = {CAMERAS[c][1]: to_policy_image(rgbs[c]) for c in cli.camera_ids}
            obs["robot0_eef_pos"] = pos
            obs["robot0_eef_quat"] = quat
            obs["robot0_gripper_qpos"] = np.array([f, -f], dtype=np.float32)

            a = policy(obs)
            a[:6] *= cli.action_gain
            dp = np.linalg.norm(a[:3]); dr = np.linalg.norm(a[3:6])
            if dp > cli.max_dpos:
                a[:3] *= cli.max_dpos / dp
            if dr > cli.max_drot:
                a[3:6] *= cli.max_drot / dr
            raw_grasp = float(a[6])
            want_close = raw_grasp > cli.grasp_threshold
            if cli.grasp_debounce > 0:
                if want_close == grasp_state[0]:
                    grasp_pending[0] = 0
                else:
                    grasp_pending[0] += 1
                    if grasp_pending[0] >= cli.grasp_debounce:
                        grasp_state[0] = want_close
                        grasp_pending[0] = 0
                want_close = grasp_state[0]
            if cli.auto_grasp_on_stall:
                eef_hist.append(pos.copy())
                if len(eef_hist) > cli.stall_window + 1:
                    eef_hist.pop(0)
                if (auto_closed_at[0] is None and len(eef_hist) > cli.stall_window
                        and pos[2] < cli.stall_z_max):
                    moved = float(np.linalg.norm(pos - eef_hist[0])) * 1000.0
                    if moved < cli.stall_move_mm:
                        auto_closed_at[0] = step
                        cprint(f"[auto-grasp] stalled at {np.round(pos, 4)} "
                               f"({moved:.2f} mm over {cli.stall_window} steps) -> closing", "yellow")
                if auto_closed_at[0] is not None and step - auto_closed_at[0] < cli.auto_grasp_hold:
                    want_close = True
            if cli.auto_release_on_stall:
                closed_since[0] = (closed_since[0] if want_close and closed_since[0] is not None
                                   else (step if want_close else None))
                rel_hist.append(pos.copy())
                if len(rel_hist) > cli.release_window + 1:
                    rel_hist.pop(0)
                held = 0 if closed_since[0] is None else step - closed_since[0]
                if (auto_opened_at[0] is None and want_close and held >= cli.release_min_closed
                        and len(rel_hist) > cli.release_window):
                    moved = float(np.linalg.norm(pos - rel_hist[0])) * 1000.0
                    if moved < cli.release_move_mm:
                        auto_opened_at[0] = step
                        cprint(f"[auto-release] stalled at {np.round(pos, 4)} "
                               f"({moved:.2f} mm over {cli.release_window} steps) after {held} "
                               f"closed steps -> opening", "yellow")
                if auto_opened_at[0] is not None and step - auto_opened_at[0] < cli.auto_release_hold:
                    want_close = False
            fired = want_close and not was_close[0]
            was_close[0] = want_close  # the intent, not the value the tilt overrides below
            if cli.pregrasp_wrist_deg and fired and pregrasp_left[0] == 0:
                pregrasp_left[0] = cli.pregrasp_wrist_steps
                cprint(f"[pregrasp] wrist {cli.pregrasp_wrist_deg:+.1f} deg over "
                       f"{cli.pregrasp_wrist_steps} steps, then closing", "yellow")
            if pregrasp_left[0] > 0:
                # Hold position, rotate about the hand's wrist axis, keep the fingers open. The
                # hand pivots about joint 6, ~23 cm behind the fingertips, so the TCP swings about
                # 4 mm per degree -- this is a tilt into the grasp, not a rotation in place.
                a = np.zeros(7, dtype=np.float32)
                a[3:6] = WRIST_PITCH_AXIS_HAND * (float(np.radians(cli.pregrasp_wrist_deg))
                                                  / cli.pregrasp_wrist_steps)
                want_close = False
                pregrasp_left[0] -= 1
            a[6] = 1.0 if want_close else -1.0
            eef_rot_holder[0] = eef[:3, :3].copy()
            action_holder[0] = a

            if step % 5 == 0 or not cli.execute:
                cprint(f"[{step:4d}] dpos {a[:3]*1000} mm  drot {a[3:6]} rad  "
                       f"grasp {a[6]:+.0f} (raw {raw_grasp:+.2f})"
                       f"   eef {pos}   {'CLAMPED' if dp > cli.max_dpos or dr > cli.max_drot else ''}",
                       "cyan" if cli.execute else "white")
            if obs_dump is not None:
                blank = np.zeros((3, 84, 84), dtype=np.uint8)
                img = lambda k: ((obs[k] * 255).astype(np.uint8) if k in obs else blank)
                obs_dump.append(dict(
                    step=np.int32(step),
                    agentview_image=img("agentview_image"),
                    robot0_eye_in_hand_image=img("robot0_eye_in_hand_image"),
                    robot0_eef_pos=pos.astype(np.float32),
                    robot0_eef_quat=quat.astype(np.float32),
                    robot0_gripper_qpos=np.array([f, -f], dtype=np.float32),
                    action=a.astype(np.float32),
                    raw_grasp=np.float32(raw_grasp)))
            step += 1

            if enter_is_pressed():
                cprint(" --> Resetting", "yellow")
                action_holder[0] = None
                if cli.execute:
                    reset_joints_to(robot, reset_pose, gripper_open=True)
                policy.reset()
                step = 0
                cprint("Press ENTER to continue", "yellow")
                while not enter_is_pressed():
                    INFERENCE_RATE.sleep()
                cprint("Running policy 🏁", "yellow")
                continue

            INFERENCE_RATE.sleep()

        if step >= cli.max_steps:
            cprint(f"Reached --max-steps ({cli.max_steps}), stopping.", "yellow")
    except KeyboardInterrupt:
        cprint("\n⏹️  Interrupted", "yellow")
    except Exception as exc:  # noqa: BLE001
        cprint(f"❌ {exc!r}", "red")
        exit_code = 1
    finally:
        cprint("Shutting down", "red")
        if obs_dump:
            # Written here, not per step: the rollout is the thing being measured and file I/O inside
            # the 14.4 Hz loop would perturb it.
            np.savez_compressed(cli.dump_obs,
                                **{k: np.stack([d[k] for d in obs_dump]) for k in obs_dump[0]})
            cprint(f"wrote {len(obs_dump)} observations to {cli.dump_obs}", "green")
        SHUTDOWN = True
        if control_failed[0]:
            cprint("❌ Control thread died - the arm was left holding its last target.", "red")
            exit_code = 1
        try:
            control_thread.join(timeout=2.0)
        except Exception:  # noqa: BLE001
            pass
        try:
            robot.close()
        except Exception:  # noqa: BLE001
            pass
        spin_stop.set()
        cams.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
