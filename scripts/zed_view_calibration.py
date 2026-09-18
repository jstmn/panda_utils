#!/usr/bin/env python3
"""Check the live ZED views against the views the ACT-CLIP policy was trained on.

WHY THIS EXISTS
    act_clip_inference2.py runs an end-to-end pixels->actions policy. Nothing in it
    verifies that the cameras are still pointing where they pointed during data
    collection. A camera that got nudged between 2026-07-29 (recording) and today does
    not raise an error - it silently shifts the whole input distribution and the policy
    degrades in ways that look like a bad checkpoint.

    This tool measures that drift and helps you physically re-aim a camera until it
    matches. It does NOT calibrate intrinsics and does NOT solve hand-eye extrinsics;
    the policy needs neither.

HOW IT MEASURES
    Coarse-to-fine ECC (intensity alignment) between the live frame and a reference frame
    pulled from the training HDF5s, reported as the median displacement of the image
    corners in pixels - a number that stays meaningful whether the camera was translated,
    rotated or knocked off-axis. Measured on this rig's own frames it is accurate to
    ~0.03 px and captures displacements up to ~90 px on the textured views and ~200 px on
    zed2i. Beyond that it stops converging and SIFT + RANSAC homography takes over as a
    coarse fallback, flagged as such in the output.

    ECC rather than feature matching, because the wrist camera looks at a near-featureless
    white table and starves SIFT to ~10 inliers, which produced a garbage 785 px estimate
    from a degenerate homography.

    Thresholds are not hardcoded guesses. `--baseline` measures how much the alignment
    wobbles between demos in which the camera was definitionally untouched, and that
    per-camera noise floor is what live drift is judged against. Currently: 0.2 px median
    for zed2i, 0.5 px for zed, 2.0 px for zedm.

THE MOVING-CONTENT CONFOUND
    ECC is a global fit, so anything large that moves between the two frames drags it.
    Two instances of that on this rig, both handled:

      * The task object is re-positioned every demo and is the only strong texture in the
        wrist view - full-frame alignment tracked the peg, not the camera, reading up to
        74 px of phantom drift and failing on 3 of 9 demos. Fixed by registering that
        camera on a band of the image containing the static hardware (see DEFAULT_ROI);
        that gives 9/9 alignments with a 3 px worst case. Masking the object out by
        colour instead does not work - it removes the only gradient in the frame.
      * The arm fills much of the `zed` view, so a reference frame at a different joint
        configuration reads as ~25 px of phantom drift. Fixed by reading the live joint
        state from deoxys and picking the reference frame nearest to it, for every
        camera. Without deoxys the tool says so and falls back to a fixed frame, which is
        only fair if the arm is sitting at the demo start pose.

STATIC vs WRIST-MOUNTED CAMERAS
    Measured from data/lift_peg_0729: `zed` and `zed2i` are static, `zedm` is
    wrist-mounted (424 px of view motion by frame 51, then no match at all). A wrist
    camera's view is a function of the joint configuration, so without a live robot pose
    it is skipped rather than guessed at. `--classify` re-derives this from the data
    rather than trusting the note above.

USAGE
    source /opt/ros/humble/setup.bash
    source ~/Desktop/ros2_ws/install/setup.bash
    source ~/anaconda3/bin/activate act

    # what counts as "untouched" for each camera - run this once, no cameras needed
    python scripts/zed_view_calibration.py --baseline

    # which cameras are static and which move with the arm - no cameras needed
    python scripts/zed_view_calibration.py --classify

    # one-shot verdict + annotated PNG (cameras must be streaming)
    python scripts/zed_view_calibration.py --report

    # live overlay for physically nudging a camera back into place
    python scripts/zed_view_calibration.py --live --camera zed2i

Cameras must already be up:  bash ~/Desktop/ros2_ws/zed_tools/launch_all.sh
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import threading
import time
from pathlib import Path

import cv2
import h5py
import numpy as np
from termcolor import cprint

# --------------------------------------------------------------------------------------
# Conventions shared with act_clip_inference2.py / log_demonstration2.py
# --------------------------------------------------------------------------------------

DEFAULT_TOPIC_TEMPLATE = "/zed_multi/{cam}"
COLOR_SUFFIX = "rgb/image_rect_color"
DEFAULT_CAMERAS = ["zed", "zed2i", "zedm"]
DEFAULT_DATA_ROOT = "/home/res1/Desktop/data"
DEFAULT_DATASET = "raise_cube_0729"

# A camera whose within-demo view displacement exceeds this is moving with the arm.
EIH_CLASSIFY_PX = 50.0

# Live drift is judged against the per-camera baseline: OK below WARN_MULT * baseline,
# FAIL above FAIL_MULT * baseline, with absolute floors so a camera with a tiny baseline
# does not fail on sub-pixel noise.
WARN_MULT, FAIL_MULT = 3.0, 6.0
WARN_FLOOR_PX, FAIL_FLOOR_PX = 6.0, 15.0


# --------------------------------------------------------------------------------------
# Feature alignment
# --------------------------------------------------------------------------------------

_sift = cv2.SIFT_create(nfeatures=4000)
_matcher = cv2.BFMatcher()

# Coarse-to-fine ECC. The coarse levels buy capture range, the last level buys sub-pixel
# accuracy. Measured on this rig's own frames: exact to ~0.03 px, and it recovers a 200 px
# shift that single-scale ECC diverges on.
ECC_LEVELS = (8, 4, 1)
ECC_MIN_CC = 0.30          # below this the intensity fit did not really lock on
SIFT_MIN_INLIERS = 25      # below this a RANSAC homography goes degenerate (seen: 785 px
                           # from 11 inliers on the low-texture wrist view)

# Vertical band (top, bottom as fractions of height) each camera is registered on.
#
# The wrist view is mostly blank white table; its only strong texture is the task object,
# which is deliberately re-positioned every demo. Aligning on the full frame therefore
# tracks the OBJECT, not the camera - measured on lift_peg_0729 that reads as up to 74 px
# of "drift" with the camera untouched, and fails outright on 3 of 9 demos. Restricting to
# the bottom band, where the static hardware sits, gives 9/9 alignments with a 2.97 px
# median and 4.78 px worst case, and still recovers a synthetic 30 px shift to 0.04 px.
#
# Masking the object out by colour instead does NOT work: it strips the only gradient in
# the frame and ECC diverges to hundreds of px.
#
# The two fixed cameras see plenty of static structure and are registered full-frame.
DEFAULT_ROI = {"zedm": (0.75, 1.0)}


def _corner_disp(warp_2x3_or_3x3, w, h) -> np.ndarray:
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    if warp_2x3_or_3x3.shape[0] == 2:
        moved = cv2.transform(corners, warp_2x3_or_3x3)
    else:
        moved = cv2.perspectiveTransform(corners, warp_2x3_or_3x3)
    return np.linalg.norm(moved - corners, axis=2).ravel()


def roi_mask(shape, roi) -> np.ndarray | None:
    """uint8 mask selecting a vertical band, or None for the whole frame."""
    if roi is None:
        return None
    h, w = shape[:2]
    top, bot = roi
    m = np.zeros((h, w), np.uint8)
    m[int(top * h):int(bot * h), :] = 255
    return m


def _ecc(g_ref, g_live, mask=None):
    """Euclidean ECC, coarse to fine. Returns (cc, 2x3 warp) or None."""
    def prep(g, ds):
        h, w = g.shape
        s = (cv2.resize(g, (w // ds, h // ds), interpolation=cv2.INTER_AREA)
             if ds > 1 else g)
        return cv2.GaussianBlur(s, (5, 5), 0).astype(np.float32)

    def prep_mask(m, ds):
        if m is None or ds == 1:
            return m
        h, w = m.shape
        return cv2.resize(m, (w // ds, h // ds), interpolation=cv2.INTER_NEAREST)

    warp, cc = np.eye(2, 3, dtype=np.float32), 0.0
    for ds in ECC_LEVELS:
        w = warp.copy()
        w[0, 2] /= ds
        w[1, 2] /= ds
        try:
            cc, w = cv2.findTransformECC(
                prep(g_ref, ds), prep(g_live, ds), w, cv2.MOTION_EUCLIDEAN,
                (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 300, 1e-7),
                prep_mask(mask, ds), 5)
        except cv2.error:
            return None
        w[0, 2] *= ds
        w[1, 2] *= ds
        warp = w
    return (cc, warp) if cc >= ECC_MIN_CC else None


def _sift_homography(g_ref, g_live, mask=None):
    """RANSAC homography from SIFT matches. Returns (H, inliers) or None.

    Only trusted above SIFT_MIN_INLIERS: this is the fallback for displacements outside
    ECC's capture range, where being approximately right still beats reporting nothing.
    """
    k_ref, d_ref = _sift.detectAndCompute(g_ref, mask)
    k_live, d_live = _sift.detectAndCompute(g_live, mask)
    if d_ref is None or d_live is None or len(k_ref) < 10 or len(k_live) < 10:
        return None
    matches = _matcher.knnMatch(d_ref, d_live, k=2)
    good = [a for a, b in matches if a.distance < 0.75 * b.distance]
    if len(good) < 12:
        return None
    p_ref = np.float32([k_ref[g.queryIdx].pt for g in good]).reshape(-1, 1, 2)
    p_live = np.float32([k_live[g.trainIdx].pt for g in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(p_ref, p_live, cv2.RANSAC, 3.0)
    if H is None or mask is None:
        return None
    n = int(mask.sum())
    return (H, n) if n >= SIFT_MIN_INLIERS else None


def estimate_drift(ref_rgb: np.ndarray, live_rgb: np.ndarray,
                   roi=None, fast: bool = False) -> dict | None:
    """Apparent view drift from ref -> live, as median image-corner displacement in px.

    ECC first: it is sub-pixel accurate and, unlike feature matching, does not need
    texture - which matters because the wrist camera stares at a near-featureless white
    table and starves SIFT down to ~10 inliers. SIFT takes over only when ECC fails to
    converge, i.e. the camera moved further than ECC's basin (>~90 px on this rig).

    `roi` is a (top, bottom) pair of height fractions restricting both methods to a band
    of the image; see DEFAULT_ROI for why the wrist camera needs one.

    `fast`: half-res ECC only (no SIFT). Used by --live so a hard-to-align view like
    zed2i cannot freeze the overlay via multi-second SIFT on the GIL.

    Returns None when neither method can relate the frames. For a wrist camera that
    usually means the arm is in a different configuration, not that the camera moved.
    """
    if ref_rgb.shape[:2] != live_rgb.shape[:2]:
        live_rgb = cv2.resize(live_rgb, (ref_rgb.shape[1], ref_rgb.shape[0]),
                              interpolation=cv2.INTER_AREA)

    scale = 1.0
    if fast:
        # Half-res is enough for a live overlay number and ~4x cheaper.
        live_rgb = cv2.resize(live_rgb, (live_rgb.shape[1] // 2, live_rgb.shape[0] // 2),
                              interpolation=cv2.INTER_AREA)
        ref_rgb = cv2.resize(ref_rgb, (live_rgb.shape[1], live_rgb.shape[0]),
                             interpolation=cv2.INTER_AREA)
        scale = 2.0

    g_ref = cv2.cvtColor(ref_rgb, cv2.COLOR_RGB2GRAY)
    g_live = cv2.cvtColor(live_rgb, cv2.COLOR_RGB2GRAY)
    h, w = g_ref.shape
    mask = roi_mask(g_ref.shape, roi)

    r = _ecc(g_ref, g_live, mask)
    if r is not None:
        cc, warp = r
        disp = _corner_disp(warp, w, h) * scale
        return dict(
            method="ecc", confidence=float(cc),
            median_px=float(np.median(disp)), max_px=float(disp.max()),
            dx=float(warp[0, 2] * scale), dy=float(warp[1, 2] * scale),
            rot_deg=float(np.degrees(np.arctan2(warp[1, 0], warp[0, 0]))),
            scale=1.0, inliers=None, warp=warp,
        )

    if fast:
        return None

    s = _sift_homography(g_ref, g_live, mask)
    if s is not None:
        H, n = s
        disp = _corner_disp(H, w, h)
        return dict(
            method="sift", confidence=float(min(1.0, n / 100.0)),
            median_px=float(np.median(disp)), max_px=float(disp.max()),
            dx=float(H[0, 2]), dy=float(H[1, 2]), rot_deg=float("nan"),
            scale=1.0, inliers=n, warp=H,
        )
    return None


# --------------------------------------------------------------------------------------
# Reference frames from the training HDF5s
# --------------------------------------------------------------------------------------

def roi_for(cam: str, args) -> tuple[float, float] | None:
    """Registration band for a camera: --roi override, else the measured default."""
    for spec in (args.roi or []):
        name, _, bounds = spec.partition("=")
        if name == cam:
            if bounds.lower() in ("full", "none", ""):
                return None
            top, _, bot = bounds.partition(",")
            return (float(top), float(bot))
    return DEFAULT_ROI.get(cam)


def dataset_dir(root: str, name: str) -> Path:
    p = Path(root) / name
    if not p.is_dir():
        raise SystemExit(f"no such dataset dir: {p}")
    return p


def traj_files(d: Path) -> list[Path]:
    fs = sorted(d.glob("traj_*.h5"), key=lambda p: int(p.stem.split("_")[1]))
    if not fs:
        raise SystemExit(f"no traj_*.h5 under {d}")
    return fs


def read_frame(h5path: Path, cam: str, idx: int) -> np.ndarray:
    with h5py.File(h5path, "r") as h:
        key = f"{cam}__color_image"
        if key not in h:
            raise SystemExit(f"{h5path.name} has no '{key}' "
                             f"(has: {[k for k in h.keys()]})")
        n = h[key].shape[0]
        return np.asarray(h[key][min(idx, n - 1)])


def read_joints(h5path: Path) -> np.ndarray | None:
    with h5py.File(h5path, "r") as h:
        return np.asarray(h["joint_states_q"][:]) if "joint_states_q" in h else None


def nearest_pose_frame(h5path: Path, q_now: np.ndarray) -> tuple[int, float]:
    """Index of the frame whose arm configuration is closest to q_now, and the L2 error."""
    q = read_joints(h5path)
    if q is None:
        return 0, float("inf")
    d = np.linalg.norm(q[:, :7] - np.asarray(q_now)[:7], axis=1)
    i = int(np.argmin(d))
    return i, float(d[i])


# --------------------------------------------------------------------------------------
# Live ROS 2 frames
# --------------------------------------------------------------------------------------

def decode_color(msg) -> np.ndarray:
    """ZED colour Image -> HxWx3 uint8 RGB. Same convention as log_demonstration2."""
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


class LiveCameras:
    """Minimal rclpy subscriber over the same topics act_clip_inference2.py uses."""

    def __init__(self, cameras, topic_template, best_effort):
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                               ReliabilityPolicy)
        from sensor_msgs.msg import Image

        if not rclpy.ok():
            rclpy.init()
        self._rclpy = rclpy
        self._node = Node("zed_view_calibration")
        self._latest = {c: None for c in cameras}
        self._cameras = list(cameras)
        self._lock = threading.Lock()

        qos = QoSProfile(
            reliability=(ReliabilityPolicy.BEST_EFFORT if best_effort
                         else ReliabilityPolicy.RELIABLE),
            history=HistoryPolicy.KEEP_LAST, depth=5,
            durability=DurabilityPolicy.VOLATILE,
        )
        for cam in cameras:
            topic = f"{topic_template.format(cam=cam).rstrip('/')}/{COLOR_SUFFIX}"
            self._node.create_subscription(
                Image, topic, lambda m, c=cam: self._cb(m, c), qos)
            cprint(f"   subscribed: {topic}", "white")

        self._spin = threading.Thread(target=self._spin_loop, daemon=True)
        self._stop = threading.Event()
        self._spin.start()

    def _spin_loop(self):
        while not self._stop.is_set() and self._rclpy.ok():
            self._rclpy.spin_once(self._node, timeout_sec=0.1)

    def _cb(self, msg, cam):
        try:
            img = decode_color(msg)
        except ValueError as exc:
            self._node.get_logger().warn(f"{cam}: {exc}")
            return
        with self._lock:
            self._latest[cam] = img

    def latest(self, cam):
        with self._lock:
            return self._latest[cam]

    def wait(self, timeout_s=20.0) -> bool:
        cprint(f"⏳ Waiting for frames from {self._cameras} ...", "cyan")
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if all(self.latest(c) is not None for c in self._cameras):
                cprint("✅ All cameras streaming", "green")
                return True
            time.sleep(0.2)
        missing = [c for c in self._cameras if self.latest(c) is None]
        cprint(f"❌ No frames from {missing} after {timeout_s:.0f}s", "red")
        cprint("   Start them:  bash ~/Desktop/ros2_ws/zed_tools/launch_all.sh", "yellow")
        cprint("   If topics exist but no data arrives, retry with --qos-best-effort",
               "yellow")
        return False

    def close(self):
        self._stop.set()
        self._spin.join(timeout=2.0)
        self._node.destroy_node()


def current_joints():
    """Current arm configuration from deoxys, or None if the robot is not reachable."""
    try:
        from deoxys.franka_interface import FrankaInterface
    except Exception:
        return None
    cfg = Path(__file__).resolve().parent.parent / "configs" / "charmander.yml"
    if not cfg.is_file():
        return None
    try:
        robot = FrankaInterface(str(cfg), use_visualizer=False, control_freq=20)
        deadline = time.time() + 5.0
        while time.time() < deadline:
            if robot.last_q is not None:
                return np.asarray(robot.last_q)
            time.sleep(0.1)
    except Exception:
        return None
    return None


# --------------------------------------------------------------------------------------
# Modes
# --------------------------------------------------------------------------------------

def mode_classify(args) -> int:
    """Which cameras are static and which move with the arm - measured, not assumed."""
    d = dataset_dir(args.data_root, args.dataset)
    f0 = traj_files(d)[0]
    cprint(f"\nClassifying cameras from {f0}", "cyan")
    with h5py.File(f0, "r") as h:
        cams = [k.replace("__color_image", "") for k in h.keys()
                if k.endswith("__color_image")]
        n = h[f"{cams[0]}__color_image"].shape[0]

    q = read_joints(f0)
    if q is not None:
        cprint(f"  arm moves {np.abs(q - q[0]).max(axis=0)[:7].round(2)} rad over the demo",
               "white")

    print(f"\n  {'camera':8s} {'drift vs frame 0':>32s}   verdict")
    print("  " + "-" * 62)
    verdicts = {}
    for cam in cams:
        ref = read_frame(f0, cam, 0)
        cells, worst = [], 0.0
        for fi in (n // 4, n // 2, n - 1):
            # Deliberately full-frame, no ROI: the wrist camera's registration band is
            # hardware that moves WITH it, so an ROI there would show no motion and
            # misclassify a wrist camera as static.
            r = estimate_drift(ref, read_frame(f0, cam, fi))
            if r is None:
                cells.append(f"f{fi}:no-match")
                worst = max(worst, 1e9)
            else:
                cells.append(f"f{fi}:{r['median_px']:.0f}px")
                worst = max(worst, r["median_px"])
        static = worst < EIH_CLASSIFY_PX
        verdicts[cam] = "static" if static else "wrist-mounted"
        colour = "green" if static else "yellow"
        cprint(f"  {cam:8s} {'  '.join(cells):>32s}   {verdicts[cam]}", colour)

    cprint("\n  Static cameras can be checked against any reference frame.", "white")
    cprint("  Wrist-mounted cameras need the arm at a matching configuration,", "white")
    cprint("  so --report reads the live joint state and picks the nearest frame.", "white")
    return 0


def mode_baseline(args) -> int:
    """Per-camera noise floor: drift between demos where the camera was untouched."""
    d = dataset_dir(args.data_root, args.dataset)
    files = traj_files(d)[: args.baseline_trajs]
    if len(files) < 2:
        raise SystemExit("need at least 2 trajectories for a baseline")

    cprint(f"\nBaseline from {len(files)} demos in {d.name} "
           f"(camera untouched by definition)", "cyan")
    with h5py.File(files[0], "r") as h:
        cams = [k.replace("__color_image", "") for k in h.keys()
                if k.endswith("__color_image")]

    print(f"\n  {'camera':8s} {'median':>9s} {'p90':>9s} {'worst':>9s} {'n':>4s} {'conf':>6s}"
          f"   {'warn above':>11s} {'fail above':>11s}")
    print("  " + "-" * 78)
    out = {}
    for cam in cams:
        ref = read_frame(files[0], cam, 0)
        vals, confs = [], []
        for f in files[1:]:
            r = estimate_drift(ref, read_frame(f, cam, 0), roi_for(cam, args))
            if r:
                vals.append(r["median_px"])
                confs.append(r["confidence"])
        if not vals:
            cprint(f"  {cam:8s} {'could not align any pair - cannot baseline':>60s}", "red")
            continue
        med, p90, worst = float(np.median(vals)), float(np.percentile(vals, 90)), max(vals)
        warn = max(WARN_MULT * p90, WARN_FLOOR_PX)
        fail = max(FAIL_MULT * p90, FAIL_FLOOR_PX)
        out[cam] = dict(baseline_p90=p90, warn=warn, fail=fail)
        print(f"  {cam:8s} {med:8.2f}p {p90:8.2f}p {worst:8.2f}p {len(vals):4d} "
              f"{np.mean(confs):6.3f}   {warn:10.1f}p {fail:10.1f}p")

    path = Path(args.baseline_file)
    path.write_text(repr(out))
    cprint(f"\n  written: {path}", "green")
    cprint("  --report reads this file; re-run --baseline if you re-mount a camera "
           "on purpose.", "white")
    return 0


def load_baseline(args) -> dict:
    p = Path(args.baseline_file)
    if not p.is_file():
        cprint(f"⚠️  no baseline at {p} - falling back to absolute thresholds "
               f"({WARN_FLOOR_PX:.0f}px warn / {FAIL_FLOOR_PX:.0f}px fail). "
               f"Run --baseline for per-camera limits.", "yellow")
        return {}
    try:
        return eval(p.read_text())  # written by this script, plain dict of floats
    except Exception:
        cprint(f"⚠️  could not parse {p}; using absolute thresholds", "yellow")
        return {}


def verdict_for(cam, median_px, baseline) -> tuple[str, str]:
    b = baseline.get(cam)
    warn = b["warn"] if b else WARN_FLOOR_PX
    fail = b["fail"] if b else FAIL_FLOOR_PX
    if median_px >= fail:
        return "FAIL", "red"
    if median_px >= warn:
        return "WARN", "yellow"
    return "OK", "green"


def annotate(ref_rgb, live_rgb, result, cam, verdict) -> np.ndarray:
    """live | reference | blend strip, labelled."""
    if live_rgb.shape[:2] != ref_rgb.shape[:2]:
        live_rgb = cv2.resize(live_rgb, (ref_rgb.shape[1], ref_rgb.shape[0]),
                              interpolation=cv2.INTER_AREA)
    blend = cv2.addWeighted(ref_rgb, 0.5, live_rgb, 0.5, 0)
    strip = np.hstack([live_rgb, ref_rgb, blend])
    strip = cv2.cvtColor(strip, cv2.COLOR_RGB2BGR)

    h = ref_rgb.shape[0]
    scale = max(0.6, h / 720.0)
    colour = {"OK": (0, 220, 0), "WARN": (0, 200, 255), "FAIL": (0, 0, 255)}[verdict]
    txt = (f"{cam}  {verdict}  {result['median_px']:.1f}px"
           if result else f"{cam}  {verdict}  no match")
    cv2.putText(strip, txt, (12, int(34 * scale)), cv2.FONT_HERSHEY_SIMPLEX,
                scale, colour, 2, cv2.LINE_AA)
    for i, label in enumerate(("LIVE", "TRAINING REF", "BLEND")):
        cv2.putText(strip, label, (12 + i * ref_rgb.shape[1], h - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale, (255, 255, 255), 2, cv2.LINE_AA)
    return strip


def mode_report(args) -> int:
    d = dataset_dir(args.data_root, args.dataset)
    files = traj_files(d)
    ref_file = files[min(args.ref_traj, len(files) - 1)]
    baseline = load_baseline(args)

    cams = args.cameras
    live = LiveCameras(cams, args.topic_template, args.qos_best_effort)
    try:
        if not live.wait(args.timeout):
            return 1

        # Every camera wants a pose-matched reference, not just the wrist ones. ECC is a
        # global intensity fit, so the arm - which fills a good part of the `zed` view -
        # drags the alignment when it sits at a different configuration than the
        # reference: measured at 25 px of phantom drift on an untouched camera. Picking
        # the reference frame nearest the live joint state removes that confound.
        eih = set(args.eih_cameras)
        q_now = current_joints()
        if q_now is None:
            cprint("\n⚠️  deoxys not reachable - falling back to reference frame "
                   f"{args.ref_frame} for every camera. That is only a fair comparison "
                   "if the arm is currently at the demo start pose.", "yellow")
            if eih and not args.allow_poseless_eih:
                cprint(f"   {sorted(eih)} will be SKIPPED (a wrist view is meaningless "
                       "without a matching arm pose). Pass --allow-poseless-eih to "
                       "compare anyway.", "yellow")
        else:
            cprint(f"\n🔗 arm pose read from deoxys; matching reference frames to it.",
                   "cyan")

        cprint(f"\nReference: {ref_file.name} from {d.name}", "cyan")
        print(f"\n  {'camera':8s} {'drift':>9s} {'shift(dx,dy)':>16s} {'rot':>8s} "
              f"{'via':>5s} {'conf':>6s}   verdict")
        print("  " + "-" * 76)

        strips, worst = [], "OK"
        for cam in cams:
            live_img = live.latest(cam)
            frame_idx, pose_err = args.ref_frame, 0.0

            if cam in eih and q_now is None and not args.allow_poseless_eih:
                cprint(f"  {cam:8s} {'skipped (no robot pose)':>50s}", "yellow")
                continue
            if q_now is not None:
                frame_idx, pose_err = nearest_pose_frame(ref_file, q_now)

            ref_img = read_frame(ref_file, cam, frame_idx)
            r = estimate_drift(ref_img, live_img, roi_for(cam, args))

            if r is None:
                v, colour = "FAIL", "red"
                cprint(f"  {cam:8s} {'no match':>9s} {'':>16s} {'':>8s} {'':>5s} "
                       f"{'':>6s}   {v}", colour)
                cprint(f"  {'':8s} could not align at all - either the camera moved a "
                       f"long way, or the view is too featureless to register.", "red")
                worst = "FAIL"
                strips.append(annotate(ref_img, live_img, r, cam, v))
                continue

            v, colour = verdict_for(cam, r["median_px"], baseline)
            if v == "FAIL" or (v == "WARN" and worst == "OK"):
                worst = v
            shift = f"{r['dx']:+.0f},{r['dy']:+.0f}"
            rot = ("    n/a" if np.isnan(r["rot_deg"]) else f"{r['rot_deg']:+7.2f}")
            cprint(f"  {cam:8s} {r['median_px']:8.1f}p {shift:>16s} "
                   f"{rot}° {r['method']:>5s} {r['confidence']:6.3f}   {v}", colour)
            if r["method"] == "sift":
                cprint(f"  {'':8s} fell back to feature matching ({r['inliers']} inliers) "
                       f"- intensity alignment did not converge, so treat the magnitude "
                       f"as approximate.", "yellow")
            if q_now is not None and pose_err > args.pose_tol:
                cprint(f"  {'':8s} nearest reference pose is {pose_err:.3f} rad away "
                       f"(frame {frame_idx}) - that gap alone can explain the drift "
                       f"above. Move the arm to the demo start pose and re-run.",
                       "yellow")
            strips.append(annotate(ref_img, live_img, r, cam, v))

        if strips:
            width = max(s.shape[1] for s in strips)
            padded = [cv2.copyMakeBorder(s, 0, 0, 0, width - s.shape[1],
                                         cv2.BORDER_CONSTANT, value=(0, 0, 0))
                      for s in strips]
            out = Path(args.out)
            cv2.imwrite(str(out), np.vstack(padded))
            cprint(f"\n  diagnostic image: {out}", "green")

        cprint(f"\n  OVERALL: {worst}",
               {"OK": "green", "WARN": "yellow", "FAIL": "red"}[worst])
        if worst != "OK":
            cprint("  Re-aim the flagged camera with:  "
                   "python scripts/zed_view_calibration.py --live --camera <name>", "white")
        return 0 if worst != "FAIL" else 2
    finally:
        live.close()


def mode_live(args) -> int:
    """Continuously updating overlay so you can nudge a camera back into place."""
    cam = args.camera
    d = dataset_dir(args.data_root, args.dataset)
    ref_file = traj_files(d)[min(args.ref_traj, len(traj_files(d)) - 1)]
    baseline = load_baseline(args)

    eih = cam in set(args.eih_cameras)
    q_now = current_joints() if eih else None
    frame_idx = args.ref_frame
    if eih and q_now is not None:
        frame_idx, err = nearest_pose_frame(ref_file, q_now)
        cprint(f"wrist camera: using frame {frame_idx} (arm pose within {err:.3f} rad)",
               "cyan")
    elif eih:
        cprint("⚠️  wrist camera but no robot pose available - the overlay will only "
               "line up if the arm happens to be at the reference pose.", "yellow")

    ref_img = read_frame(ref_file, cam, frame_idx)
    live = LiveCameras([cam], args.topic_template, args.qos_best_effort)
    try:
        if not live.wait(args.timeout):
            return 1
        cprint("\n  keys:  [b] blend   [d] difference   [e] edges-on-live   "
               "[s] save png   [q] quit", "white")
        # Distinct title + position per camera so zed2i is not hidden under a leftover
        # 'calibrate zed' window (both used to open at 40,40).
        win = f"ZED_CALIB__{cam}"
        x_off = {"zed": 40, "zed2i": 80, "zedm": 120}.get(cam, 40)
        y_off = {"zed": 40, "zed2i": 80, "zedm": 120}.get(cam, 40)
        cprint("  OpenCV opens a DESKTOP window (not inside Cursor).", "yellow")
        cprint(f"  Alt+Tab to '{win}'  (placed at +{x_off}+{y_off})", "yellow")
        view = "b"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 1280, 480)
        cv2.moveWindow(win, x_off, y_off)
        try:
            cv2.setWindowProperty(win, cv2.WND_PROP_TOPMOST, 1)
        except cv2.error:
            pass
        # Live drift must stay cheap: full-res ECC+SIFT on zed2i can take seconds and
        # stalls the UI via the GIL even in a worker thread (zed often ECC-succeeds
        # quickly, which is why --camera zed looked fine while zed2i felt broken).
        est_lock = threading.Lock()
        est_state = {"cached": None, "busy": False, "last_started": 0.0}
        est_period = max(args.est_period, 1.0)
        preview_path = Path(f"/tmp/zed_calib_live_{cam}.jpg")
        last_preview = 0.0
        n_shown = 0

        def _kick_estimate(live_snap: np.ndarray) -> None:
            with est_lock:
                if est_state["busy"]:
                    return
                if time.time() - est_state["last_started"] < est_period:
                    return
                est_state["busy"] = True
                est_state["last_started"] = time.time()

            def _worker() -> None:
                try:
                    result = estimate_drift(ref_img, live_snap, roi_for(cam, args),
                                            fast=True)
                except Exception as exc:  # noqa: BLE001 - keep the overlay alive
                    cprint(f"  drift estimate failed: {exc}", "yellow")
                    result = None
                with est_lock:
                    est_state["cached"] = result
                    est_state["busy"] = False

            threading.Thread(target=_worker, daemon=True).start()

        while True:
            live_img = live.latest(cam)
            if live_img is None:
                time.sleep(0.05)
                continue
            if live_img.shape[:2] != ref_img.shape[:2]:
                live_img = cv2.resize(live_img, (ref_img.shape[1], ref_img.shape[0]),
                                      interpolation=cv2.INTER_AREA)

            if view == "b":
                vis = cv2.addWeighted(ref_img, 0.5, live_img, 0.5, 0)
            elif view == "d":
                vis = cv2.absdiff(ref_img, live_img)
            else:
                # Reference edges drawn over the live frame: the most precise way to see
                # a few pixels of misalignment by eye.
                e = cv2.Canny(cv2.cvtColor(ref_img, cv2.COLOR_RGB2GRAY), 60, 160)
                vis = live_img.copy()
                vis[e > 0] = (255, 0, 255)

            _kick_estimate(live_img)
            with est_lock:
                r = est_state["cached"]
                busy = est_state["busy"]

            if r:
                v, _ = verdict_for(cam, r["median_px"], baseline)
                rot = "" if np.isnan(r["rot_deg"]) else f"  rot {r['rot_deg']:+.2f}deg"
                txt = (f"{v}  {r['median_px']:.1f}px   "
                       f"shift {r['dx']:+.0f},{r['dy']:+.0f}{rot}  [{r['method']}]")
                col = {"OK": (0, 220, 0), "WARN": (0, 200, 255),
                       "FAIL": (0, 0, 255)}[v]
            elif busy:
                txt, col = "estimating drift...", (200, 200, 200)
            else:
                txt, col = "live view OK (no drift lock yet)", (200, 200, 200)

            frame = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
            cv2.putText(frame, txt, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2,
                        cv2.LINE_AA)
            cv2.putText(frame, f"{cam}  frames={n_shown}", (12, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2, cv2.LINE_AA)
            cv2.imshow(win, frame)
            n_shown += 1

            now = time.time()
            if now - last_preview > 1.0:
                cv2.imwrite(str(preview_path), frame)
                last_preview = now
                if n_shown <= 3:
                    cprint(f"  preview also at {preview_path}  (open in image viewer if "
                           f"the window is hidden)", "cyan")

            k = cv2.waitKey(30) & 0xFF
            if k == ord("q"):
                break
            if k in (ord("b"), ord("d"), ord("e")):
                view = chr(k)
            if k == ord("s"):
                p = Path(args.out).with_name(f"calib_{cam}.png")
                cv2.imwrite(str(p), frame)
                cprint(f"  saved {p}", "green")
        cv2.destroyAllWindows()
        return 0
    finally:
        live.close()


# --------------------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Check live ZED views against the ACT training distribution.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--classify", action="store_true",
                    help="determine which cameras are static vs wrist-mounted (no cameras needed)")
    ap.add_argument("--baseline", action="store_true",
                    help="measure the per-camera noise floor from the training set (no cameras needed)")
    ap.add_argument("--report", action="store_true",
                    help="one-shot drift check of all cameras (default)")
    ap.add_argument("--live", action="store_true",
                    help="live overlay for re-aiming one camera")

    ap.add_argument("--camera", type=str, default="zed2i", help="camera for --live")
    ap.add_argument("--cameras", nargs="*", default=DEFAULT_CAMERAS,
                    help=f"cameras for --report (default: {' '.join(DEFAULT_CAMERAS)})")
    ap.add_argument("--eih-cameras", nargs="*", default=["zedm"],
                    help="wrist-mounted cameras needing arm-pose matching (default: zedm)")
    ap.add_argument("--allow-poseless-eih", action="store_true",
                    help="compare wrist cameras even without a live robot pose")
    ap.add_argument("--pose-tol", type=float, default=0.15,
                    help="joint L2 (rad) above which a wrist-camera result is called suspect")
    ap.add_argument("--roi", nargs="*", default=None, metavar="CAM=TOP,BOT",
                    help="registration band per camera as height fractions, e.g. "
                         "'zedm=0.75,1.0', or 'zedm=full' to use the whole frame. "
                         f"Default: {DEFAULT_ROI}")

    ap.add_argument("--data-root", type=str, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--dataset", type=str, default=DEFAULT_DATASET,
                    help=f"task dir under --data-root (default: {DEFAULT_DATASET})")
    ap.add_argument("--ref-traj", type=int, default=0, help="reference trajectory index")
    ap.add_argument("--ref-frame", type=int, default=0,
                    help="reference frame index for static cameras")
    ap.add_argument("--baseline-trajs", type=int, default=8,
                    help="how many demos to use for --baseline")
    ap.add_argument("--baseline-file", type=str,
                    default=str(Path(__file__).resolve().parent / "zed_view_baseline.txt"))

    ap.add_argument("--topic-template", type=str, default=DEFAULT_TOPIC_TEMPLATE)
    ap.add_argument("--qos-best-effort", action="store_true")
    ap.add_argument("--timeout", type=float, default=20.0)
    ap.add_argument("--est-period", type=float, default=0.3,
                    help="seconds between drift estimates in --live (the overlay itself "
                         "always runs at full frame rate)")
    ap.add_argument("--out", type=str, default="/tmp/zed_view_calibration.png")

    args = ap.parse_args()

    if args.classify:
        return mode_classify(args)
    if args.baseline:
        return mode_baseline(args)
    if args.live:
        return mode_live(args)
    return mode_report(args)


if __name__ == "__main__":
    sys.exit(main())
