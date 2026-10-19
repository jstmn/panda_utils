#!/usr/bin/env python3
"""Real-time frozen-camera detector for log_demonstration.py.

WHY TWO MODES
-------------
log_demonstration.py does NOT write images during recording. It accumulates every
frame in RAM (ZedDataCollector.camera_data[cam]["color_images"]) and only flushes to
disk in save(), which runs *after* you press ENTER to stop the demo. So the images/
folder is a post-hoc artefact, not a live stream -- polling it can never tell you a
camera froze while you are still driving the SpaceMouse.

Therefore:

  topics mode (default, real-time)
      Subscribe to the same ZED topics the logger subscribes to
      (/zed_multi/<cam>/rgb/image_rect_color) and watch frame arrival + frame content
      directly. Detects a freeze within ~2-4 s of it happening, mid-demo.
      Requires ROS 2 Humble sourced (same shell setup as the logger).

  files mode (fallback, post-hoc)
      Watch the newest demo folder's images/ and split each combined_rgb montage into
      per-camera panels. Only fires once a demo has been saved, so it tells you "demo
      N was ruined" rather than "stop now" -- useful only when ROS is not reachable
      from this shell.

  auto (default) = try topics, fall back to files if rclpy will not import.

WHAT THE OUTPUT LOOKS LIKE ON DISK (verified against ~/Desktop/data/zed_demos)
-----------------------------------------------------------------------------
    <output_dir>/<description>__<MM-DD_HH-MM-SS>/
        traj_<i>.h5                       # joint states + <cam>__color_image stacks
        <cam>_camera_info_color.json      # one per camera, 1280x720
        images/
            combined_rgb__traj_<i>__<nnn>.png    # 720 x (1280*N) BGR montage

There is one PNG per timestep, NOT one per camera: save_combined_rgbs() hstacks all
cameras into a single image, in --camera_ids order, each panel labelled in green text
at the top-left. Panels are equal width because every ZED here is 1280x720.

USAGE
-----
    source /opt/ros/humble/setup.bash
    source ~/Desktop/ros2_ws/install/setup.bash
    source ~/anaconda3/bin/activate deoxys

    python ~/Desktop/panda_utils/scripts/chk_camera_crash.py

Start it in its own terminal any time -- before or after the logger. It idles until it
sees log_demonstration.py in the process table, auto-discovers that run's
--camera_ids / --output_dir / --topic-template from /proc/<pid>/cmdline, monitors, and
goes back to idling when the logger exits. It never exits on its own so it survives
across logger restarts; Ctrl-C to stop it.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# TUNABLES -- see the "TUNING" section of the accompanying notes
# ---------------------------------------------------------------------------

# How often the monitor wakes up to evaluate camera state (seconds).
POLL_INTERVAL_S = 0.5

# (a) SILENCE DETECTION: no new frame at all for this long -> frozen.
# Cameras publish at ~30 Hz and the logger samples at 15 Hz, so 2.0 s is 30-60
# missed frames. Well past any scheduling hiccup, still fast enough to abort a demo.
FRAME_TIMEOUT_S = 2.0

# (b) DUPLICATE-FRAME DETECTION: frames keep arriving but the pixels never change.
# Mean absolute difference between consecutive frames, on a 64x36 grayscale downsample
# (0-255 scale).
#
# Calibrated against 6057 healthy real frame-pairs from logged demos on this rig:
#     healthy minimum MAD .... 0.296   (never lower, across all three cameras)
#     healthy median MAD ..... 0.72 - 1.55
#     exact-zero frames ...... 0
#     longest healthy run below 0.2 ... 0 frames
#     longest healthy run below 0.8 ... 137 frames   <-- why a loose threshold fails
#
# Then validated over 10509 real frames spanning every trajectory in three demo
# folders. It flagged 11 (camera, trajectory) pairs and every one turned out to be a
# genuine freeze already sitting in the logged data -- zero false positives. Spot
# checks by full-resolution md5 confirmed them: e.g. raise_cube__07-29_16-39-26
# traj_18 has ONE unique 'zed' frame across all 174 timesteps, and traj_17 has a
# 425-frame identical run, while zed2i and zedm were 100% unique in the same demos.
#
# So the real-world freeze signature on this rig is a byte-identical republished
# buffer: MAD exactly 0.000. Sensor noise alone keeps a live camera above ~0.3 even
# on a motionless scene. 0.15 sits halfway between, with 2x margin to the healthy
# floor. Do not raise it above ~0.25 -- a real static scene lives in 0.3-0.5 for
# stretches well over a hundred frames.
MAD_THRESHOLD = 0.15

# Below this, the frames are byte-for-byte identical for practical purposes: a wedged
# sensor, not a quiet scene. Alerts at this level skip the peer-motion cross-check,
# so an all-cameras-stall still gets reported.
DUPLICATE_HARD_MAD = 0.01

# How long the MAD must stay under threshold before we call it a freeze. Used by
# topics mode, where frames arrive in real time and wall clock is meaningful.
STATIC_FREEZE_S = 2.0

# The same rule for files mode, counted in frames instead of seconds. A saved demo is
# ingested in one burst, so elapsed wall clock says nothing about how long the camera
# was actually stuck -- the frame index is the only honest clock there. 10 frames is
# ~0.7 s at the default --recording_rate_hz 15, against a healthy run length of 0.
STATIC_FREEZE_FRAMES = 10

# Secondary guard against a "nothing in the scene is moving" false positive: a soft
# static reading (below MAD_THRESHOLD but above DUPLICATE_HARD_MAD) only alerts if
# some OTHER camera is moving. With the calibrated threshold this rarely matters, but
# it costs nothing and covers a camera whose freeze still carries faint noise.
REQUIRE_PEER_MOTION = True

# Downsample size for the cheap comparison. 64x36 keeps a full 3-camera check at well
# under a millisecond, so the monitor never competes with the logger for CPU.
THUMB_W, THUMB_H = 64, 36

# Only decode/compare content this often per camera. Arrival timestamps are still
# recorded for every message; this throttles the (cheap but non-zero) pixel work.
CONTENT_SAMPLE_INTERVAL_S = 0.5

# ---------------------------------------------------------------------------
# Constants mirrored from log_demonstration.py -- keep in sync
# ---------------------------------------------------------------------------

LOGGER_SCRIPT = "log_demonstration.py"
DEFAULT_TOPIC_TEMPLATE = "/zed_multi/{cam}"
COLOR_SUFFIX = "rgb/image_rect_color"
DEFAULT_DATA_ROOT = Path.home() / "Desktop" / "data"
MONTAGE_GLOB = "combined_rgb__*.png"
MONTAGE_RE = re.compile(r"combined_rgb__(?P<traj>.+)__(?P<idx>\d+)\.png$")


# ===========================================================================
# Alerting
# ===========================================================================

def raise_alert(camera: str, reason: str, detail: str) -> None:
    """Fire an always-on-top popup + desktop notification + terminal bell.

    The popup runs in a detached child process (this same file re-invoked with
    --popup) rather than inline, for two reasons: tkinter's mainloop blocks, and a
    modal dialog sitting in the monitor's own thread would hide a *second* camera
    freezing while you are reading about the first. Fire-and-forget child processes
    mean N simultaneous alerts stack up as N windows and monitoring never pauses.
    """
    title = f"CAMERA FROZEN: {camera}"
    body = f"{reason}\n\n{detail}\n\nStop the log, unplug/replug '{camera}', restart."

    # 1. Terminal bell + loud stderr line (visible even if the GUI is on another desktop).
    sys.stderr.write("\a")
    sys.stderr.write(f"\n{'!' * 70}\n!! {title}\n!! {reason} -- {detail}\n{'!' * 70}\n")
    sys.stderr.flush()

    # 2. Detached tkinter popup, always on top.
    try:
        subprocess.Popen(
            [sys.executable, os.path.abspath(__file__), "--popup", title, body],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except Exception as exc:  # noqa: BLE001 - an alert must never kill the monitor
        print(f"[warn] popup failed: {exc}")

    # 3. Desktop notification as a second channel, in case the popup lands behind
    #    a fullscreen window. Critical urgency makes it persist until dismissed.
    try:
        subprocess.Popen(
            ["notify-send", "-u", "critical", "-t", "0", title, body.replace("\n", " ")],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        pass


def show_popup(title: str, body: str) -> int:
    """Child-process entry point: a single always-on-top tkinter dialog."""
    import tkinter as tk
    from tkinter import messagebox

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    root.lift()
    # Ring the X11 bell too -- some terminals swallow \a.
    try:
        root.bell()
    except Exception:  # noqa: BLE001
        pass
    messagebox.showerror(title, body, parent=root)
    root.destroy()
    return 0


# ===========================================================================
# Discovering the running logger
# ===========================================================================

def find_logger_process() -> tuple[int, list[str]] | None:
    """Return (pid, argv) of a running log_demonstration.py, or None.

    Scans /proc directly rather than shelling out to pgrep so the argv comes back
    already split -- we need it to recover --camera_ids and --output_dir.
    """
    me = os.getpid()
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        pid = int(entry)
        if pid == me:
            continue
        try:
            raw = Path(f"/proc/{pid}/cmdline").read_bytes()
        except (OSError, PermissionError):
            continue
        argv = [a for a in raw.decode("utf-8", "replace").split("\0") if a]
        if not argv:
            continue
        # Match the script name in any argument, but skip our own monitor process
        # and any editor/grep that happens to mention the file.
        if any(LOGGER_SCRIPT in a for a in argv) and "python" in argv[0].lower():
            if any(Path(__file__).name in a for a in argv):
                continue
            return pid, argv
    return None


def parse_logger_args(argv: list[str]) -> dict:
    """Recover the settings that matter from the logger's own command line.

    Mirrors log_demonstration.py's argparse for the three flags we care about:
    --camera_ids (nargs='*', also defines montage panel order), --output_dir,
    --topic-template.
    """
    out = {"camera_ids": [], "output_dir": None, "topic_template": DEFAULT_TOPIC_TEMPLATE}
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--camera_ids":
            i += 1
            while i < len(argv) and not argv[i].startswith("--"):
                out["camera_ids"].append(argv[i])
                i += 1
            continue
        if a in ("--output_dir", "--output-dir") and i + 1 < len(argv):
            out["output_dir"] = argv[i + 1]
            i += 2
            continue
        if a in ("--topic-template", "--topic_template") and i + 1 < len(argv):
            out["topic_template"] = argv[i + 1]
            i += 2
            continue
        if a.startswith("--output_dir="):
            out["output_dir"] = a.split("=", 1)[1]
        elif a.startswith("--topic-template=") or a.startswith("--topic_template="):
            out["topic_template"] = a.split("=", 1)[1]
        i += 1
    return out


def pid_alive(pid: int) -> bool:
    return Path(f"/proc/{pid}").exists()


# ===========================================================================
# Per-camera freeze state machine (shared by both modes)
# ===========================================================================

def thumbnail(gray_or_bgr: np.ndarray) -> np.ndarray:
    """Cheap comparable fingerprint: 64x36 float32 grayscale."""
    img = gray_or_bgr
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.resize(img, (THUMB_W, THUMB_H), interpolation=cv2.INTER_AREA).astype(np.float32)


class CameraState:
    """Tracks liveness for one camera and decides when to fire / clear an alert.

    Two independent freeze signals:
      SILENT    - no frame observed for FRAME_TIMEOUT_S (camera stopped publishing)
      DUPLICATE - frames keep coming but MAD < MAD_THRESHOLD for STATIC_FREEZE_S
                  (sensor is wedged but the driver keeps republishing the last buffer)

    `alerted` latches so a freeze fires exactly once. It clears the moment healthy
    frames resume, which re-arms the camera for a genuine second freeze later.
    """

    def __init__(self, cam_id: str):
        self.cam_id = cam_id
        self.frame_count = 0
        self.last_frame_time: float | None = None   # last message/file observed
        self.last_thumb: np.ndarray | None = None
        self.last_thumb_time: float = 0.0
        self.last_mad: float | None = None
        self.moving_since: float | None = None      # last time MAD cleared threshold
        self.static_frames = 0                      # consecutive comparisons below MAD
        self.alerted = False
        self.alert_reason = ""
        # Short history so status output can show a smoothed motion estimate.
        self.recent_mads: deque[float] = deque(maxlen=10)

    # -- ingestion ---------------------------------------------------------

    def note_frame(self, now: float) -> None:
        """A frame arrived (content not necessarily inspected)."""
        self.frame_count += 1
        self.last_frame_time = now

    def note_content(self, thumb: np.ndarray, now: float) -> None:
        """Inspect frame content and update the motion estimate."""
        if self.last_thumb is not None:
            self.last_mad = float(np.abs(thumb - self.last_thumb).mean())
            self.recent_mads.append(self.last_mad)
            if self.last_mad >= MAD_THRESHOLD:
                self.moving_since = now
                self.static_frames = 0
            else:
                self.static_frames += 1
        else:
            # First frame: assume alive until proven otherwise.
            self.moving_since = now
        self.last_thumb = thumb
        self.last_thumb_time = now

    # -- evaluation --------------------------------------------------------

    def age(self, now: float) -> float:
        return float("inf") if self.last_frame_time is None else now - self.last_frame_time

    def static_for(self, now: float) -> float:
        return 0.0 if self.moving_since is None else now - self.moving_since

    def evaluate(self, now: float, peer_motion: bool,
                 static_frames_limit: int | None = None) -> tuple[str, str] | None:
        """Return (reason, detail) if this camera JUST transitioned into frozen.

        `static_frames_limit` selects the clock used for the duplicate-frame rule:
        None means wall clock (topics mode), an int means consecutive-frame count
        (files mode, where a whole demo lands on disk in one burst).
        """
        if self.frame_count == 0:
            return None  # never started -- the logger's own wait_for_frames covers that

        silent = self.age(now) > FRAME_TIMEOUT_S
        if static_frames_limit is None:
            stuck = self.static_for(now) > STATIC_FREEZE_S
            stuck_for = f"{self.static_for(now):.1f}s"
        else:
            stuck = self.static_frames >= static_frames_limit
            stuck_for = f"{self.static_frames} consecutive frames"

        # A byte-identical frame is proof on its own; only the softer "small but
        # non-zero change" case needs a moving peer to rule out a static scene.
        hard_dup = self.last_mad is not None and self.last_mad < DUPLICATE_HARD_MAD
        duplicate = not silent and stuck and (
            hard_dup or peer_motion or not REQUIRE_PEER_MOTION)

        if silent or duplicate:
            if self.alerted:
                return None  # already reported this freeze event -- no spam
            self.alerted = True
            if silent:
                self.alert_reason = "SILENT"
                return ("No new frames.",
                        f"Last frame was {self.age(now):.1f}s ago "
                        f"(timeout {FRAME_TIMEOUT_S:.1f}s). Total frames seen: {self.frame_count}.")
            self.alert_reason = "DUPLICATE"
            return ("Frames arriving but the image is not changing.",
                    f"Pixel change (MAD) has been below {MAD_THRESHOLD} for "
                    f"{stuck_for} while other cameras kept moving. "
                    f"Last MAD: {self.last_mad:.3f}.")

        # Healthy -> clear the latch so a later freeze can fire again.
        if self.alerted:
            self.alerted = False
            self.alert_reason = ""
            # Leading newline: the status line is redrawn with \r and has no newline.
            print(f"\n[recovered] {self.cam_id} is producing changing frames again.")
        return None

    def status(self, now: float) -> str:
        if self.frame_count == 0:
            return f"{self.cam_id:<6} waiting"
        age = self.age(now)
        mad = np.mean(self.recent_mads) if self.recent_mads else float("nan")
        flag = f"FROZEN({self.alert_reason})" if self.alerted else "ok"
        return (f"{self.cam_id:<6} {flag:<16} last={age:5.2f}s  "
                f"mad={mad:6.2f}  n={self.frame_count}")


def any_peer_motion(states: dict[str, CameraState], me: str, now: float) -> bool:
    """True if some camera other than `me` is currently showing real motion.

    This is what keeps a legitimately static scene (you thinking between demos) from
    being reported as three simultaneous freezes.

    Uses the peak of the recent MAD history rather than the single latest value: one
    momentarily quiet frame from a peer must not suppress a real freeze alert.
    """
    for cam, st in states.items():
        if cam == me or st.frame_count == 0 or st.alerted:
            continue
        if st.recent_mads and max(st.recent_mads) >= MAD_THRESHOLD:
            return True
    return False


def print_status(states: dict[str, CameraState], now: float, mode: str, extra: str = "") -> None:
    line = " | ".join(st.status(now) for st in states.values())
    stamp = time.strftime("%H:%M:%S")
    sys.stdout.write(f"\r[{stamp}] {mode}: {line}{extra}   ")
    sys.stdout.flush()


# ===========================================================================
# Mode 1: live ROS 2 topic monitoring (real-time, the one that saves a demo)
# ===========================================================================

def monitor_topics(camera_ids: list[str], topic_template: str, logger_pid: int) -> str:
    """Subscribe to the ZED colour topics and watch for freezes until the logger exits.

    Returns a short reason string for why monitoring ended.
    """
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import Image

    states = {c: CameraState(c) for c in camera_ids}

    def decode_for_thumb(msg) -> np.ndarray:
        """Minimal decode -- we only need luminance, so skip the colour conversion.

        ZED publishes bgra8 here. For a frozen-frame check any single channel is a
        perfectly good proxy for the whole image, so we take one channel with a stride
        and avoid copying 3.5 MB per frame.
        """
        buf = np.frombuffer(msg.data, dtype=np.uint8)
        if msg.encoding in ("bgra8", "rgba8"):
            return buf.reshape(msg.height, msg.width, 4)[:, :, 1]      # green channel
        if msg.encoding in ("bgr8", "rgb8"):
            return buf.reshape(msg.height, msg.width, 3)[:, :, 1]
        if msg.encoding == "mono8":
            return buf.reshape(msg.height, msg.width)
        raise ValueError(f"unsupported encoding {msg.encoding}")

    class FreezeWatcher(Node):
        def __init__(self):
            super().__init__("chk_camera_crash")
            # BEST_EFFORT + depth 1: we only ever care about the newest frame, and a
            # RELIABLE subscriber would make the DDS stack work harder on our behalf
            # while the logger is the one that actually needs the bandwidth.
            qos = QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
                durability=DurabilityPolicy.VOLATILE,
            )
            for cam in camera_ids:
                ns = topic_template.format(cam=cam).rstrip("/")
                topic = f"{ns}/{COLOR_SUFFIX}"
                self.create_subscription(Image, topic,
                                         lambda m, c=cam: self._cb(m, c), qos)
                print(f"   watching {topic}")

        def _cb(self, msg: Image, cam: str) -> None:
            now = time.time()
            st = states[cam]
            st.note_frame(now)
            # Throttle the pixel work: arrival is counted for every message, content is
            # only fingerprinted a couple of times a second.
            if now - st.last_thumb_time < CONTENT_SAMPLE_INTERVAL_S:
                return
            try:
                st.note_content(thumbnail(decode_for_thumb(msg)), now)
            except Exception as exc:  # noqa: BLE001 - a bad frame must not kill the watcher
                self.get_logger().warn(f"{cam}: could not decode frame: {exc}")

    rclpy.init(args=None)
    node = FreezeWatcher()
    print(f"\n👁  Live topic monitoring active for {camera_ids} (logger pid {logger_pid})\n")

    reason = "logger exited"
    try:
        next_poll = time.time()
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)
            now = time.time()
            if now < next_poll:
                continue
            next_poll = now + POLL_INTERVAL_S

            if not pid_alive(logger_pid):
                break

            for cam, st in states.items():
                hit = st.evaluate(now, any_peer_motion(states, cam, now))
                if hit:
                    print()  # break out of the \r status line before shouting
                    raise_alert(cam, hit[0], hit[1])
            print_status(states, now, "topics")
    except KeyboardInterrupt:
        reason = "interrupted"
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    print()
    return reason


# ===========================================================================
# Mode 2: montage-file monitoring (post-hoc fallback)
# ===========================================================================

def newest_demo_dir(data_root: Path) -> Path | None:
    """Newest directory containing an images/ subfolder, searched one level deep.

    data/ holds both demo folders directly (rotate_arrow_0729/) and a container of
    them (zed_demos/<description>__<timestamp>/), so both shapes are considered.
    """
    candidates: list[Path] = []
    if not data_root.is_dir():
        return None
    for lvl1 in data_root.iterdir():
        if not lvl1.is_dir():
            continue
        if (lvl1 / "images").is_dir():
            candidates.append(lvl1)
        try:
            for lvl2 in lvl1.iterdir():
                if lvl2.is_dir() and (lvl2 / "images").is_dir():
                    candidates.append(lvl2)
        except PermissionError:
            continue
    if not candidates:
        return None
    return max(candidates, key=lambda p: (p / "images").stat().st_mtime)


def split_montage(path: Path, n_panels: int) -> list[np.ndarray] | None:
    """Split a combined_rgb montage into per-camera grayscale thumbnails.

    save_combined_rgbs() hstacks the cameras in --camera_ids order at a common height,
    and every ZED on this rig is 1280x720, so the panels are equal width.
    """
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None  # still being written -- try again next poll
    h, w = img.shape[:2]
    if n_panels <= 0 or w % n_panels != 0:
        return None
    pw = w // n_panels
    return [thumbnail(img[:, i * pw:(i + 1) * pw]) for i in range(n_panels)]


def monitor_files(camera_ids: list[str], data_root: Path, logger_pid: int) -> str:
    """Watch the newest demo's images/ folder. Post-hoc: fires only after a save.

    Note the timing reality this cannot escape: log_demonstration.py writes the whole
    images/ folder in one burst inside save(), so a 'freeze' here is always reported
    after the affected demo has already been recorded.
    """
    states = {c: CameraState(c) for c in camera_ids}
    current_dir: Path | None = None
    seen: set[str] = set()

    print(f"\n📁 File monitoring active under {data_root} for {camera_ids} "
          f"(logger pid {logger_pid})")
    print("   NOTE: images/ is written only when a demo is saved, so alerts are "
          "post-hoc.\n")

    try:
        while True:
            if not pid_alive(logger_pid):
                break

            demo_dir = newest_demo_dir(data_root)
            if demo_dir is not None and demo_dir != current_dir:
                # New session/demo folder: reset every latch and every baseline.
                current_dir = demo_dir
                seen.clear()
                states = {c: CameraState(c) for c in camera_ids}
                print(f"\n[session] now watching {demo_dir}")

            if current_dir is not None:
                img_dir = current_dir / "images"
                files = sorted(p for p in img_dir.glob(MONTAGE_GLOB) if p.name not in seen)
                for f in files:
                    panels = split_montage(f, len(camera_ids))
                    if panels is None:
                        continue  # partial write; it will be picked up next cycle
                    seen.add(f.name)
                    ftime = f.stat().st_mtime
                    for cam, panel in zip(camera_ids, panels):
                        states[cam].note_frame(ftime)
                        states[cam].note_content(panel, ftime)

                    # Evaluate after EVERY frame, not once per poll. A whole demo's
                    # images/ appears in a single burst, so a freeze that started and
                    # ended inside one batch would otherwise be stepped straight over.
                    for cam, st in states.items():
                        hit = st.evaluate(ftime, any_peer_motion(states, cam, ftime),
                                          static_frames_limit=STATIC_FREEZE_FRAMES)
                        if hit:
                            print()
                            raise_alert(cam, hit[0], hit[1])

                latest = max((s.last_frame_time or 0.0) for s in states.values())
                if latest:
                    print_status(states, latest, "files",
                                 extra=f" [{len(seen)} frames]")

            time.sleep(POLL_INTERVAL_S)
    except KeyboardInterrupt:
        print()
        return "interrupted"
    print()
    return "logger exited"


# ===========================================================================
# Main loop: idle until the logger runs, monitor, then idle again
# ===========================================================================

def ros_available() -> bool:
    try:
        import rclpy  # noqa: F401
        import sensor_msgs.msg  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["auto", "topics", "files"], default="auto",
                    help="auto = live ROS topics if available, else montage files")
    ap.add_argument("--data-root", type=str, default=str(DEFAULT_DATA_ROOT),
                    help="root under which demo folders appear (files mode)")
    ap.add_argument("--camera-ids", type=str, nargs="*", default=None,
                    help="override; normally read from the running logger's cmdline")
    ap.add_argument("--topic-template", type=str, default=None,
                    help="override; normally read from the running logger's cmdline")
    ap.add_argument("--once", action="store_true",
                    help="exit after the logger stops instead of idling for the next run")
    ap.add_argument("--test-alert", action="store_true", help="fire a test popup and exit")
    ap.add_argument("--popup", nargs=2, metavar=("TITLE", "BODY"),
                    help=argparse.SUPPRESS)  # internal: child-process popup renderer
    args = ap.parse_args()

    if args.popup:
        return show_popup(args.popup[0], args.popup[1])
    if args.test_alert:
        raise_alert("zedm", "Test alert.", "If you can see this, alerting works.")
        time.sleep(1.0)
        return 0

    data_root = Path(args.data_root).expanduser()
    have_ros = ros_available()
    if args.mode == "topics" and not have_ros:
        print("❌ --mode topics but rclpy is not importable. Source ROS first:")
        print("   source /opt/ros/humble/setup.bash && "
              "source ~/Desktop/ros2_ws/install/setup.bash")
        return 1

    print("=" * 70)
    print("🧊 Camera freeze detector for log_demonstration.py")
    print("=" * 70)
    print(f"mode          : {args.mode}"
          f"{'' if have_ros else '   (rclpy unavailable -> files)'}")
    print(f"data root     : {data_root}")
    print(f"freeze rules  : SILENT    no frame for > {FRAME_TIMEOUT_S}s")
    print(f"                DUPLICATE MAD < {MAD_THRESHOLD} for > {STATIC_FREEZE_S}s "
          f"(topics) / {STATIC_FREEZE_FRAMES} frames (files)")
    print("=" * 70)

    while True:
        # ---- idle: wait for the logger to appear -------------------------
        proc = find_logger_process()
        while proc is None:
            sys.stdout.write(f"\r[{time.strftime('%H:%M:%S')}] idle -- waiting for "
                             f"{LOGGER_SCRIPT} to start...   ")
            sys.stdout.flush()
            time.sleep(1.0)
            proc = find_logger_process()

        pid, argv = proc
        cfg = parse_logger_args(argv)
        camera_ids = args.camera_ids or cfg["camera_ids"]
        topic_template = args.topic_template or cfg["topic_template"]
        root = data_root
        if cfg["output_dir"]:
            # Prefer the directory the logger is actually writing to.
            root = Path(cfg["output_dir"]).expanduser()

        print(f"\n\n▶️  Detected {LOGGER_SCRIPT} (pid {pid})")
        print(f"   cameras: {camera_ids or '(none declared)'}")
        print(f"   output : {root}")

        if not camera_ids:
            print("   No --camera_ids on the logger's command line and none passed to "
                  "the monitor; nothing to watch. Waiting for it to exit.")
            while pid_alive(pid):
                time.sleep(1.0)
        else:
            use_topics = args.mode == "topics" or (args.mode == "auto" and have_ros)
            if use_topics:
                reason = monitor_topics(camera_ids, topic_template, pid)
            else:
                reason = monitor_files(camera_ids, root, pid)
            print(f"⏹  Monitoring stopped: {reason}")
            if reason == "interrupted":
                return 0

        if args.once:
            return 0
        print("   Returning to idle. Ctrl-C to quit.\n")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nbye")
        sys.exit(0)
