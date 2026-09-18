#!/usr/bin/env python3
"""Live overlay of the camera against a frame from an already-recorded demo, to see
whether the camera has moved since that dataset was collected.

WHY
    The hand-eye extrinsics in `extrinsics_cam_1*.yaml` is a fixed camera pose. Every
    object pose derived from it -- every aux label -- is wrong by however much the camera
    has shifted since. A still-image comparison tells you *that* something moved; this
    tells you *while you are nudging the camera back*, which is what you actually need.

WHAT YOU SEE
    A window with the live stream composited against the reference frame, plus a running
    estimate of how far the two disagree in pixels. Static scenery is the thing to watch:
    the table edges and whatever is behind it. The arm and the manipulated objects will
    obviously differ and mean nothing here.

    The readout is the median displacement of RANSAC-inlier SIFT matches. Under ~1 px is
    as good as this measurement gets; a few px is a real but small shift; tens of px means
    the camera was knocked. At fx=912 (D435 at 1280x720), 4.5 px is about 0.28 deg, or
    5 mm of lateral error at 1 m.

KEYS
    1 live   2 reference   3 blend   4 edges (red=reference, green=live)   5 abs-diff
    s        save the current composite next to the reference dataset
    q / ESC  quit

USAGE
    source /opt/ros/humble/setup.bash && source ~/anaconda3/bin/activate deoxys
    python scripts/r2st_camera_drift.py \
        --h5 ~/Desktop/data/square60/square__09-07_01-01-34/traj_0.h5
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np


def load_reference(h5_path: Path, key: str, frame: int) -> np.ndarray:
    import h5py

    with h5py.File(h5_path, "r") as h:
        if key not in h:
            merged = f"obs/sensor_data/cam_1/rgb"
            assert merged in h, f"{h5_path} has neither {key!r} nor {merged!r}; keys: {list(h.keys())}"
            key = merged
        return np.asarray(h[key][frame])


class Matcher:
    """Median displacement between reference and live, over RANSAC-inlier SIFT matches."""

    def __init__(self) -> None:
        import cv2

        self._sift = cv2.SIFT_create(3000)
        self._bf = cv2.BFMatcher()
        self._ref_kp = self._ref_des = None
        self.text = "measuring..."

    def set_reference(self, gray_ref) -> None:
        self._ref_kp, self._ref_des = self._sift.detectAndCompute(gray_ref, None)

    def update(self, gray_live) -> None:
        import cv2

        kp, des = self._sift.detectAndCompute(gray_live, None)
        if des is None or self._ref_des is None or len(kp) < 8:
            self.text = "too few features"
            return
        good = [m for m, n in self._bf.knnMatch(self._ref_des, des, k=2) if m.distance < 0.75 * n.distance]
        if len(good) < 8:
            self.text = f"only {len(good)} matches"
            return
        src = np.float32([self._ref_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
        if H is None or mask is None or int(mask.sum()) < 6:
            self.text = "no stable homography"
            return
        inl = mask.ravel().astype(bool)
        d = np.linalg.norm(dst.reshape(-1, 2)[inl] - src.reshape(-1, 2)[inl], axis=1)
        scale = float(np.sqrt(abs(np.linalg.det(H[:2, :2]))))
        self.text = (f"median {np.median(d):5.2f} px   max {d.max():5.2f} px   "
                     f"scale {scale:6.4f}   inliers {int(mask.sum())}/{len(good)}")


def main() -> int:
    import cv2
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--h5", type=Path, required=True, help="a recorded demo to compare against")
    p.add_argument("--key", default="d435__color_image", help="dataset inside the h5 (default d435__color_image)")
    p.add_argument("--frame", type=int, default=0)
    p.add_argument("--topic", default="/camera/d435/color/image_raw")
    p.add_argument("--mode", type=int, default=4, choices=[1, 2, 3, 4, 5], help="starting view (default 4, edges)")
    p.add_argument("--measure-every", type=float, default=1.0, help="seconds between displacement measurements")
    args = p.parse_args()

    ref_rgb = load_reference(args.h5.expanduser(), args.key, args.frame)
    ref = cv2.cvtColor(ref_rgb, cv2.COLOR_RGB2BGR)
    ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    ref_edges = cv2.Canny(ref_gray, 60, 160)
    print(f"reference: {args.h5.name} [{args.key}] frame {args.frame}  {ref.shape[1]}x{ref.shape[0]}")

    matcher = Matcher()
    matcher.set_reference(ref_gray)

    latest: dict[str, np.ndarray] = {}
    rclpy.init()
    node = Node("r2st_camera_drift")
    node.create_subscription(
        Image, args.topic,
        lambda m: latest.__setitem__("rgb", np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, 3)), 10)

    mode = args.mode
    last_measure = 0.0
    win = "camera drift  |  1 live  2 ref  3 blend  4 edges  5 diff  s save  q quit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1280, 720)
    print("waiting for frames ...")

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)
            if "rgb" not in latest:
                continue
            live = cv2.cvtColor(latest["rgb"], cv2.COLOR_RGB2BGR)
            if live.shape != ref.shape:
                live = cv2.resize(live, (ref.shape[1], ref.shape[0]))
            gray = cv2.cvtColor(live, cv2.COLOR_BGR2GRAY)

            now = time.time()
            if now - last_measure > args.measure_every:
                matcher.update(gray)
                last_measure = now

            if mode == 1:
                view = live.copy()
            elif mode == 2:
                view = ref.copy()
            elif mode == 3:
                view = cv2.addWeighted(ref, 0.5, live, 0.5, 0)
            elif mode == 4:
                view = np.zeros_like(ref)
                view[..., 2] = ref_edges                      # red   = reference
                view[..., 1] = cv2.Canny(gray, 60, 160)       # green = live
            else:
                view = cv2.absdiff(ref, live)

            label = {1: "LIVE", 2: "REFERENCE", 3: "BLEND", 4: "EDGES red=ref green=live", 5: "ABS DIFF"}[mode]
            cv2.rectangle(view, (0, 0), (view.shape[1], 62), (0, 0, 0), -1)
            cv2.putText(view, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(view, matcher.text, (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imshow(win, view)

            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27):
                break
            if ord("1") <= k <= ord("5"):
                mode = k - ord("0")
            if k == ord("s"):
                out = args.h5.expanduser().parent / f"drift_mode{mode}_{int(now)}.png"
                cv2.imwrite(str(out), view)
                print("saved", out)
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
