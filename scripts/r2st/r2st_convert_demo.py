#!/usr/bin/env python3
"""Convert `log_demonstration.py` trajectories into the merged sensor-data HDF5 that
real2sim-toolkit (`examples/track_object.py`, `examples/estimate_camera_extrinsics.py`) reads.

WHY THIS EXISTS
    The two repos disagree on layout, not on content:

        log_demonstration.py  ->  traj_0__VGA.h5
                                     joint_states_q            (T, 9)   float64
                                     joint_states_dq           (T, 9)   float64
                                     <cam>__color_image        (T,H,W,3) uint8   RGB
                                     <cam>__depth_image        (T,H,W)   uint16  mm
                                   <cam>_camera_info_color.json          (sibling file)

        real2sim-toolkit       ->  merged_sensor_data.h5
                                     obs/qpos                            (T, D)
                                     obs/sensor_data/<cam>/rgb           (T,H,W,3) uint8
                                     obs/sensor_data/<cam>/depth         (T,H,W)   uint16 mm
                                     obs/sensor_data/<cam>/timestamp_ms  (T,)
                                     obs/sensor_data/<cam>/rgb_timestamp_ms (T,)

    Both store colour as RGB and depth as uint16 millimetres registered to the colour grid
    (log_demonstration.decode_depth converts ZED's 32FC1 metres for exactly this reason), so
    this is a regroup and rename, not a conversion. Timestamps are not recorded by the logger;
    they are synthesised from --rate-hz, which is sound because both toolkit scripts index
    RGB-D and qpos by frame, and only the schema check reads the timestamps.

    qpos is TRUNCATED to --arm-dof columns. The logger writes 7 arm joints plus 2 finger
    joints; estimate_camera_extrinsics._expand_demo_qpos_to_joint_names asserts the demo's
    column count is <= the Jrl2 robot's actuator count and maps columns onto a leading prefix
    of the actuated joint names. Jrl2's `panda` has NINE actuated joints --
    panda_joint1..7 plus panda_finger_joint1/2 -- and its URDF really does carry
    panda_hand/leftfinger/rightfinger (the README's "wrist flange only" describes the
    __eef NAMING, not the model). Every actuated joint the demo does not supply is pinned
    at 0, i.e. a CLOSED gripper.

    So on this rig, which does have a Franka Hand, --arm-dof 9 is the right choice when
    the demo was recorded without --no-gripper: the render then shows the fingers where
    they actually were. --arm-dof 7 silently renders them shut, and because
    estimate_camera_extrinsics' chamfer is one-sided real->sim, the real open fingers
    have no sim point to match and push the fitted camera pose around.

INTRINSICS ARE NOT CARRIED IN THE H5
    track_object.py looks intrinsics up by --camera-model-id in
    real2sim-toolkit/src/r2st/constants.py and hard-asserts the RGB resolution matches. Only
    'd435' ships. `--emit-intrinsics` prints the constants.py block for this rig's camera from
    the logger's own camera_info JSON.

USAGE
    source ~/anaconda3/bin/activate act

    # 1. print the constants.py entry to paste into real2sim-toolkit
    python scripts/r2st/r2st_convert_demo.py --demo-dir ~/Desktop/data/hyeonhoo/lift_peg_0729 \
        --camera zed2i --emit-intrinsics

    # 2. convert every traj_*.h5 in the directory
    python scripts/r2st/r2st_convert_demo.py --demo-dir ~/Desktop/data/hyeonhoo/lift_peg_0729 \
        --camera zed2i --out-camera cam_1 --rate-hz 15 \
        --outdir ~/Desktop/data/r2st/lift_peg_0729
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

COLOR_SUFFIX = "__color_image"
DEPTH_SUFFIX = "__depth_image"


def emit_intrinsics(demo_dir: Path, camera: str, model_id: str) -> int:
    """Print a constants.py block from the logger's camera_info JSON."""
    info_path = demo_dir / f"{camera}_camera_info_color.json"
    if not info_path.is_file():
        print(f"ERROR: {info_path} not found. The logger writes it next to the traj_*.h5 files;\n"
              f"       recordings made with --no-depth still write the colour one.", file=sys.stderr)
        return 1
    info = json.loads(info_path.read_text())
    # ROS2 sensor_msgs/CameraInfo names the 3x3 matrix 'k'; ROS1 named it 'K'. The logger
    # dumps whatever the message carried, so accept both.
    k = info.get("k", info.get("K"))
    assert k is not None, f"{info_path}: neither 'k' nor 'K' present, keys={sorted(info)}"
    K = np.asarray(k, dtype=np.float64).reshape(3, 3)
    w, h = int(info["width"]), int(info["height"])
    print(f'    "{model_id}": CameraIntrinsics(')
    print(f"        width={w},")
    print(f"        height={h},")
    print(f"        fx={K[0, 0]!r},")
    print(f"        fy={K[1, 1]!r},")
    print(f"        cx={K[0, 2]!r},")
    print(f"        cy={K[1, 2]!r},")
    print("    ),")
    print()
    print(f"# Paste into COLOR_INTRINSICS in real2sim-toolkit/src/r2st/constants.py.", file=sys.stderr)
    print(f"# DEPTH_INTRINSICS, DEPTH_TO_COLOR_ROTATION and DEPTH_TO_COLOR_TRANSLATION need a",
          file=sys.stderr)
    print(f"# '{model_id}' key too -- the module asserts the four dicts share their keys. For a",
          file=sys.stderr)
    print(f"# depth stream already registered to colour (ZED depth_registered, RealSense", file=sys.stderr)
    print(f"# align_depth), depth intrinsics == colour intrinsics and the extrinsics are", file=sys.stderr)
    print(f"# identity/zero.", file=sys.stderr)
    return 0


def convert_one(src: Path, dst: Path, camera: str, out_camera: str,
                rate_hz: float, arm_dof: int) -> None:
    with h5py.File(src, "r") as f:
        color_key, depth_key = camera + COLOR_SUFFIX, camera + DEPTH_SUFFIX
        if color_key not in f:
            cams = sorted(k[: -len(COLOR_SUFFIX)] for k in f if k.endswith(COLOR_SUFFIX))
            raise SystemExit(f"{src}: no '{color_key}'. Cameras present: {cams}")
        if depth_key not in f:
            raise SystemExit(
                f"{src}: no '{depth_key}'.\n"
                f"  This demo was recorded with --no-depth. FoundationPose needs metric depth;\n"
                f"  RGB alone cannot produce the aux pose target. Re-record without --no-depth."
            )
        rgb = f[color_key][:]
        depth = f[depth_key][:]
        if "joint_states_q" not in f:
            raise SystemExit(f"{src}: no 'joint_states_q' (needed as obs/qpos for extrinsics)")
        qpos_full = f["joint_states_q"][:]

    # Mirror real2sim-toolkit/src/r2st/utils.py:validate_merged_camera_group, here rather than
    # after the (slow, large) write.
    assert rgb.ndim == 4 and rgb.shape[3] == 3, f"{src}: rgb must be NxHxWx3, got {rgb.shape}"
    assert rgb.dtype == np.uint8, f"{src}: rgb must be uint8, got {rgb.dtype}"
    assert depth.ndim == 3, f"{src}: depth must be NxHxW, got {depth.shape}"
    assert depth.dtype == np.uint16, f"{src}: depth must be uint16 mm, got {depth.dtype}"
    assert depth.shape[1:] == rgb.shape[1:3], \
        f"{src}: depth {depth.shape[1:]} != rgb {rgb.shape[1:3]} (depth must be registered to colour)"

    n = min(rgb.shape[0], depth.shape[0], qpos_full.shape[0])
    if len({rgb.shape[0], depth.shape[0], qpos_full.shape[0]}) > 1:
        print(f"  ! ragged lengths rgb={rgb.shape[0]} depth={depth.shape[0]} "
              f"qpos={qpos_full.shape[0]} -> truncating to {n}")
    rgb, depth, qpos_full = rgb[:n], depth[:n], qpos_full[:n]

    assert qpos_full.shape[1] >= arm_dof, \
        f"{src}: joint_states_q has {qpos_full.shape[1]} columns, --arm-dof is {arm_dof}"
    qpos = np.ascontiguousarray(qpos_full[:, :arm_dof].astype(np.float64))

    ts = (np.arange(n, dtype=np.float64) * (1000.0 / rate_hz))

    # Depth sanity: FoundationPose registers off frame 0, so a frame 0 with no returns on the
    # object is a silent failure later, not an error here.
    valid0 = float((depth[0] > 0).mean())
    d = depth[depth > 0]
    span = f"{d.min() / 1000:.2f}-{np.percentile(d, 99) / 1000:.2f} m" if d.size else "EMPTY"

    dst.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(dst, "w") as g:
        grp = g.create_group(f"obs/sensor_data/{out_camera}")
        grp.create_dataset("rgb", data=rgb, compression="gzip", compression_opts=1)
        grp.create_dataset("depth", data=depth, compression="gzip", compression_opts=1)
        grp.create_dataset("timestamp_ms", data=ts)
        grp.create_dataset("rgb_timestamp_ms", data=ts)
        g.create_dataset("obs/qpos", data=qpos)

    print(f"  {src.name} -> {dst}  T={n}  {rgb.shape[1]}x{rgb.shape[2]}  "
          f"depth valid[0]={valid0:.1%}  range={span}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--demo-dir", type=Path, required=True,
                   help="Directory of traj_*.h5 written by log_demonstration.py")
    p.add_argument("--camera", required=True,
                   help="Camera id as recorded, e.g. zed2i / zed / zedm / d435")
    p.add_argument("--out-camera", default="cam_1",
                   help="Camera name inside the merged h5, passed to the toolkit as --camera")
    p.add_argument("--outdir", type=Path,
                   help="Output root; one demonstration_<i>/ per traj. Required unless --emit-intrinsics")
    p.add_argument("--rate-hz", type=float, default=15.0,
                   help="Recording rate the demos were captured at, for synthetic timestamps")
    p.add_argument("--arm-dof", type=int, default=7,
                   help="Leading joint columns to keep as obs/qpos. 7 = Panda arm only "
                        "(correct when recorded with --no-gripper). Use 9 to keep the two "
                        "finger columns too: Jrl2's panda actuates 9 joints and pins any "
                        "the demo omits at 0 (closed gripper), which mis-poses the render.")
    p.add_argument("--model-id", default=None,
                   help="camera_model_id for --emit-intrinsics (default: --camera)")
    p.add_argument("--emit-intrinsics", action="store_true",
                   help="Print the r2st constants.py intrinsics block and exit")
    args = p.parse_args()

    if not args.demo_dir.is_dir():
        raise SystemExit(f"--demo-dir not a directory: {args.demo_dir}")

    if args.emit_intrinsics:
        return emit_intrinsics(args.demo_dir, args.camera, args.model_id or args.camera)

    if args.outdir is None:
        raise SystemExit("--outdir is required unless --emit-intrinsics")

    trajs = sorted(args.demo_dir.glob("traj_*.h5"),
                   key=lambda q: int(q.stem.split("_")[1]))
    if not trajs:
        raise SystemExit(f"no traj_*.h5 under {args.demo_dir}")

    print(f"{len(trajs)} trajectories: {args.demo_dir} -> {args.outdir}")
    for src in trajs:
        idx = int(src.stem.split("_")[1])
        convert_one(src, args.outdir / f"demonstration_{idx}" / "merged_sensor_data.h5",
                    args.camera, args.out_camera, args.rate_hz, args.arm_dof)
    return 0


if __name__ == "__main__":
    sys.exit(main())
