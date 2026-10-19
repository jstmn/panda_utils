#!/usr/bin/env python3
"""Read this rig's RealSense calibration off the device and emit the real2sim-toolkit
`src/r2st/constants.py` block for it.

WHY THIS EXISTS
    constants.py opens with "Values are hardware readouts, not guessed defaults" -- but they
    are readouts from the toolkit author's D435, not from ours. RealSense factory calibration
    is per-unit.

    That would be harmless if it were checked, and it is not. track_object.py asserts only
    that the RGB frame's HxW equals the intrinsics entry's HxW; a different unit's fx/fy/cx/cy
    at the same resolution passes silently and biases every pose FoundationPose returns --
    which is to say, every aux label. Intrinsics also change with the stream profile, so the
    shipped 640x480 entry is simply wrong for a 1280x720 recording.

    So: read them from the camera that will actually record, at the profile it will actually
    record at.

ALIGNED DEPTH
    Record with depth aligned to colour (realsense2_camera `align_depth.enable:=true`), which
    is what log_demonstration.py's uint16-mm HDF5 layout already assumes. Then depth shares
    the colour grid, and you pass `--depth-intrinsics-source rgb` to
    estimate_camera_extrinsics.py, whose `depth` branch (the reproject_depth_to_color_frame
    call) is the only consumer of DEPTH_INTRINSICS / DEPTH_TO_COLOR_*. Those three entries are
    then unused -- but they must still EXIST, because constants.py asserts all four dicts
    share their keys. They are emitted here with the device's true values rather than
    placeholders, so the `depth` branch also stays correct if you ever record unaligned.

USAGE
    source ~/anaconda3/bin/activate act
    python scripts/r2st/r2st_d435_intrinsics.py                          # 640x480, the USB2-safe profile
    python scripts/r2st/r2st_d435_intrinsics.py --color 1280 720 --depth 848 480 --model-id d435_720p
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

try:
    import pyrealsense2 as rs
except ImportError:
    sys.exit("pyrealsense2 not installed. `pip install pyrealsense2` in this env.")


def fmt(model_id: str, name: str, i) -> str:
    return (f'    "{model_id}": CameraIntrinsics(\n'
            f"        width={i.width},\n"
            f"        height={i.height},\n"
            f"        fx={i.fx!r},\n"
            f"        fy={i.fy!r},\n"
            f"        cx={i.ppx!r},\n"
            f"        cy={i.ppy!r},\n"
            f"    ),")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--color", type=int, nargs=2, metavar=("W", "H"), default=[640, 480])
    p.add_argument("--depth", type=int, nargs=2, metavar=("W", "H"), default=[640, 480])
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--model-id", default="d435",
                   help="Key to write into constants.py, passed later as --camera-model-id")
    p.add_argument("--serial", default=None, help="Device serial, if more than one is attached")
    args = p.parse_args()

    cfg = rs.config()
    if args.serial:
        cfg.enable_device(args.serial)
    cfg.enable_stream(rs.stream.color, *args.color, rs.format.bgr8, args.fps)
    cfg.enable_stream(rs.stream.depth, *args.depth, rs.format.z16, args.fps)

    pipe = rs.pipeline()
    try:
        profile = pipe.start(cfg)
    except RuntimeError as e:
        return _report_unsupported(e, args)

    try:
        dev = profile.get_device()
        color_p = profile.get_stream(rs.stream.color).as_video_stream_profile()
        depth_p = profile.get_stream(rs.stream.depth).as_video_stream_profile()
        ci, di = color_p.get_intrinsics(), depth_p.get_intrinsics()
        ext = depth_p.get_extrinsics_to(color_p)
        scale = dev.first_depth_sensor().get_depth_scale()
        usb = dev.get_info(rs.camera_info.usb_type_descriptor)
        serial = dev.get_info(rs.camera_info.serial_number)
        name = dev.get_info(rs.camera_info.name)
    finally:
        pipe.stop()

    # librealsense stores rs2_extrinsics.rotation COLUMN-major; constants.py holds ordinary
    # row-major numpy. Transposing after the (3,3) reshape is what converts between them --
    # skipping it yields the inverse rotation, which is a silent few-mm error in the depth
    # reprojection rather than a crash.
    R = np.asarray(ext.rotation, dtype=np.float64).reshape(3, 3).T
    t = np.asarray(ext.translation, dtype=np.float64)

    log = sys.stderr.write
    log(f"# device      : {name}  serial {serial}  ({usb})\n")
    log(f"# color       : {ci.width}x{ci.height}  model={ci.model}  coeffs={list(ci.coeffs)}\n")
    log(f"# depth       : {di.width}x{di.height}  scale={scale} m/unit\n")
    if abs(scale - 0.001) > 1e-9:
        log(f"# !! depth scale is {scale}, not 0.001 m/unit. log_demonstration.decode_depth\n"
            f"#    passes 16UC1 through as millimetres, so a non-mm unit would be recorded\n"
            f"#    wrong. Set the depth units to 1000 um, or scale in the logger.\n")
    if any(abs(c) > 1e-6 for c in ci.coeffs):
        log("# !! color distortion coeffs are non-zero. The toolkit unprojects with a plain\n"
            "#    pinhole K, so record rectified colour or accept the residual.\n")
    log("#\n# Paste into real2sim-toolkit/src/r2st/constants.py:\n\n")

    print("# --- COLOR_INTRINSICS ---")
    print(fmt(args.model_id, "color", ci))
    print("\n# --- DEPTH_INTRINSICS ---")
    print(fmt(args.model_id, "depth", di))
    print("\n# --- DEPTH_TO_COLOR_ROTATION ---")
    print(f'    "{args.model_id}": np.array(')
    print(f"        {np.array2string(R, separator=', ', precision=9)},")
    print("        dtype=np.float64,\n    ),")
    print("\n# --- DEPTH_TO_COLOR_TRANSLATION ---")
    print(f'    "{args.model_id}": np.array({np.array2string(t, separator=", ", precision=9)}, '
          "dtype=np.float64),")
    return 0


def _report_unsupported(err: RuntimeError, args) -> int:
    print(f"Could not start the requested profile: {err}\n", file=sys.stderr)
    try:
        dev = rs.context().query_devices()[0]
    except IndexError:
        print("No RealSense device attached.", file=sys.stderr)
        return 1
    print(f"{dev.get_info(rs.camera_info.name)} is on "
          f"{dev.get_info(rs.camera_info.usb_type_descriptor)}. Supported profiles:", file=sys.stderr)
    for sensor in dev.query_sensors():
        seen = set()
        for prof in sensor.get_stream_profiles():
            v = prof.as_video_stream_profile()
            if v and prof.stream_type() in (rs.stream.color, rs.stream.depth):
                seen.add((str(prof.stream_type()).split(".")[-1], v.width(), v.height(), prof.fps()))
        for s in sorted(seen, reverse=True):
            print(f"  {s[0]:6s} {s[1]}x{s[2]} @ {s[3]}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
