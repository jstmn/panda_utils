"""Standalone SpaceMouse check — no robot, no ROS, no network.

Verifies the device opens, the reader thread stays alive, and every axis reports.
Use this to isolate SpaceMouse problems from robot problems before teleoperating.

    python scripts/test_spacemouse.py

Defaults match the SpaceNavigator on this machine (046d:c626). Pass --vendor-id /
--product-id for a different unit.

If it reports "open failed", see README_SETUP.md — usually the HID driver bind
(/sys/bus/hid/drivers/logitech/bind) or the udev rule.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time

from deoxys.utils.io_devices import SpaceMouse
from termcolor import cprint

AXES = ["x", "y", "z", "roll", "pitch", "yaw"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vendor-id", type=int, default=1133)   # 0x046D
    parser.add_argument("--product-id", type=int, default=50726)  # 0xC626
    parser.add_argument("--seconds", type=float, default=20.0)
    args = parser.parse_args()

    try:
        device = SpaceMouse(vendor_id=args.vendor_id, product_id=args.product_id)
    except OSError as exc:
        cprint(f"[FAIL] could not open SpaceMouse: {exc}", "red")
        cprint("       Check the driver bind and udev rule - see README_SETUP.md.", "red")
        return 1

    device.start_control()
    cprint(f"\nMove and TWIST the puck for {args.seconds:.0f}s. Ctrl-C to stop early.\n", "cyan")

    # Track the largest magnitude seen per axis so a brief nudge still registers.
    peak = [0.0] * 6
    grasp_seen = False
    deadline = time.time() + args.seconds

    try:
        while time.time() < deadline:
            time.sleep(0.05)
            control = [float(v) for v in device.control]
            for i, value in enumerate(control):
                peak[i] = max(peak[i], abs(value))
            if device.control_gripper:
                grasp_seen = True
            bar = "  ".join(f"{a}={v:+.2f}" for a, v in zip(AXES, control))
            print(f"\r  {bar}   ", end="", flush=True)
    except KeyboardInterrupt:
        pass

    print("\n")
    reader_alive = any(t.is_alive() and t is not threading.current_thread()
                       for t in threading.enumerate())

    ok = True
    for axis, value in zip(AXES, peak):
        if value > 0.01:
            cprint(f"  [ OK ] {axis:<5} responded (peak {value:.2f})", "green")
        else:
            cprint(f"  [WARN] {axis:<5} never moved", "yellow")
            ok = False

    cprint(f"  [{'  OK ' if grasp_seen else 'WARN'}] gripper button "
           f"{'pressed' if grasp_seen else 'never pressed'}",
           "green" if grasp_seen else "yellow")
    cprint(f"  [{'  OK ' if reader_alive else 'FAIL'}] reader thread "
           f"{'alive' if reader_alive else 'DIED - check for IndexError above'}",
           "green" if reader_alive else "red")

    if ok and reader_alive:
        cprint("\nSpaceMouse fully working.\n", "green")
        return 0
    cprint("\nSome axes did not report. If you did move them, the report layout "
           "may need adjusting in deoxys spacemouse.py (see LOCAL PATCH notes).\n", "yellow")
    return 1


if __name__ == "__main__":
    sys.exit(main())
