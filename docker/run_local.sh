#!/usr/bin/env bash
# Same as run.sh, but for a checkout outside /data/zeyu: the workspace is the directory holding this repo
# (e.g. /data/ericli/Deformable), so its own deoxys_control and zeyu_panda_utils are mounted and used. The
# panda-deoxys image bakes in /data/zeyu paths (WORKDIR, PYTHONPATH); both are overridden here, no rebuild needed.
#   docker/run_local.sh python3 scripts/panda_teleop.py --vendor-id 1133 --product-id 50726
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE="$(dirname "$REPO")"
TTY_FLAGS=()
[ -t 0 ] && TTY_FLAGS=(-it)

# deoxys' logger writes logs/debug.log and logs/error.log relative to the working directory and fails if logs/ is
# missing (it is gitignored).
mkdir -p "$REPO/logs"

# Pass through the SpaceMouse (any 3Dconnexion / Logitech-3Dconnexion HID) if plugged in; the
# hidraw number changes on replug, so look it up each time.
DEVICE_FLAGS=()
for dev in /sys/class/hidraw/hidraw*; do
    if grep -q -i -E 'HID_ID=0003:0000(046D:0000C6|256F:0000C6)' "$dev/device/uevent" 2>/dev/null; then
        DEVICE_FLAGS+=(--device "/dev/$(basename "$dev")")
    fi
done
# The pip hidapi wheel opens devices through libusb, so the raw USB node is needed too.
for dev in /sys/bus/usb/devices/*; do
    vid=$(cat "$dev/idVendor" 2>/dev/null) || continue
    pid=$(cat "$dev/idProduct" 2>/dev/null) || continue
    if [[ "$vid:$pid" =~ ^(046d|256f):c6 ]]; then
        DEVICE_FLAGS+=(--device "$(printf '/dev/bus/usb/%03d/%03d' "$(cat "$dev/busnum")" "$(cat "$dev/devnum")")")
    fi
done
# Pass through the front USB webcam (Logitech C310) if plugged in: found by its stable by-id path, since the
# /dev/videoN number changes when cameras are plugged in. Same as zeyu-Policy-deploy-Franka/run.sh.
FRONT_CAMERA_ID="${FRONT_CAMERA_ID:-/dev/v4l/by-id/usb-046d_081b_C1531760-video-index0}"
if [[ -e "$FRONT_CAMERA_ID" ]]; then
    DEVICE_FLAGS+=(--device "$(readlink -f "$FRONT_CAMERA_ID")" -v /dev/v4l:/dev/v4l:ro
                   --group-add "$(getent group video | cut -d: -f3)")
fi

exec docker run --rm "${TTY_FLAGS[@]}" "${DEVICE_FLAGS[@]}" -v /run/udev:/run/udev:ro \
    --network host --ipc host \
    --user "$(id -u):$(id -g)" --group-add plugdev \
    -e HOME=/tmp -e DISPLAY="${DISPLAY:-}" -v /tmp/.X11-unix:/tmp/.X11-unix \
    -e FASTRTPS_DEFAULT_PROFILES_FILE="$REPO/docker/fastdds_udp_only.xml" \
    -e PYTHONPATH="$WORKSPACE/deoxys_control/deoxys:$REPO/src" \
    -v "$WORKSPACE:$WORKSPACE" -w "$REPO" \
    panda-deoxys "${@:-bash}"
