#!/bin/bash
# Stop everything ~/Desktop/ros2_ws/zed_tools/launch_all.sh started, freeing the
# cameras for ZED_Depth_Viewer or a fresh launch.
#
# Needed because launch_all.sh starts each container / launch under setsid, so
# Ctrl+C only kills the script itself and the camera processes keep running.
#
# Usage: bash scripts/stop_zed.sh

PATS='launch_all\.sh|zed_container_[0-9]|zed_camera\.launch\.py|__ns:=/zed_multi'

# Never match this script or the shells that invoked it.
SKIP=" $$ "
p=$$
while [ -n "$p" ] && [ "$p" -gt 1 ]; do
  p=$(ps -o ppid= -p "$p" 2>/dev/null | tr -d ' ')
  SKIP="$SKIP$p "
done

targets() { # [pattern]
  for pid in $(pgrep -f "${1:-$PATS}"); do
    [[ $SKIP == *" $pid "* ]] || echo "$pid"
  done
}

pids=$(targets)
if [ -z "$pids" ]; then
  echo "No ZED processes running."
else
  echo "Stopping:"
  ps -o pid,stat,cmd -p "$(echo $pids | tr ' ' ,)" | cut -c1-160

  # Stop launch_all.sh first so it can't spawn a retry while we clean up.
  launcher=$(targets 'launch_all\.sh')
  [ -n "$launcher" ] && kill -TERM $launcher 2>/dev/null

  # SIGINT lets the ZED SDK close the cameras cleanly. SIGCONT wakes anything
  # suspended with Ctrl+Z, which would otherwise never see the signal.
  kill -CONT $pids 2>/dev/null
  kill -INT $pids 2>/dev/null
  for _ in $(seq 10); do
    [ -z "$(targets)" ] && break
    sleep 1
  done

  left=$(targets)
  if [ -n "$left" ]; then
    echo "Still running after 10s, force-killing:" $left
    kill -9 $left 2>/dev/null
    sleep 2
  fi
fi

left=$(targets)
if [ -n "$left" ]; then
  echo "NOT STOPPED:"
  ps -o pid,stat,cmd -p "$(echo $left | tr ' ' ,)"
  exit 1
fi

# Anything else holding a camera (e.g. ZED_Explorer) is reported, not killed.
if [ -n "$(fuser /dev/video* 2>/dev/null)" ]; then
  echo "Video devices still in use by other programs:"
  fuser -v /dev/video* 2>&1 | grep -v '^Cannot stat'
else
  echo "Cameras free."
fi
