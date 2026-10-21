# Franka bring-up (this machine)

State of this workstation as of 2026-07-26, and the order things must be started in.

## Addresses

| Host | Address | Notes |
|---|---|---|
| GPU PC (this machine) | 172.16.0.5 | `enp6s0` |
| NUC | **172.16.0.8** | confirmed; `charmander.yml` previously said `.1` |
| Robot control box | 172.16.0.2 | **not reachable from this PC** — Desk must be opened from the NUC |

Both copies of `charmander.yml` (`panda_utils/configs/` and `deoxys_control/deoxys/config/`)
have been updated to `172.16.0.8`. If you edit one, edit the other.

## How control actually flows

It is **not** "GPU PC sends commands to the NUC over ssh". Deoxys is a two-process system:

```
this GPU PC (172.16.0.5)                NUC                     robot control box
  python FrankaInterface  --ZMQ-->  bin/franka-interface  --FCI/libfranka 1kHz--> arm
       (scripts here)      5555/5556   bin/gripper-interface   5557/5558
```

* **ZMQ**, not ssh, carries every control command. Ports come from `configs/charmander.yml`.
* **ssh is only used to log into the NUC and launch the two C++ binaries.** Once they run,
  ssh is irrelevant to control.
* The **SpaceMouse is read locally on this PC** over hidraw. It never touches the NUC.

The NUC needs a realtime kernel; this GPU PC does not.

## Startup order

Nothing works out of order — the FCI refuses connections if joints are locked, and
`franka-interface` exits if it cannot reach the arm.

### 1. Franka Desk — unlock joints, enable FCI

Desk is served by the **control box** (`ROBOT.IP`, 172.16.0.2), not the NUC.

> This box does not respond from this workstation. The NUC almost certainly holds the
> direct link to the control box, so open Desk from the NUC (or a machine on that link).

1. Open `https://172.16.0.2` and log in.
2. Release the E-stop (twist the red mushroom button out).
3. **Unlock joints** — the joint icons go from locked to open; you hear the brakes click.
4. Enable **FCI** (Activate FCI in the Desk sidebar). The status light turns blue.

Locked joints or an unactivated FCI is the single most common cause of
`franka-interface` failing at startup.

### 2. NUC — start the deoxys interfaces

Two binaries, two terminals, both must stay running. Note the path on the NUC is under
`~/Desktop/`, not `~/`:

```bash
# terminal 1 - arm
ssh robot@172.16.0.8
cd ~/Desktop/deoxys_control/deoxys
./auto_scripts/auto_arm.sh config/charmander.yml
```

```bash
# terminal 2 - gripper
ssh robot@172.16.0.8
cd ~/Desktop/deoxys_control/deoxys
./auto_scripts/auto_gripper.sh config/charmander.yml
```

`auto_arm.sh` warns if the CPU governor is not `performance` — heed it, powersave causes
realtime violations and dropped control cycles.

The NUC has its **own** `config/charmander.yml`, separate from the two copies on this PC.
Check its `NUC.IP` is also `172.16.0.8`:

```bash
ssh robot@172.16.0.8 "grep -A3 '^NUC:' ~/Desktop/deoxys_control/deoxys/config/charmander.yml"
```

### 3. This PC — verify, then move

deoxys is pip-installed editable into the `deoxys` conda env, so no `PYTHONPATH` export is
needed. `conda` is not initialized in `~/.bashrc`, so activate by path:

```bash
source ~/anaconda3/bin/activate deoxys
cd /home/res1/Desktop/panda_utils

python scripts/franka_manipulation.py --check          # no motion, just checks
python scripts/franka_manipulation.py --demo home
python scripts/franka_manipulation.py --demo square    # 10 cm square, Cartesian smoke test
python scripts/franka_manipulation.py --demo pick_place
```

Keep a hand on the E-stop the first time each demo runs.

## Open blockers

### ssh key not installed

Password auth only right now. Install the existing key so scripts and repeated logins
do not prompt:

```bash
ssh-copy-id -i ~/.ssh/id_ed25519.pub robot@172.16.0.8
```

### SpaceMouse needs a udev rule

`/dev/hidraw0` (the SpaceNavigator) is `root`-only, so `hid.open()` fails as a normal user.
Install the bundled rule and replug the device:

```bash
sudo cp /home/res1/Desktop/panda_utils/udev/99-spacemouse.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger
```

### SpaceMouse vendor/product IDs in `run_teleop.py` are wrong for this device

The connected device is `046d:c626` — a **Logitech-branded 3Dconnexion SpaceNavigator**:

| | vendor | product |
|---|---|---|
| `run_teleop.py` defaults | 9583 (`0x256F`) | 50741 (`0xC635`) |
| **this device** | **1133 (`0x046D`)** | **50726 (`0xC626`)** |

The defaults are for a 3Dconnexion-branded SpaceMouse Compact. Either pass
`--vendor-id 1133 --product-id 50726` or change the defaults before teleoperating.

### (resolved) the rig DOES have a gripper

Earlier notes here and in `log_demonstration.py` said no gripper was fitted. That is
wrong as of 2026-09-06: `bin/gripper-interface` runs on the NUC and
`FrankaInterface.last_gripper_q` reads `0.0804` (Franka Hand, fully open). Record
demonstrations WITHOUT `--no-gripper`, and convert with `--arm-dof 9` so the two finger
columns survive into `obs/qpos` -- Jrl2's `panda` actuates 9 joints and pins any the demo
omits at 0, rendering the gripper shut.

### ROS1 scripts cannot run here

`reset_panda.py` imports `rospy`; `panda_utils.utils` and
`panda_utils.deoxys_controller` do too (plus `tf.transformations`, `jrl2`, `open3d`).
**Only ROS2 Humble is installed** — there is no `/opt/ros/noetic`.

`scripts/log_demonstration.py` and `scripts/run_teleop.py` are ROS 2.
`scripts/franka_manipulation.py` is deliberately ROS-free so it runs today.
`src/panda_utils/rgb_subscriber_ros2_zed.py` is ROS 2 as well.

## What was changed to get this working

* **Generated the missing Python protobuf bindings.** `deoxys/proto/` did not exist, so
  `import deoxys.franka_interface` failed outright. Regenerate with:
  ```bash
  cd /home/res1/Desktop/deoxys_control/deoxys
  mkdir -p deoxys/proto/franka_interface
  protoc --proto_path=proto/franka-interface --python_out=deoxys/proto/franka_interface \
      proto/franka-interface/franka_controller.proto \
      proto/franka-interface/franka_robot_state.proto
  touch deoxys/proto/__init__.py deoxys/proto/franka_interface/__init__.py
  ```
* **Pinned `protobuf==3.20.0`** in the `deoxys` conda env (was 7.35.1, which rejects code
  from the installed protoc 3.13). This is what the main README prescribes. It prints
  conflict warnings for `google-api-core` / `proto-plus`, which deoxys does not use.
