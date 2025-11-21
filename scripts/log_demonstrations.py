import argparse
from time import sleep
from typing import Callable
import rospy
from sensor_msgs.msg import JointState, Image, CameraInfo
from deoxys.franka_interface import FrankaInterface
from deoxys.experimental.motion_utils import reset_joints_to
from deoxys.utils.config_utils import get_default_controller_config
from deoxys.utils.input_utils import input2action
from deoxys.utils.io_devices import SpaceMouse
import numpy as np
from pathlib import Path
import h5py
from datetime import datetime
import json
import cv2
import threading
from termcolor import cprint

from panda_utils.deoxys_controller import RESET_JOINT_POSITIONS
from panda_utils.utils import to_public_dict

""" This script starts teleoperation using the spacemouse, and provides a ui for logging robot demonstrations.


(hardware_env) resl@coldbrew:~/Desktop/panda_utils$ rostopic list
/clicked_point
/diagnostics
/initialpose
/move_base_simple/goal
/panda/joint_states
...
/realsense_south/color/camera_info
/realsense_south/color/image_raw
/realsense_south/color/metadata
/realsense_south/depth/camera_info
/realsense_south/depth/color/points
/realsense_south/depth/image_rect_raw
/realsense_south/depth/metadata
/realsense_north/color/camera_info
/realsense_north/color/image_raw
/realsense_north/color/metadata
/realsense_north/depth/camera_info
/realsense_north/depth/color/points
/realsense_north/depth/image_rect_raw
/realsense_north/depth/metadata
/realsense_eih/color/camera_info
/realsense_eih/color/image_raw
/realsense_eih/color/metadata
/realsense_eih/depth/camera_info
/realsense_eih/depth/color/points
/realsense_eih/depth/image_rect_raw
/realsense_eih/depth/metadata
...
/rosout
/rosout_agg
/tf
/tf_static

"""

CONTROLLER_TYPE = "OSC_POSE"
MISSING_CAMERA_DATA_ERROR_PRESENT = False


class DataCollector:
    """Collects camera images and joint states during demonstration."""

    def __init__(self, output_dir: Path, description: str, demo_index: int, camera_ids: list | None = None):
        if camera_ids is None:
            for cam_id in camera_ids:
                assert "_" not in cam_id, f"Camera id cannot contain underscore: {cam_id}"

        self.output_dir = (
            output_dir / f"{datetime.now().strftime('%m-%d_%H:%M:%S')}__{description}__demo_{demo_index:03d}"
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if camera_ids is None:
            camera_ids = [""]
            cprint("No camera ids were provided! Only joint states will be recorded", "yellow")
        self._camera_ids = camera_ids

        # Storage
        self.joint_states_msgs = []
        self.camera_data = {}
        for cam_id in camera_ids:
            self.camera_data[cam_id] = {
                "depth_images": [],
                "color_images": [],
                "depth_camera_info": None,
                "color_camera_info": None,
            }

        # Latest data
        self._latest_joint_state = None
        for cam_id in camera_ids:
            self.camera_data[cam_id]["_latest_depth"] = None
            self.camera_data[cam_id]["_latest_color"] = None

        # ROS subscribers
        self._subscribers = []
        self._joint_sub = rospy.Subscriber("/panda/joint_states", JointState, self._joint_callback)

        for cam_id in camera_ids:
            self._subscribers.append(
                rospy.Subscriber(
                    f"/realsense_{cam_id}/depth/image_rect_raw",
                    Image,
                    lambda msg, s=cam_id: self._depth_callback(msg, s),
                )
            )
            self._subscribers.append(
                rospy.Subscriber(
                    f"/realsense_{cam_id}/color/image_raw", Image, lambda msg, s=cam_id: self._color_callback(msg, s)
                )
            )
            self._subscribers.append(
                rospy.Subscriber(
                    f"/realsense_{cam_id}/depth/camera_info",
                    CameraInfo,
                    lambda msg, s=cam_id: self._depth_info_callback(msg, s),
                )
            )
            self._subscribers.append(
                rospy.Subscriber(
                    f"/realsense_{cam_id}/color/camera_info",
                    CameraInfo,
                    lambda msg, s=cam_id: self._color_info_callback(msg, s),
                )
            )
        self._is_recording = False

    @property
    def is_recording(self) -> bool:
        return self._is_recording

    def _joint_callback(self, msg):
        self._latest_joint_state = msg

    def _depth_callback(self, msg, cam_id):
        self.camera_data[cam_id]["_latest_depth"] = msg

    def _color_callback(self, msg, cam_id):
        self.camera_data[cam_id]["_latest_color"] = msg

    def _depth_info_callback(self, msg, cam_id):
        if self.camera_data[cam_id]["depth_camera_info"] is None:
            self.camera_data[cam_id]["depth_camera_info"] = msg

    def _color_info_callback(self, msg, cam_id):
        if self.camera_data[cam_id]["color_camera_info"] is None:
            self.camera_data[cam_id]["color_camera_info"] = msg

    def start_recording(self):
        """Start recording data."""
        self._is_recording = True
        self.joint_states_msgs = []
        for cam_id in self._camera_ids:
            self.camera_data[cam_id]["depth_images"] = []
            self.camera_data[cam_id]["color_images"] = []
        cprint("🎬 Started recording", "green")

    def record_sample(self) -> list[str]:
        """Record one sample of all data."""
        if not self.is_recording:
            return []

        # Record joint state
        if self._latest_joint_state is not None:
            self.joint_states_msgs.append(self._latest_joint_state)

        # Record camera data
        collected_camera_ids = []
        for cam_id in self._camera_ids:
            if self.camera_data[cam_id]["_latest_depth"] is not None:
                self.camera_data[cam_id]["depth_images"].append(self.camera_data[cam_id]["_latest_depth"])
                collected_camera_ids.append(cam_id)

            if self.camera_data[cam_id]["_latest_color"] is not None:
                self.camera_data[cam_id]["color_images"].append(self.camera_data[cam_id]["_latest_color"])
                assert (
                    collected_camera_ids[-1] == cam_id
                ), f"For some reason, depth wasn't read for {cam_id}, but color was"

        return collected_camera_ids

    def stop_recording(self):
        """Stop recording."""
        self._is_recording = False
        cprint("⏹️  Stopped recording", "yellow")

    def save(self):
        """Save collected data to disk."""
        if len(self.joint_states_msgs) == 0:
            cprint("⚠️  No data to save!", "red")
            return

        # Save HDF5
        h5_path = self.output_dir / "data.h5"
        with h5py.File(h5_path, "w") as f:
            # Joint states
            q = np.array([msg.position for msg in self.joint_states_msgs])
            dq = np.array([msg.velocity for msg in self.joint_states_msgs])
            f.create_dataset("joint_states_q", data=q)
            f.create_dataset("joint_states_dq", data=dq)

            # Camera data
            for cam_id in self._camera_ids:
                if len(self.camera_data[cam_id]["depth_images"]) > 0:
                    depth_msgs = self.camera_data[cam_id]["depth_images"]
                    color_msgs = self.camera_data[cam_id]["color_images"]

                    depth_images = np.array(
                        [np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width) for msg in depth_msgs]
                    )
                    color_images = np.array(
                        [
                            np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
                            for msg in color_msgs
                        ]
                    )
                    f.create_dataset(f"{cam_id}__depth_image", data=depth_images)
                    f.create_dataset(f"{cam_id}__color_image", data=color_images)

        # Save camera info
        for cam_id in self._camera_ids:
            if self.camera_data[cam_id]["depth_camera_info"] is not None:
                with open(self.output_dir / f"{cam_id}_camera_info_depth.json", "w") as f:
                    json.dump(to_public_dict(self.camera_data[cam_id]["depth_camera_info"]), f)
            if self.camera_data[cam_id]["color_camera_info"] is not None:
                with open(self.output_dir / f"{cam_id}_camera_info_color.json", "w") as f:
                    json.dump(to_public_dict(self.camera_data[cam_id]["color_camera_info"]), f)

        # Save sample images
        img_dir = self.output_dir / "images"
        img_dir.mkdir(exist_ok=True)

        cprint(f"\n📸 Saving sample images to {img_dir}...", "cyan")
        cprint(f"   Processing {len(self._camera_ids)} camera(s): {self._camera_ids}", "cyan")

        for cam_id in self._camera_ids:
            num_depth = len(self.camera_data[cam_id]["depth_images"])
            num_color = len(self.camera_data[cam_id]["color_images"])
            cprint(f"   Camera{cam_id}: {num_depth} depth, {num_color} color images collected", "white")

            if num_depth == 0:
                cprint(f"   ⚠️  No images collected for camera: '{cam_id}', skipping...", "yellow")
                continue

            depth_msgs = self.camera_data[cam_id]["depth_images"]
            color_msgs = self.camera_data[cam_id]["color_images"]

            # Convert messages to numpy arrays
            depth_images = []
            for msg in depth_msgs:
                img = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
                depth_images.append(img)
            depth_images = np.array(depth_images)

            color_images = []
            for msg in color_msgs:
                img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
                color_images.append(img)
            color_images = np.array(color_images)
            num_samples = len(depth_images)  # Save all images
            cprint(f"   Depth shape: {depth_images.shape}, Color shape: {color_images.shape}", "white")
            cprint(f"   Saving all {num_samples} images as PNG...", "white")

            for i in range(num_samples):
                # Save depth (already in uint16, good for PNG)
                depth_path = img_dir / f"{cam_id}__depth_{i:03d}.png"
                success = cv2.imwrite(str(depth_path), depth_images[i])
                assert success, f"Failed to save {depth_path.name}"

                # Convert RGB to BGR for OpenCV
                color_bgr = cv2.cvtColor(color_images[i], cv2.COLOR_RGB2BGR)
                color_path = img_dir / f"{cam_id}__color_{i:03d}.png"
                success = cv2.imwrite(str(color_path), color_bgr)
                assert success, f"Failed to save {color_path.name}"

        cprint(f"💾 Saved {len(self.joint_states_msgs)} samples to {h5_path}", "green")


def joint_publish_thread_target(robot_interface: FrankaInterface):
    """Publish joint states to ROS (from panda_teleop.py)."""
    pub = rospy.Publisher("panda/joint_states", JointState, queue_size=10)
    joint_names = [
        "panda_joint1",
        "panda_joint2",
        "panda_joint3",
        "panda_joint4",
        "panda_joint5",
        "panda_joint6",
        "panda_joint7",
        "panda_finger_joint1",
        "panda_finger_joint2",
    ]
    rate = rospy.Rate(50)

    counter = 0
    while not rospy.is_shutdown():
        rate.sleep()
        arm_q = robot_interface.last_q
        last_gripper_q = robot_interface.last_gripper_q

        if arm_q is None or last_gripper_q is None:
            continue

        gripper_angle = float(last_gripper_q) / 2.0
        js_msg = JointState()
        js_msg.header.stamp = rospy.Time.now()
        js_msg.name = joint_names
        js_msg.position = arm_q.tolist() + [gripper_angle, gripper_angle]
        js_msg.velocity = robot_interface.last_dq.tolist() + [0.0, 0.0]
        js_msg.effort = [0.0] * 9
        pub.publish(js_msg)

        counter += 1


def teleop_control_thread_target(
    robot_interface: FrankaInterface, device: SpaceMouse, should_continue: Callable[[], bool]
):
    """Teleop control loop (from panda_teleop.py) + data collection."""
    controller_cfg = get_default_controller_config(controller_type=CONTROLLER_TYPE)
    rate = rospy.Rate(50)

    cprint("🎮 Teleop active - use SpaceMouse to control robot", "cyan")
    cprint("   Data collection runs in background at 10Hz", "cyan")

    while not rospy.is_shutdown():
        rate.sleep()
        if not should_continue():
            continue

        try:
            action, grasp = input2action(device=device, controller_type=CONTROLLER_TYPE)
            robot_interface.control(
                controller_type=CONTROLLER_TYPE,
                action=action,
                controller_cfg=controller_cfg,
            )
        except Exception as e:
            rospy.logerr(f"Teleop error: {e}")
            break


def data_collection_thread_target(data_collector: DataCollector, rate_hz: float, camera_ids: list[str]):
    """Background thread to collect data at fixed rate."""
    rate = rospy.Rate(rate_hz)
    global MISSING_CAMERA_DATA_ERROR_PRESENT
    while not rospy.is_shutdown():
        rate.sleep()
        if not data_collector.is_recording:
            continue
        found_cams = data_collector.record_sample()
        missing_cams = [x for x in camera_ids if x not in found_cams]
        if len(missing_cams) > 0:
            MISSING_CAMERA_DATA_ERROR_PRESENT = True
            rospy.logerr(f"RGBD data is missing from specified cameras. Missing data from: {missing_cams}")
    rospy.loginfo("Data collection thread finished")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--description", type=str, default="")
    parser.add_argument("--recording_rate_hz", type=float, default=15.0)
    parser.add_argument(
        "--camera_ids", type=str, nargs="*", default=None, help="List of camera ids (e.g., north south eih)"
    )
    parser.add_argument(
        "--interface-cfg",
        type=str,
        default=None,
        help="Path to robot interface config (default: configs/charmander.yml)",
    )
    parser.add_argument("--vendor-id", type=int, default=9583)
    parser.add_argument("--product-id", type=int, default=50741)
    parser.add_argument("--no-teleop", action="store_true", help="Disable teleop (for kinesthetic teaching)")
    args = parser.parse_args()
    #
    rospy.init_node("log_demonstrations", anonymous=False)
    global MISSING_CAMERA_DATA_ERROR_PRESENT

    # Set default interface config if not provided
    project_dir = Path(__file__).parent.parent
    if args.interface_cfg is None:
        args.interface_cfg = f"{project_dir}/configs/charmander.yml"

    # Initialize SpaceMouse
    try:
        device = SpaceMouse(vendor_id=args.vendor_id, product_id=args.product_id)
        device.start_control()
    except OSError as e:
        cprint(
            f"❌ Failed to initialize SpaceMouse: {e}. You likely need to change the vendor-id and product-id.", "red"
        )
        exit(1)
    except Exception as e:
        cprint(f"❌ Unknown error starting SpaceMouse: {e}", "red")
        exit(1)

    #
    cprint("\n" + "=" * 60, "cyan")
    cprint("🤖 Demo Collection with Teleop", "cyan", attrs=["bold"])
    cprint("=" * 60, "cyan")
    cprint(f"Output: {args.output_dir}", "white")
    cprint(f"Mode: {'🎮 SpaceMouse Teleop' if device else '✋ Kinesthetic Teaching'}", "white")
    cprint("=" * 60 + "\n", "cyan")

    # Initialize robot interface once
    cprint("🔗 Initializing robot interface...", "cyan")
    try:
        robot_interface = FrankaInterface(args.interface_cfg, use_visualizer=False)
        robot_interface._state_buffer = []  # Clear buffer like original panda_teleop.py
        cprint("✅ Robot interface ready", "green")
    except Exception as e:
        cprint(f"❌ Failed to initialize robot interface: {e}", "red")
        return

    # Start joint state publisher once
    joint_thread = threading.Thread(target=joint_publish_thread_target, args=(robot_interface,))
    joint_thread.daemon = True
    joint_thread.start()

    # Start teleop control thread
    SHOULD_CONTINUE = True

    def should_continue():
        return SHOULD_CONTINUE

    teleop_thread = threading.Thread(
        target=teleop_control_thread_target,
        args=(robot_interface, device, should_continue),
    )
    teleop_thread.daemon = True
    teleop_thread.start()

    # Initialize classes collector
    demo_index = 0
    data_collector = DataCollector(Path(args.output_dir), args.description, demo_index, args.camera_ids)

    while not rospy.is_shutdown():
        cprint(f"\n{'='*60}", "cyan")
        cprint(f"📝 DEMO #{demo_index}", "cyan", attrs=["bold"])
        cprint(f"{'='*60}", "cyan")

        # Reset robot
        cprint("🔄 Resetting robot...", "yellow")
        SHOULD_CONTINUE = False
        sleep(0.1)
        reset_joints_to(robot_interface, RESET_JOINT_POSITIONS)
        SHOULD_CONTINUE = True

        # Start data collection thread
        data_thread = threading.Thread(
            target=data_collection_thread_target, args=(data_collector, args.recording_rate_hz, args.camera_ids)
        )
        data_thread.daemon = True
        data_thread.start()

        # Wait for user to start
        input("\n▶️  Press ENTER to START recording demo...")
        data_collector.start_recording()

        # Wait for user to finish
        input("\n⏸️  Press ENTER to STOP recording demo...")
        data_collector.stop_recording()

        if MISSING_CAMERA_DATA_ERROR_PRESENT:
            cprint("❌ Missing camera in last demonstration. Please check the logs for more details.", "red")
            return

        # Save data
        data_collector.save()

        cprint(f"✅ Demo #{demo_index} complete!", "green")

        # Continue?
        response = input("\n❓ Collect another demo? (y/n): ").strip().lower()
        if response != "y":
            break

        demo_index += 1

    # Cleanup (done once at the end)
    robot_interface.close()
    cprint(f"\n🏁 Collection complete! Total demos: {demo_index + 1}", "green", attrs=["bold"])
    rospy.signal_shutdown("Done")


if __name__ == "__main__":
    main()
