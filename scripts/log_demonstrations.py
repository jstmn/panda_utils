import argparse
import traceback
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

from panda_utils.deoxys_controller import DeoxysController, RESET_JOINT_POSITIONS
from panda_utils.utils import to_public_dict

""" This script starts teleoperation using the spacemouse, and provides a ui for logging robot demonstrations.
"""


class DataCollector:
    """Collects camera images and joint states during demonstration."""

    def __init__(self, output_dir: Path, description: str, demo_index: int, camera_suffixes: list = None):
        self.output_dir = (
            output_dir / f"{datetime.now().strftime('%m-%d_%H:%M:%S')}__{description}__demo_{demo_index:03d}"
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if camera_suffixes is None:
            camera_suffixes = [""]
            cprint("No camera suffixes provided! Only joint states will be recorded", "yellow")
        self.camera_suffixes = camera_suffixes

        # Storage
        self.joint_states_msgs = []
        self.camera_data = {}
        for suffix in camera_suffixes:
            self.camera_data[suffix] = {
                "depth_images": [],
                "color_images": [],
                "depth_camera_info": None,
                "color_camera_info": None,
            }

        # Latest data
        self._latest_joint_state = None
        for suffix in camera_suffixes:
            self.camera_data[suffix]["_latest_depth"] = None
            self.camera_data[suffix]["_latest_color"] = None

        # ROS subscribers
        self._subscribers = []
        self._joint_sub = rospy.Subscriber("/panda/joint_states", JointState, self._joint_callback)

        for suffix in camera_suffixes:
            prefix = f"/realsense{suffix}"
            self._subscribers.append(
                rospy.Subscriber(
                    f"{prefix}/depth/image_rect_raw", Image, lambda msg, s=suffix: self._depth_callback(msg, s)
                )
            )
            self._subscribers.append(
                rospy.Subscriber(f"{prefix}/color/image_raw", Image, lambda msg, s=suffix: self._color_callback(msg, s))
            )
            self._subscribers.append(
                rospy.Subscriber(
                    f"{prefix}/depth/camera_info", CameraInfo, lambda msg, s=suffix: self._depth_info_callback(msg, s)
                )
            )
            self._subscribers.append(
                rospy.Subscriber(
                    f"{prefix}/color/camera_info", CameraInfo, lambda msg, s=suffix: self._color_info_callback(msg, s)
                )
            )
        self.is_recording = False

    def _joint_callback(self, msg):
        self._latest_joint_state = msg

    def _depth_callback(self, msg, suffix):
        self.camera_data[suffix]["_latest_depth"] = msg

    def _color_callback(self, msg, suffix):
        self.camera_data[suffix]["_latest_color"] = msg

    def _depth_info_callback(self, msg, suffix):
        if self.camera_data[suffix]["depth_camera_info"] is None:
            self.camera_data[suffix]["depth_camera_info"] = msg

    def _color_info_callback(self, msg, suffix):
        if self.camera_data[suffix]["color_camera_info"] is None:
            self.camera_data[suffix]["color_camera_info"] = msg

    def start_recording(self):
        """Start recording data."""
        self.is_recording = True
        self.joint_states_msgs = []
        for suffix in self.camera_suffixes:
            self.camera_data[suffix]["depth_images"] = []
            self.camera_data[suffix]["color_images"] = []
        cprint("🎬 Started recording", "green")

    def record_sample(self):
        """Record one sample of all data."""
        if not self.is_recording:
            return

        # Record joint state
        if self._latest_joint_state is not None:
            self.joint_states_msgs.append(self._latest_joint_state)

        # Record camera data
        for suffix in self.camera_suffixes:
            if self.camera_data[suffix]["_latest_depth"] is not None:
                self.camera_data[suffix]["depth_images"].append(self.camera_data[suffix]["_latest_depth"])
            if self.camera_data[suffix]["_latest_color"] is not None:
                self.camera_data[suffix]["color_images"].append(self.camera_data[suffix]["_latest_color"])

    def stop_recording(self):
        """Stop recording."""
        self.is_recording = False
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
            for suffix in self.camera_suffixes:
                if len(self.camera_data[suffix]["depth_images"]) > 0:
                    depth_msgs = self.camera_data[suffix]["depth_images"]
                    color_msgs = self.camera_data[suffix]["color_images"]

                    depth_images = np.array(
                        [np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width) for msg in depth_msgs]
                    )
                    color_images = np.array(
                        [
                            np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
                            for msg in color_msgs
                        ]
                    )

                    # Use consistent naming: camera_north, camera_south, or camera (default)
                    if suffix:
                        prefix = f"camera{suffix}"  # suffix has underscore like "_north"
                    else:
                        prefix = "camera"

                    f.create_dataset(f"{prefix}_depth_image", data=depth_images)
                    f.create_dataset(f"{prefix}_color_image", data=color_images)

        # Save camera info
        for suffix in self.camera_suffixes:
            if self.camera_data[suffix]["depth_camera_info"] is not None:
                # Consistent naming for JSON files
                if suffix:
                    prefix = f"camera{suffix}"  # suffix has underscore like "_north"
                else:
                    prefix = "camera"

                with open(self.output_dir / f"{prefix}_camera_info_depth.json", "w") as f:
                    json.dump(to_public_dict(self.camera_data[suffix]["depth_camera_info"]), f)
                with open(self.output_dir / f"{prefix}_camera_info_color.json", "w") as f:
                    json.dump(to_public_dict(self.camera_data[suffix]["color_camera_info"]), f)

        # Save sample images
        img_dir = self.output_dir / "images"
        img_dir.mkdir(exist_ok=True)
        saved_count = 0

        cprint(f"\n📸 Saving sample images to {img_dir}...", "cyan")
        cprint(f"   Processing {len(self.camera_suffixes)} camera(s): {self.camera_suffixes}", "cyan")

        for suffix in self.camera_suffixes:
            num_depth = len(self.camera_data[suffix]["depth_images"])
            num_color = len(self.camera_data[suffix]["color_images"])
            cprint(f"   Camera{suffix}: {num_depth} depth, {num_color} color images collected", "white")

            if num_depth == 0:
                cprint(f"   ⚠️  No images collected for camera{suffix}, skipping...", "yellow")
                continue

            try:
                depth_msgs = self.camera_data[suffix]["depth_images"]
                color_msgs = self.camera_data[suffix]["color_images"]

                # Convert messages to numpy arrays
                cprint(f"   Converting camera{suffix} messages to arrays...", "white")
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

                cprint(f"   Depth shape: {depth_images.shape}, Color shape: {color_images.shape}", "white")

                # Determine prefix
                if suffix:
                    prefix = f"camera{suffix}_"  # suffix already has underscore like "_north"
                else:
                    prefix = "camera_"

                num_samples = len(depth_images)  # Save all images
                cprint(f"   Saving all {num_samples} images as PNG...", "white")

                for i in range(num_samples):
                    # Save depth (already in uint16, good for PNG)
                    depth_path = img_dir / f"{prefix}depth_{i}.png"
                    success = cv2.imwrite(str(depth_path), depth_images[i])
                    if success:
                        saved_count += 1
                        # cprint(f"      ✓ Saved {depth_path.name}", "green")
                    else:
                        cprint(f"      ✗ Failed to save {depth_path.name}", "red")

                    # Convert RGB to BGR for OpenCV
                    color_bgr = cv2.cvtColor(color_images[i], cv2.COLOR_RGB2BGR)
                    color_path = img_dir / f"{prefix}color_{i}.png"
                    success = cv2.imwrite(str(color_path), color_bgr)
                    if success:
                        saved_count += 1
                        # cprint(f"      ✓ Saved {color_path.name}", "green")
                    else:
                        cprint(f"      ✗ Failed to save {color_path.name}", "red")

            except Exception as e:
                cprint(f"   ❌ Error processing camera{suffix}: {e}", "red")
                traceback.print_exc()
                raise e

        cprint(f"💾 Saved {len(self.joint_states_msgs)} samples to {h5_path}", "green")
        if saved_count > 0:
            cprint(f"   📸 Saved {saved_count} sample color images to {img_dir}", "green")


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
    robot_interface: FrankaInterface, device: SpaceMouse, controller_type: str, data_collector: DataCollector
):
    """Teleop control loop (from panda_teleop.py) + data collection."""
    controller_cfg = get_default_controller_config(controller_type=controller_type)
    rate = rospy.Rate(50)

    cprint("🎮 Teleop active - use SpaceMouse to control robot", "cyan")
    cprint("   Data collection runs in background at 10Hz", "cyan")

    while not rospy.is_shutdown():
        rate.sleep()
        try:
            action, grasp = input2action(device=device, controller_type=controller_type)
            robot_interface.control(
                controller_type=controller_type,
                action=action,
                controller_cfg=controller_cfg,
            )
        except Exception as e:
            rospy.logerr(f"Teleop error: {e}")
            break


def data_collection_thread_target(data_collector: DataCollector):
    """Background thread to collect data at fixed rate."""
    rate = rospy.Rate(10)  # 10 Hz data collection

    while not rospy.is_shutdown():
        rate.sleep()
        data_collector.record_sample()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--description", type=str, default="")
    parser.add_argument(
        "--camera_suffixes", type=str, nargs="*", default=None, help="List of camera suffixes (e.g., _north _south)"
    )
    parser.add_argument(
        "--interface-cfg",
        type=str,
        default=None,
        help="Path to robot interface config (default: configs/charmander.yml)",
    )
    parser.add_argument("--controller-type", type=str, default="OSC_POSE")
    parser.add_argument("--vendor-id", type=int, default=9583)
    parser.add_argument("--product-id", type=int, default=50734)
    parser.add_argument("--no-teleop", action="store_true", help="Disable teleop (for kinesthetic teaching)")
    args = parser.parse_args()

    rospy.init_node("log_demo_ui", anonymous=False)
    project_dir = Path(__file__).parent.parent

    # Set default interface config if not provided
    if args.interface_cfg is None:
        args.interface_cfg = f"{project_dir}/configs/charmander.yml"

    # Initialize SpaceMouse
    device = None
    if not args.no_teleop:
        try:
            cprint("Initializing SpaceMouse...", "cyan")
            device = SpaceMouse(vendor_id=args.vendor_id, product_id=args.product_id)
            device.start_control()
            cprint("✅ SpaceMouse ready", "green")
        except Exception as e:
            cprint(f"❌ SpaceMouse failed: {e}", "red")
            response = input("Continue without teleop? (y/n): ").strip().lower()
            if response != "y":
                return

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

    demo_index = 0

    while not rospy.is_shutdown():
        cprint(f"\n{'='*60}", "cyan")
        cprint(f"📝 DEMO #{demo_index}", "cyan", attrs=["bold"])
        cprint(f"{'='*60}", "cyan")

        # Reset robot
        cprint("🔄 Resetting robot...", "yellow")
        try:
            deoxys_controller = DeoxysController(robot_interface, launch_viser=False, viser_use_visual=False)
            reset_joints_to(robot_interface, RESET_JOINT_POSITIONS)
            del deoxys_controller
            cprint("✅ Reset complete", "green")
        except Exception as e:
            cprint(f"❌ Reset failed: {e}", "red")
            continue

        # Initialize data collector
        data_collector = DataCollector(Path(args.output_dir), args.description, demo_index, args.camera_suffixes)

        # Start data collection thread
        data_thread = threading.Thread(target=data_collection_thread_target, args=(data_collector,))
        data_thread.daemon = True
        data_thread.start()

        # Start teleop if enabled
        if device is not None:
            teleop_thread = threading.Thread(
                target=teleop_control_thread_target,
                args=(robot_interface, device, args.controller_type, data_collector),
            )
            teleop_thread.daemon = True
            teleop_thread.start()
        else:
            cprint("✋ Move robot by hand (kinesthetic teaching)", "cyan")

        # Wait for user to start
        input("\n▶️  Press ENTER to START recording demo...")
        data_collector.start_recording()

        # Wait for user to finish
        input("\n⏸️  Press ENTER to STOP recording demo...")
        data_collector.stop_recording()

        # Stop teleop
        # if device is not None:
        #     controller_cfg = get_default_controller_config(controller_type=args.controller_type)
        #     robot_interface.control(
        #         controller_type=args.controller_type,
        #         action=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0] + [1.0],
        #         controller_cfg=controller_cfg,
        #         termination=True,
        #     )

        # Save data
        data_collector.save()

        cprint(f"✅ Demo #{demo_index} complete!", "green")

        # Continue?
        response = input("\n❓ Collect another demo? (y/n): ").strip().lower()
        if response != "y":
            break

        demo_index += 1

    # Cleanup (done once at the end)
    cprint("\n🔧 Cleaning up...", "cyan")
    if device is not None:
        device.stop_control()
    robot_interface.close()

    cprint(f"\n🏁 Collection complete! Total demos: {demo_index + 1}", "green", attrs=["bold"])
    rospy.signal_shutdown("Done")


if __name__ == "__main__":
    main()
