"""ACT-CLIP policy inference with ZED cameras (ROS 2).

ZED port of scripts/act_clip_inference.py. Same policy, same temporal ensembling,
same gripper hysteresis, same JOINT_IMPEDANCE control - only the camera source and
the ROS version differ.

What changed vs act_clip_inference.py:
  * ROS 1 (rospy) -> ROS 2 (rclpy). Only Humble is installed on this machine, so
    rospy.Rate / rospy.is_shutdown / rospy.on_shutdown are replaced by a local Rate
    helper, an rclpy.ok() check and a finally-block shutdown.
  * panda_utils.rgb_subscriber.RGBSubscriber (RealSense, ROS 1) -> the ZedRGBSubscriber
    node below, subscribing to the topics that zed_tools/launch_all.sh publishes:
    `namespace:=zed_multi camera_name:=<name>` gives /zed_multi/<cam>/rgb/image_rect_color.
    NOTE: src/panda_utils/rgb_subscriber_ros2_zed.py uses the stock single-camera
    convention (/zed_<id>/zed_node/...) which does NOT match this rig's launcher.
    Override with --topic-template if your launch differs; --list-topics shows what is live.
  * ZED colour is bgra8; converted to uint8 RGB exactly as log_demonstration2.py does,
    so the images fed to the policy match the images that were recorded for training.
  * Does not import panda_utils.constants / .utils / .deoxys_controller: all three pull
    in ROS 1 (rospy) and open3d and are unimportable here. The few values needed
    (interface cfg path, reset joints, deoxys-ready wait) are inlined below.

CAMERA ORDER MATTERS. The training HDF5s store the cameras as `zed__color_image`,
`zed2i__color_image` and `zedm__color_image`, i.e. under the literal ZED names, and
TRAINING_CAMERA_ROLES below is that same order. Pass the names in it:

    --camera_ids zed zed2i zedm

(Earlier revisions of this docstring named the order ["eih", "base", "north"], carried
over from the RealSense rig. There are no such roles in this dataset, and the labels were
misleading besides: `zedm` is the wrist-mounted camera here, not the first entry.
scripts/zed_view_calibration.py --classify re-derives this from the data.)

The startup banner prints the resulting role -> ZED mapping; check it before trusting a
rollout. A silently permuted camera stack produces confident, wrong actions.

Cameras must already be running:
    bash ~/Desktop/ros2_ws/zed_tools/launch_all.sh
Robot must already be running on the NUC (see README_SETUP.md).

Usage:
    source /opt/ros/humble/setup.bash
    source ~/Desktop/ros2_ws/install/setup.bash
    source ~/anaconda3/bin/activate deoxys

    python scripts/act_clip_inference2.py --camera_ids zed zed2i zedm

    python scripts/act_clip_inference2.py --list-topics    # discover live camera topics
"""

from __future__ import annotations

import argparse
import select
import sys
import threading
import time
from pathlib import Path

import numpy as np
import rclpy
import torch
import torchvision.transforms as T
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from termcolor import cprint

from act_clip.backbone import build_backbone
from act_clip.transformer import build_transformer
from deoxys.experimental.motion_utils import reset_joints_to
from deoxys.franka_interface import FrankaInterface
from deoxys.utils.config_utils import get_default_controller_config
from act_clip.detr_vae import DETRVAE, build_encoder
# ^ NOTE: act_clip_inference.py imports DETRVAE as `scripts.act_clip.detr_vae` and the rest
#   as `act_clip.*`. There is only one act_clip on disk (scripts/act_clip), and the package's
#   own internal imports are all `act_clip.*` / `act.*`, i.e. `scripts/` is the intended
#   sys.path root - so the two spellings named the same file. Unified here; `scripts.` only
#   resolved when the repo root happened to be on sys.path as well.

torch.set_printoptions(precision=8, sci_mode=False, linewidth=200)
np.set_printoptions(precision=8, suppress=True, linewidth=200)


# --------------------------------------------------------------------------------------
# Constants inlined from panda_utils (that package imports rospy and is unusable on ROS 2)
# --------------------------------------------------------------------------------------

# panda_utils.deoxys_controller.RESET_JOINT_POSITIONS
RESET_JOINT_POSITIONS = [
    0.09162008, -0.19826458, -0.01990020, -2.47322699, -0.01307074, 2.30396583, 0.84809397,
]

STARTING_CONFIGS = {
    "cabinet".upper(): [
        0.55254022, 0.80662328, -0.43850005, -1.49100197, -1.17349812, 1.40865538, 2.8047689,
    ],
    "default".upper(): RESET_JOINT_POSITIONS,
}

# zed_tools/launch_all.sh launches with namespace:=zed_multi, camera_name:=<zed|zed2i|zedm>.
DEFAULT_TOPIC_TEMPLATE = "/zed_multi/{cam}"
COLOR_SUFFIX = "rgb/image_rect_color"

# The camera roles the checkpoint was trained with, in model-input order.
TRAINING_CAMERA_ROLES = ["zed", "zed2i", "zedm"]

CONTROLLER_TYPE = "JOINT_IMPEDANCE"  # THIS IS BETTER THAN JOINT_POSITION


class Args:
    hidden_dim: int = 512           # --hidden_dim 512
    dim_feedforward: int = 1600     # --dim_feedforward 1600
    enc_layers: int = 4             # --enc_layers 4
    dec_layers: int = 7             # --dec_layers 7
    num_queries: int = 30           # --num_queries 30

    lr_backbone: float = 1e-5
    masks: bool = False
    dilation: bool = False

    backbone: str = 'resnet18'
    position_embedding: str = 'sine'
    dropout: float = 0.1
    nheads: int = 8
    pre_norm: bool = False

    real: bool = True
    include_depth: bool = False
    kl_weight: float = 10.0

    checkpoint_path: str = "/home/res1/Desktop/panda_utils/scripts/download_data_0731/checkpoints/80000.pt"
    camera_ids: list[str] = ["zed", "zed2i", "zedm"]
    starting_qpos_alias: str = "default"

    
    #lang_instruction = "lift the peg and stand it upright on the table"
    #lang_instruction = "rotate the arrow lever to the target direction"
    #lang_instruction = "lift the cube up to a certain height"
    lang_instruction = "pick up a dish from the drying rack"
    #lang_instruction = "pick up the soda can from the cabinet"
    #lang_instruction = " "

# --------------------------------------------------------------------------------------
# ROS 2 helpers
# --------------------------------------------------------------------------------------

class Rate:
    """rospy.Rate replacement.

    rclpy.Rate needs a spinning executor to tick, and this script already runs its own
    spin thread for the camera node; a plain monotonic sleeper avoids that coupling and
    behaves the same as rospy.Rate for the loops here.
    """

    def __init__(self, hz: float):
        self._period = 1.0 / float(hz)
        self._next = time.perf_counter() + self._period

    def sleep(self) -> None:
        now = time.perf_counter()
        delay = self._next - now
        if delay > 0.0:
            time.sleep(delay)
            self._next += self._period
        else:
            # Overran the period: resynchronise instead of spiralling into catch-up.
            self._next = now + self._period


def make_qos(best_effort: bool) -> QoSProfile:
    """QoS for ZED image topics.

    The wrapper defaults to RELIABLE. A RELIABLE subscriber never receives from a
    BEST_EFFORT publisher, which shows up as "no data" with no error at all - hence the
    --qos-best-effort escape hatch.
    """
    return QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT if best_effort else ReliabilityPolicy.RELIABLE,
        history=HistoryPolicy.KEEP_LAST,
        depth=10,
        durability=DurabilityPolicy.VOLATILE,
    )


def decode_color(msg: Image) -> np.ndarray:
    """ZED colour Image -> HxWx3 uint8 RGB.

    Identical to log_demonstration2.decode_color so inference sees exactly the pixel
    convention the demonstrations were recorded in.
    """
    buf = np.frombuffer(msg.data, dtype=np.uint8)
    if msg.encoding in ("bgra8", "rgba8"):
        img = buf.reshape(msg.height, msg.width, 4)
        img = img[:, :, [2, 1, 0]] if msg.encoding == "bgra8" else img[:, :, :3]
    elif msg.encoding in ("bgr8", "rgb8"):
        img = buf.reshape(msg.height, msg.width, 3)
        if msg.encoding == "bgr8":
            img = img[:, :, ::-1]
    else:
        raise ValueError(f"unsupported colour encoding: {msg.encoding}")
    return np.ascontiguousarray(img)


class ZedRGBSubscriber(Node):
    """ROS 2 stand-in for panda_utils.rgb_subscriber.RGBSubscriber.

    Same surface the inference loop used before: latest_rgb(cam_id) returns the most
    recent frame or None.
    """

    def __init__(self, camera_ids: list[str], topic_template: str, best_effort: bool):
        super().__init__("act_clip_inference2_cameras")
        for cam_id in camera_ids:
            assert "_" not in cam_id, f"Camera id cannot contain underscore: {cam_id}"

        self._camera_ids = list(camera_ids)
        self._latest: dict[str, np.ndarray | None] = {c: None for c in camera_ids}
        self._lock = threading.Lock()

        qos = make_qos(best_effort)
        self._subscribers = []
        for cam_id in camera_ids:
            ns = topic_template.format(cam=cam_id).rstrip("/")
            topic = f"{ns}/{COLOR_SUFFIX}"
            self._subscribers.append(
                self.create_subscription(Image, topic,
                                         lambda m, c=cam_id: self._color_cb(m, c), qos)
            )
            cprint(f"   subscribed: {topic}", "white")

    def _color_cb(self, msg: Image, cam_id: str) -> None:
        try:
            img = decode_color(msg)
        except ValueError as exc:
            self.get_logger().warn(f"{cam_id}: {exc}")
            return
        with self._lock:
            self._latest[cam_id] = img

    def latest_rgb(self, cam_id: str) -> np.ndarray | None:
        with self._lock:
            return self._latest[cam_id]

    def wait_for_frames(self, timeout_s: float = 20.0) -> bool:
        cprint(f"⏳ Waiting for first colour frames from: {self._camera_ids}", "cyan")
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            missing = [c for c in self._camera_ids if self.latest_rgb(c) is None]
            if not missing:
                cprint("✅ All cameras streaming", "green")
                return True
            time.sleep(0.2)

        missing = [c for c in self._camera_ids if self.latest_rgb(c) is None]
        cprint(f"❌ No frames from: {missing} after {timeout_s:.0f}s", "red")
        cprint("   Check the cameras are up:  bash ~/Desktop/ros2_ws/zed_tools/verify.sh", "red")
        cprint("   Check topic names:         python scripts/act_clip_inference2.py --list-topics", "red")
        cprint("   If topics exist but no data arrives, retry with --qos-best-effort", "red")
        return False


def wait_for_deoxys_ready(robot: FrankaInterface, timeout_s: float = 10.0,
                          need_gripper: bool = True) -> bool:
    """Inlined panda_utils.utils.wait_for_deoxys_ready (that module imports rospy)."""
    cprint("🔗 Waiting for deoxys robot state...", "cyan")
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if robot.last_q is not None and (not need_gripper or robot.last_gripper_q is not None):
            cprint("✅ Robot interface ready", "green")
            return True
        time.sleep(0.1)
    cprint("❌ Deoxys is not streaming robot state. Is franka-interface running on the NUC?", "red")
    if need_gripper and robot.last_q is not None:
        cprint("   Arm state arrived but gripper state did not - is the gripper fitted "
               "and homed? Restart the franka via https://172.16.0.2/desk/", "red")
    return False


# --------------------------------------------------------------------------------------
# Policy (unchanged from act_clip_inference.py)
# --------------------------------------------------------------------------------------

class ACTRealInference:
    def __init__(self, ckpt_path, args, device="cuda"):
        self.device = torch.device(device)
        self.args = args

        ckpt = torch.load(ckpt_path, map_location=self.device)

        self.stats = ckpt['norm_stats']
        self.stats['state_std'] = torch.clip(self.stats['state_std'], min=1e-2)
        self.stats['action_std'] = torch.clip(self.stats['action_std'], min=1e-2)

        self.stats = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in self.stats.items()}
        self.agent = Agent_Inference(args).to(self.device)
        self.agent.load_state_dict(ckpt['ema_agent'])
        self.agent.eval()

        self.resize = T.Resize((224, 224), antialias=True)
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.num_queries = args.num_queries
        self.step_idx = 0
        self.max_timesteps = 10000

        self.state_dim = 18
        self.act_dim = 9
        self.all_time_actions = torch.zeros(
            [self.max_timesteps, self.max_timesteps + self.num_queries, self.act_dim]
        ).to(self.device)

    def preprocess(self, rgb_imgs):
        imgs = []
        for rgb_img in rgb_imgs:
            img_t = torch.from_numpy(rgb_img.copy()).permute(2, 0, 1).float()  # (3, H, W)
            img_t = self.resize(img_t) / 255.0
            img_t = self.normalize(img_t)
            imgs.append(img_t)
        imgs = torch.stack(imgs, dim=0)
        return imgs.unsqueeze(0).to(self.device)

    def get_real_action(self, rgb_img, robot_state):
        state_t = torch.from_numpy(robot_state).float().to(self.device)
        state_t = (state_t - self.stats["state_mean"][0]) / self.stats["state_std"][0]
        state_t = state_t.unsqueeze(0)
        obs = {'rgb': self.preprocess(rgb_img), 'state': state_t}
        with torch.no_grad():
            action_seq = self.agent.get_action(obs)

        ts = self.step_idx

        device = self.device
        num_queries = self.num_queries

        try:
            self.all_time_actions[ts, ts:ts + num_queries] = action_seq[0]
        except IndexError as e:
            print("we got the error yo")
            print(f"ts: {ts}, ts+num_queries: {ts+num_queries}")
            print(f"action_seq:               {action_seq.shape}")
            print(f"self.all_time_actions:    {self.all_time_actions.shape}")
            raise e

        actions_for_curr_step = self.all_time_actions[:, ts]  # (max_timesteps, act_dim)

        # actions_populated
        actions_populated = torch.zeros(self.max_timesteps, dtype=torch.bool, device=device)
        actions_populated[max(0, ts + 1 - num_queries):ts + 1] = True

        actions_for_curr_step = actions_for_curr_step[actions_populated]  # (num_populated, act_dim)

        k = 0.1
        exp_weights = torch.exp(-k * torch.arange(len(actions_for_curr_step), device=device))
        exp_weights = exp_weights / exp_weights.sum()
        exp_weights = exp_weights.unsqueeze(-1)  # (num_populated, 1)

        raw_action = (actions_for_curr_step * exp_weights).sum(dim=0)  # (act_dim,)

        # 3. Post-process (De-normalization)
        action = raw_action * self.stats['action_std'][0] + self.stats['action_mean'][0]
        self.step_idx += 1
        return action.cpu().numpy().flatten()


class Agent_Inference(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.state_dim = 18
        self.act_dim = 9

        backbones = [build_backbone(args)]
        transformer = build_transformer(args)
        encoder = build_encoder(args)

        self._lang_instruction = args.lang_instruction
        if self._lang_instruction is not None:
            use_lang = True
        else:
            use_lang = False

        self.model = DETRVAE(
            backbones, transformer, encoder,
            state_dim=self.state_dim, action_dim=self.act_dim,
            num_queries=args.num_queries,
            use_lang_instruction=use_lang,
        )

    def get_action(self, obs):
        a_hat, _ = self.model(obs, lang_instruction=self._lang_instruction)
        return a_hat


def enter_is_pressed() -> bool:
    if select.select([sys.stdin], [], [], 0.0)[0] == [sys.stdin]:
        key = sys.stdin.read(1)
        if key == "\n":
            return True
    return False


def np_array_to_hash(arr: np.ndarray) -> int:
    """ For a rgb array, the sum of the array is a good hash.
    """
    return np.sum(arr)


def list_topics() -> int:
    """Print live camera-ish topics so topic-name mismatches are obvious."""
    rclpy.init(args=None)
    node = rclpy.create_node("act_clip_inference2_topic_lister")
    time.sleep(2.0)  # discovery needs a moment
    names = sorted(n for n, _ in node.get_topic_names_and_types())
    cam = [n for n in names if any(k in n.lower() for k in ("zed", "image", "camera_info"))]
    cprint("\nLive camera-related topics:", "cyan")
    if cam:
        for n in cam:
            print(f"   {n}")
    else:
        cprint("   (none found - are the cameras running?)", "yellow")
        cprint("   bash ~/Desktop/ros2_ws/zed_tools/launch_all.sh", "yellow")
    cprint(f"\nExpected pattern: {DEFAULT_TOPIC_TEMPLATE.format(cam='<id>')}/{COLOR_SUFFIX}", "cyan")
    cprint("Override with --topic-template if yours differ.\n", "cyan")
    node.destroy_node()
    rclpy.shutdown()
    return 0


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camera_ids", type=str, nargs="*", default=None,
                        help=("ZED camera names as launched, IN THE MODEL'S CAMERA ORDER "
                              f"({', '.join(TRAINING_CAMERA_ROLES)}). e.g. zed zed2i zedm"))
    parser.add_argument("--topic-template", type=str, default=DEFAULT_TOPIC_TEMPLATE,
                        help=f"namespace template, '{{cam}}' substituted (default: {DEFAULT_TOPIC_TEMPLATE})")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="override Args.checkpoint_path")
    parser.add_argument("--lang_instruction", type=str, default=None,
                        help="override Args.lang_instruction")
    parser.add_argument("--starting_qpos_alias", type=str, default=None,
                        help=("override Args.starting_qpos_alias; one of: "
                              f"{', '.join(sorted(STARTING_CONFIGS))}"))
    parser.add_argument("--interface-cfg", type=str, default=None,
                        help="deoxys interface yaml (default: <project>/configs/charmander.yml)")
    parser.add_argument("--rate", type=float, default=15.0,
                        help="inference and control rate in Hz (default: 15)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no-gripper", action="store_true",
                        help=("rig has no gripper fitted: do not wait for or command it. "
                              "The 18-dim state still needs 2 gripper slots, filled with "
                              "--gripper-fill-value."))
    parser.add_argument("--gripper-fill-value", type=float, default=0.04,
                        help="gripper finger position used in the state vector when --no-gripper")
    parser.add_argument("--qos-best-effort", action="store_true",
                        help="use BEST_EFFORT QoS if topics exist but no frames arrive")
    parser.add_argument("--list-topics", action="store_true", help="list live topics and exit")
    cli = parser.parse_args()

    if cli.list_topics:
        return list_topics()

    args = Args()
    if cli.camera_ids:
        args.camera_ids = list(cli.camera_ids)
    if cli.checkpoint_path:
        args.checkpoint_path = cli.checkpoint_path
    if cli.lang_instruction:
        args.lang_instruction = cli.lang_instruction
    if cli.starting_qpos_alias:
        args.starting_qpos_alias = cli.starting_qpos_alias

    has_gripper = not cli.no_gripper
    camera_ids = args.camera_ids

    if len(camera_ids) != len(TRAINING_CAMERA_ROLES):
        cprint(f"⚠️  {len(camera_ids)} cameras given but the checkpoint was trained with "
               f"{len(TRAINING_CAMERA_ROLES)} ({', '.join(TRAINING_CAMERA_ROLES)}). "
               "The policy will almost certainly misbehave.", "red")

    project_dir = Path(__file__).parent.parent
    interface_cfg = cli.interface_cfg or f"{project_dir}/configs/charmander.yml"

    cprint("\n" + "=" * 60, "cyan")
    cprint("🤖 ACT-CLIP Inference (ZED / ROS 2)", "cyan", attrs=["bold"])
    cprint("=" * 60, "cyan")
    cprint(f"Checkpoint : {args.checkpoint_path}", "white")
    cprint(f"Instruction: {args.lang_instruction}", "white")
    cprint(f"Start pose : {args.starting_qpos_alias}", "white")
    cprint(f"Rate       : {cli.rate} Hz", "white")
    cprint(f"Gripper    : {'enabled' if has_gripper else 'DISABLED (--no-gripper)'}", "white")
    cprint("Camera order (model input -> ZED camera):", "white")
    for i, cam_id in enumerate(camera_ids):
        role = TRAINING_CAMERA_ROLES[i] if i < len(TRAINING_CAMERA_ROLES) else f"extra{i}"
        cprint(f"   [{i}] {role:<6} -> {cam_id}", "yellow")
    cprint("=" * 60 + "\n", "cyan")

    rclpy.init(args=None)

    running = True
    RESETTING = False
    SHUTDOWN = False
    # Shared with the control thread. A one-element list keeps the write visible to the
    # thread without a `global` in a function-scoped main().
    action_holder: list[np.ndarray | None] = [None]

    rgb_subscriber = ZedRGBSubscriber(camera_ids, cli.topic_template, cli.qos_best_effort)

    def spin_target() -> None:
        while running and rclpy.ok():
            rclpy.spin_once(rgb_subscriber, timeout_sec=0.05)

    spin_thread = threading.Thread(target=spin_target, daemon=True)
    spin_thread.start()

    def stop_spin() -> None:
        """Stop the spin thread and wait for it to leave spin_once.

        destroy_node() while the thread is inside spin_once() frees the wait set out from
        under it - rclpy raises InvalidHandle from the daemon thread, or rcl touches freed
        memory. The join must happen first.
        """
        nonlocal running
        running = False
        spin_thread.join(timeout=2.0)

    if not rgb_subscriber.wait_for_frames():
        stop_spin()
        rgb_subscriber.destroy_node()
        rclpy.shutdown()
        return 1

    inference_node = ACTRealInference(args.checkpoint_path, args, device=cli.device)
    franka_interface = FrankaInterface(interface_cfg, use_visualizer=False)
    controller_cfg = get_default_controller_config(controller_type=CONTROLLER_TYPE)
    # from deoxys.utils.yaml_config import YamlConfig
    # controller_cfg = YamlConfig("configs/joint-impedance-controller.yml").as_easydict()

    if not wait_for_deoxys_ready(franka_interface, need_gripper=has_gripper):
        stop_spin()
        franka_interface.close()
        rgb_subscriber.destroy_node()
        rclpy.shutdown()
        return 1

    reset_joints_to(franka_interface, STARTING_CONFIGS[args.starting_qpos_alias.upper()],
                    gripper_open=True)

    cprint("Press ENTER to reset the robot and reload the model", "yellow")

    CONTROL_RATE = Rate(cli.rate)
    INFERENCE_RATE = Rate(cli.rate)

    # Set by the control thread when it dies on an exception, so main() can exit non-zero
    # instead of reporting a rollout that stopped being commanded as a success.
    control_failed = [False]

    def control_loop():
        nonlocal SHUTDOWN

        # Binary open/close with hysteresis. Peg grasps sit ~0.025, so the reopen
        # threshold must be ABOVE the close threshold (old 0.022 reopen caused
        # close-then-immediate-open chatter).
        THRESHOLD_CLOSE = 0.0325   # while open:  action below this -> close
        THRESHOLD_REOPEN = 0.036   # while closed: action above this -> open
        THRESHOLD_STATE_OPEN = 0.0325  # measured finger width treated as "open"
        GRIPPER_CLOSE_CMD = 1
        GRIPPER_OPEN_CMD = -1

        while not SHUTDOWN and rclpy.ok():
            CONTROL_RATE.sleep()

            action = action_holder[0]
            if action is None or RESETTING:
                continue

            action_cp = action.copy()

            try:
                if has_gripper:
                    gripper_left_act = action_cp[7]
                    gripper_right_act = action_cp[8]
                    assert -0.001 < gripper_left_act < 0.041, f"gripper_left_act: {gripper_left_act} should be between -0.001 and 0.041"
                    assert -0.001 < gripper_right_act < 0.041, f"gripper_right_act: {gripper_right_act} should be between -0.001 and 0.041"

                    gripper_act = max(gripper_left_act, gripper_right_act)
                    gripper_state = float(franka_interface.last_gripper_q) / 2.0
                    currently_open = gripper_state > THRESHOLD_STATE_OPEN
                    # print(f"gripper_act: ({gripper_left_act:0.5f} {gripper_right_act:0.5f})\tgripper_pos: {gripper_state:0.4f}\tcurrently_open: {currently_open}", end="\t")

                    if currently_open:
                        if gripper_act < THRESHOLD_CLOSE:
                            print("(open) --> closing")
                            gripper_cmd = GRIPPER_CLOSE_CMD
                        else:
                            print("(open) --> staying open")
                            gripper_cmd = GRIPPER_OPEN_CMD
                    else:
                        if gripper_act > THRESHOLD_REOPEN:
                            print("(closed) --> opening")
                            gripper_cmd = GRIPPER_OPEN_CMD
                        else:
                            print("(closed) --> staying closed")
                            gripper_cmd = GRIPPER_CLOSE_CMD
                else:
                    # No gripper fitted: deoxys still expects an 8th element, keep it open.
                    gripper_cmd = GRIPPER_OPEN_CMD

                deoxys_action = np.concatenate([action_cp[0:7], [gripper_cmd]], axis=0)

                franka_interface.control(
                    controller_type=CONTROLLER_TYPE,
                    action=deoxys_action,
                    controller_cfg=controller_cfg,
                )
            except Exception as e:
                cprint(f"[act_inference] control_loop exception: {e!r}", "red")
                control_failed[0] = True
                SHUTDOWN = True
                break
            
            current_q = franka_interface.last_q
            desired_q = deoxys_action[0:7]
            q_delta_deg = np.degrees(desired_q - current_q)
            print(f"q_delta_deg: {q_delta_deg}")

    control_thread = threading.Thread(target=control_loop, daemon=True)
    control_thread.start()

    camera_rgb_pseudo_hashes = {cid: 0 for cid in camera_ids}
    pseudo_hash_match_counts = {cid: 0 for cid in camera_ids}

    exit_code = 0
    cprint("Running policy 🏁", "yellow")

    try:
        while rclpy.ok() and not SHUTDOWN:
            rgbs = [rgb_subscriber.latest_rgb(cid) for cid in camera_ids]
            if any(rgb is None for rgb in rgbs):
                assert False, "rgb is None"

            # Verify that the rgb images are changing
            for cid, rgb in zip(camera_ids, rgbs):
                pseudo_hash = np_array_to_hash(rgb)
                if pseudo_hash == camera_rgb_pseudo_hashes[cid]:
                    pseudo_hash_match_counts[cid] += 1
                else:
                    pseudo_hash_match_counts[cid] = 0
                camera_rgb_pseudo_hashes[cid] = pseudo_hash
                if pseudo_hash_match_counts[cid] == 10:
                    raise RuntimeError(f"Camera {cid} hasn't changed for 10 frames")

            arm_q = franka_interface.last_q
            arm_dq = franka_interface.last_dq
            last_gripper_q = franka_interface.last_gripper_q
            if arm_q is None or arm_dq is None or (has_gripper and last_gripper_q is None):
                print(
                    f"arm_q: {arm_q} is None, arm_dq: {arm_dq} is None, "
                    f"last_gripper_q: {last_gripper_q} is None"
                )
                INFERENCE_RATE.sleep()
                continue

            gripper_pos = (float(last_gripper_q) / 2.0) if has_gripper else cli.gripper_fill_value
            last_q = np.concatenate([arm_q, [gripper_pos, gripper_pos]])
            last_dq = np.concatenate([arm_dq, [0.0, 0.0]])
            # ^ NOTE: Gripper velocity is set to [0, 0] by log_demonstration2.py
            state = np.concatenate([last_q, last_dq]).astype(np.float32)
            action_holder[0] = inference_node.get_real_action(rgbs, state)

            if enter_is_pressed():
                cprint(" --> Resetting", "yellow")
                RESETTING = True
                reset_joints_to(franka_interface,
                                STARTING_CONFIGS[args.starting_qpos_alias.upper()],
                                gripper_open=True)
                cprint("Press ENTER to continue", "yellow")
                inference_node = ACTRealInference(args.checkpoint_path, args, device=cli.device)
                action_holder[0] = None
                while not enter_is_pressed():
                    INFERENCE_RATE.sleep()
                RESETTING = False
                cprint("Running policy 🏁", "yellow")
                continue

            INFERENCE_RATE.sleep()

    except KeyboardInterrupt:
        cprint("\n⏹️  Interrupted", "yellow")
    except Exception as exc:  # noqa: BLE001 - report, then always tear the robot down cleanly
        cprint(f"❌ {exc!r}", "red")
        exit_code = 1
    finally:
        cprint("Shutting down", "red")
        SHUTDOWN = True
        if control_failed[0]:
            # The arm stopped being commanded mid-rollout. Never report that as success:
            # a supervising script would otherwise log the rollout as a clean run.
            cprint("❌ Control thread died - the arm was left holding its last target.", "red")
            exit_code = 1
        try:
            control_thread.join(timeout=2.0)
        except Exception:  # noqa: BLE001 - never mask the original failure during cleanup
            pass
        try:
            franka_interface.close()
        except Exception:  # noqa: BLE001
            pass
        stop_spin()
        rgb_subscriber.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
