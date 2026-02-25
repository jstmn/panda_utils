import rospy
import threading
import torch
import select
import sys
import numpy as np
import torchvision.transforms as T
from std_msgs.msg import Float64MultiArray
from time import sleep, time
from scripts.act_clip.detr_vae import DETRVAE
from act_clip.backbone import build_backbone
from act_clip.transformer import build_transformer
from act_clip.detr_vae import build_encoder
from deoxys.utils.yaml_config import YamlConfig
import os
from datetime import datetime
import cv2
from termcolor import cprint

from deoxys.franka_interface import FrankaInterface
from deoxys.utils.config_utils import get_default_controller_config
from panda_utils.rgb_subscriber import RGBSubscriber
from panda_utils.constants import DEFAULT_DEOXYS_INTERFACE_CFG
from panda_utils.deoxys_controller import RESET_JOINT_POSITIONS
from panda_utils.utils import wait_for_deoxys_ready
from deoxys.experimental.motion_utils import reset_joints_to

torch.set_printoptions(precision=8, sci_mode=False, linewidth=200)
np.set_printoptions(precision=8, suppress=True, linewidth=200)


"""
python scripts/act_clip_inference.py
"""


STARTING_CONFIGS = {
    "OpenDrawer".upper(): [0.35648074,  0.53879886,  0.16414258, -1.67889499, -1.5206821,  1.23415236,  1.51561697],
    "default".upper(): RESET_JOINT_POSITIONS,
}

class Args:
    hidden_dim: int = 512           # --hidden_dim 512
    dim_feedforward: int = 1600     # --dim_feedforward 1600
    enc_layers: int = 4             # --enc_layers 4
    dec_layers: int = 7             # --dec_layers 7
    num_queries: int = 30           # --num_queries 30

    lr_backbone: float = 1e-5
    masks:bool = False
    dilation:bool = False

    backbone: str = 'resnet18'
    position_embedding: str = 'sine'
    masks: bool = False
    dilation: bool = False
    dropout: float = 0.1
    nheads: int = 8
    pre_norm: bool = False

    real: bool = True
    include_depth: bool = False
    kl_weight: float = 10.0


    #checkpoint_path: str = "scripts/checkpoints_multiview/hyeonho_real_results/act_real_multi-task_lang-15368560/checkpoints/100000.pt"
    checkpoint_path: str = "scripts/checkpoints_multiview/hyeonho_real_results/act_real_multi-task_lang-15368560/checkpoints/150000.pt"
    camera_ids: list[str] = ["eih", "base", "north"]
    starting_qpos_alias: str = "OpenDrawer"

    #lang_instruction = "stack red cube on top of green cube"
    #lang_instruction = "rotate the arrow lever to the target direction"
    # lang_instruction = "lift the cube up to a certain height"
    lang_instruction = "grasp the handle and pull the drawer open"
    # lang_instruction = "lift the peg and stand it upright on the table"

class ACTRealInference:
    def __init__(self, ckpt_path, args, device="cuda"):
        self.device = torch.device(device)
        self.args = args 

        ckpt = torch.load(ckpt_path, map_location=self.device)
        # stats = ckpt["norm_stats"]

        #self.stats = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in ckpt['norm_stats'].items()}
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
        self.all_time_actions = torch.zeros([self.max_timesteps, self.max_timesteps + self.num_queries,self.act_dim]).to(self.device)

    def preprocess(self, rgb_imgs):

        imgs =[]
        for rgb_img in rgb_imgs:
            img_t = torch.from_numpy(rgb_img.copy()).permute(2, 0, 1).float() # (3, H, W)
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
        # print(f"{obs['state']}")
        with torch.no_grad():
            action_seq = self.agent.get_action(obs)

        ts = self.step_idx

        device = self.device
        num_queries = self.num_queries

        try:
            self.all_time_actions[ts, ts:ts+num_queries] = action_seq[0]
        except IndexError as e:
            print("we got the error yo")
            print(f"ts: {ts}, ts+num_queries: {ts+num_queries}")
            print(f"action_seq:               {action_seq.shape}")
            print(f"self.all_time_actions:    {self.all_time_actions.shape}")
            raise e

        actions_for_curr_step = self.all_time_actions[:, ts] # (max_timesteps, act_dim)

        # actions_populated
        actions_populated = torch.zeros(self.max_timesteps, dtype=torch.bool, device=device)
        actions_populated[max(0, ts + 1 - num_queries):ts+1] = True

        actions_for_curr_step = actions_for_curr_step[actions_populated] # (num_populated, act_dim)

        k = 0.1
        exp_weights = torch.exp(-k * torch.arange(len(actions_for_curr_step), device=device))
        exp_weights = exp_weights / exp_weights.sum()
        exp_weights = exp_weights.unsqueeze(-1) # (num_populated, 1)

        raw_action = (actions_for_curr_step * exp_weights).sum(dim=0) # (act_dim,)

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

if __name__ == "__main__":
    rospy.init_node("maniskill_act_inference", anonymous=False)

    args = Args()
    rgb_subscriber = RGBSubscriber(camera_ids=args.camera_ids)
    inference_node = ACTRealInference(args.checkpoint_path, args)
    franka_interface = FrankaInterface(DEFAULT_DEOXYS_INTERFACE_CFG, use_visualizer=False)
    controller_type = "JOINT_IMPEDANCE" # THIS IS BETTER THAN JOINT_POSITION
    controller_cfg = get_default_controller_config(controller_type=controller_type)

    wait_for_deoxys_ready(franka_interface)
    reset_joints_to(franka_interface, STARTING_CONFIGS[args.starting_qpos_alias.upper()], gripper_open=True)

    RESETTING = False
    SHUTDOWN = False
    action = None

    def _on_shutdown():
        global SHUTDOWN
        SHUTDOWN = True

    rospy.on_shutdown(_on_shutdown)
    cprint("Press ENTER to reset the robot and reload the model", "yellow")

    def control_loop():
        global SHUTDOWN, RESETTING
        while franka_interface.last_gripper_q is None:
            sleep(0.01)

        THRESHOLD_OPEN = 0.0325
        GRIPPER_CLOSE_CMD = 1
        GRIPPER_OPEN_CMD = -1

        while not SHUTDOWN and not rospy.is_shutdown():
            try:
                CONTROL_RATE.sleep()
            except rospy.ROSInterruptException:
                break

            if action is None or RESETTING:
                continue

            action_cp = action.copy()

            try:
                gripper_left_act = action_cp[7]
                gripper_right_act = action_cp[8]
                assert -0.001 < gripper_left_act < 0.041, f"gripper_left_act: {gripper_left_act} should be between -0.001 and 0.041"
                assert -0.001 < gripper_right_act < 0.041, f"gripper_right_act: {gripper_right_act} should be between -0.001 and 0.041"

                gripper_state = float(franka_interface.last_gripper_q) / 2.0
                currently_open = gripper_state > THRESHOLD_OPEN
                print(f"gripper_act: ({gripper_left_act:0.5f} {gripper_right_act:0.5f})\tgripper_pos: {gripper_state:0.4f}\tcurrently_open: {currently_open}", end="\t")

                if currently_open:
                    if gripper_left_act < THRESHOLD_OPEN or gripper_right_act < THRESHOLD_OPEN:
                        print("(open) --> closing")
                        gripper_cmd = GRIPPER_CLOSE_CMD
                    else:
                        print("(open) --> staying open")
                        gripper_cmd = GRIPPER_OPEN_CMD
                else:
                    if gripper_left_act > 0.022 or gripper_right_act > 0.022:
                        print("(closed) --> opening")
                        gripper_cmd = GRIPPER_OPEN_CMD
                    else:
                        print("(closed) --> staying closed")
                        gripper_cmd = GRIPPER_CLOSE_CMD

                deoxys_action = np.concatenate([action_cp[0:7], [gripper_cmd]], axis=0)

                franka_interface.control(
                    controller_type=controller_type,
                    action=deoxys_action,
                    controller_cfg=controller_cfg,
                )
                # gripper_left = action_cp[7]
                # gripper_right_act = action_cp[8]
                # assert -0.001 < gripper_left < 0.041, f"gripper_left: {gripper_left} should be between -0.001 and 0.041"
                # assert -0.001 < gripper_right_act < 0.041, f"gripper_right_act: {gripper_right_act} should be between -0.001 and 0.041"
                # print("gripper_left/right: ", gripper_left, gripper_right_act)

                # if gripper_left < THRESHOLD_OPEN or gripper_right_act < THRESHOLD_OPEN:
                #     gripper_cmd = GRIPPER_CLOSE_CMD
                # else:
                #     gripper_cmd = GRIPPER_OPEN_CMD
                # deoxys_action = np.concatenate([action_cp[0:7], [gripper_cmd]], axis=0)

                # franka_interface.control(
                #     controller_type=controller_type,
                #     action=deoxys_action,
                #     controller_cfg=controller_cfg,
                # )
            except Exception as e:
                print(f"[act_inference] control_loop exception: {e!r}")
                SHUTDOWN = True
                break

    r = 15
    INFERENCE_RATE = rospy.Rate(r)
    CONTROL_RATE = rospy.Rate(r)
    control_thread = threading.Thread(target=control_loop, daemon=True)
    control_thread.start()

    prev_gripper_pos = None
    prev_time = None

    def shutdown():
        cprint("Shutting down", "red")
        global SHUTDOWN
        SHUTDOWN = True
        try:
            control_thread.join(timeout=2.0)
        except Exception:
            pass
        try:
            franka_interface.close()
        except Exception:
            pass


    camera_rgb_pseudo_hashes = {cid: 0 for cid in args.camera_ids}
    pseudo_hash_match_counts = {cid: 0 for cid in args.camera_ids}

    while not rospy.is_shutdown():
        rgbs = [rgb_subscriber.latest_rgb(cid) for cid in args.camera_ids]
        if any(rgb is None for rgb in rgbs):
            assert False, "rgb is None"


        # Verify that the rgb images are changing
        for cid, rgb in zip(args.camera_ids, rgbs):
            pseudo_hash = np_array_to_hash(rgb)
            if pseudo_hash == camera_rgb_pseudo_hashes[cid]:
                pseudo_hash_match_counts[cid] += 1
            else:
                pseudo_hash_match_counts[cid] = 0
            camera_rgb_pseudo_hashes[cid] = pseudo_hash
            if pseudo_hash_match_counts[cid] == 10:
                raise RuntimeError(f"Camera {cid} hasn't changed for 10 frames")


        q_desired = franka_interface.last_q_d
        arm_q = franka_interface.last_q
        arm_dq = franka_interface.last_dq
        last_gripper_q = franka_interface.last_gripper_q
        if arm_q is None or arm_dq is None or last_gripper_q is None or rgbs is None:
            print(
                f"arm_q: {arm_q} is None, arm_dq: {arm_dq} is None, "
                f"last_gripper_q: {last_gripper_q} is None"
            )
            try:
                INFERENCE_RATE.sleep()
            except rospy.ROSInterruptException:
                print("[act_inference] rate.sleep() exception")
                break
            continue

        curr_time = time()
        gripper_pos = float(last_gripper_q) / 2.0
        last_q = np.concatenate([arm_q, [gripper_pos, gripper_pos]])
        last_dq = np.concatenate([arm_dq, [0.0, 0.0]]) 
        # ^ NOTE: Gripper velocity is set to [0, 0] by log_demonstrations.py
        state = np.concatenate([last_q, last_dq]).astype(np.float32)
        action = inference_node.get_real_action(rgbs, state)

        try:
            if enter_is_pressed():
                cprint(" --> Resetting", "yellow")
                RESETTING = True
                reset_joints_to(franka_interface, STARTING_CONFIGS[args.starting_qpos_alias.upper()], gripper_open=True)
                cprint("Press ENTER to continue", "yellow")
                inference_node = ACTRealInference(args.checkpoint_path, args)
                while not enter_is_pressed():
                    INFERENCE_RATE.sleep()
                RESETTING = False
                cprint("Running policy 🏁", "yellow")
                continue

            INFERENCE_RATE.sleep()

        except rospy.ROSInterruptException:
            shutdown()
            break
