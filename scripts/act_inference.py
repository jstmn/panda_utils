import rospy
import threading
import torch
import numpy as np
import torchvision.transforms as T
from std_msgs.msg import Float64MultiArray
from time import sleep, time
# 학습 코드에서 사용한 라이브러리들
from scripts.act.detr_vae import DETRVAE
from act.backbone import build_backbone
from act.transformer import build_transformer
from act.detr_vae import build_encoder

from deoxys.franka_interface import FrankaInterface
from deoxys.utils.config_utils import get_default_controller_config
from panda_utils.rgb_subscriber import RGBSubscriber
from panda_utils.constants import DEFAULT_DEOXYS_INTERFACE_CFG

torch.set_printoptions(precision=4, sci_mode=False, linewidth=200)
np.set_printoptions(precision=4, suppress=True, linewidth=200)

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

    camera_id: str = "south"



class ACTRealInference:
    def __init__(self, ckpt_path, args, device="cuda"):
        self.device = torch.device(device)
        self.args = args 
        

        ckpt = torch.load(ckpt_path, map_location=self.device)

        print(ckpt.keys())
        print(ckpt['norm_stats'].keys())
        stats = ckpt["norm_stats"]
        print("action_mean:", stats["action_mean"])
        print("action_std:", stats["action_std"])
        print("state_mean:", stats["state_mean"])
        print("state_std:", stats["state_std"])
        print("example_state:", stats["example_state"])



        self.stats = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in ckpt['norm_stats'].items()}
        
        self.agent = Agent_Inference(args).to(self.device)
        self.agent.load_state_dict(ckpt['ema_agent'])
        self.agent.eval()

        # 3. 전처리 도구
        self.resize = T.Resize((224, 224), antialias=True)
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # 4. Temporal Aggregation 설정
        self.num_queries = args.num_queries
        self.step_idx = 0
        self.max_timesteps = 10000

        self.all_time_actions = torch.zeros([self.max_timesteps, self.max_timesteps + self.num_queries, 8]).to(self.device)
        self.state_dim = 18
        self.act_dim = 8

    def preprocess(self, rgb_img):
        img_t = torch.from_numpy(rgb_img.copy()).permute(2, 0, 1).float() # (3, H, W)
        img_t = self.resize(img_t) / 255.0
        img_t = self.normalize(img_t)

        #print(f"img_t: {img_t.shape}")
        img_t = img_t.unsqueeze(0)
        return img_t.unsqueeze(0).to(self.device) # (1, 3, 224, 224)

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

        # actions_populated 로직 (evaluate.py와 동일)
        actions_populated = torch.zeros(self.max_timesteps, dtype=torch.bool, device=device)
        actions_populated[max(0, ts + 1 - num_queries):ts+1] = True

        actions_for_curr_step = actions_for_curr_step[actions_populated] # (num_populated, act_dim)

        k = 0.01
        exp_weights = torch.exp(-k * torch.arange(len(actions_for_curr_step), device=device))
        exp_weights = exp_weights / exp_weights.sum()
        exp_weights = exp_weights.unsqueeze(-1) # (num_populated, 1)

        raw_action = (actions_for_curr_step * exp_weights).sum(dim=0) # (act_dim,)

        # 3. Post-process (De-normalization)
        action = (raw_action * self.stats['action_std']) + self.stats['action_mean']
        self.step_idx += 1
        return action.cpu().numpy().flatten()


class Agent_Inference(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.state_dim = 18
        self.act_dim = 8
        
        backbones = [build_backbone(args)]
        transformer = build_transformer(args)
        encoder = build_encoder(args)

        self.model = DETRVAE(
            backbones, transformer, encoder,
            state_dim=self.state_dim, action_dim=self.act_dim,
            num_queries=args.num_queries,
        )

    def get_action(self, obs):
        # 이미 preprocess에서 정규화됨
        a_hat, _ = self.model(obs)
        return a_hat

if __name__ == "__main__":
    rospy.init_node("maniskill_act_inference", anonymous=False)
    
    args = Args()
    
    # 2. 로거 및 컨트롤러 초기화
    rgb_subscriber = RGBSubscriber(camera_ids=[args.camera_id])
    
    # 3. 추론 노드 생성
    ckpt_path = "/home/resl/Desktop/panda_utils/scripts/act/ACT_real_RaiseCube_75000.pt"
    inference_node = ACTRealInference(ckpt_path, args)

    franka_interface = FrankaInterface(DEFAULT_DEOXYS_INTERFACE_CFG, use_visualizer=False)
    controller_type = "JOINT_IMPEDANCE" # THIS IS BETTER THAN JOINT_POSITION
    controller_cfg = get_default_controller_config(controller_type=controller_type)

    SHUTDOWN = False
    action = None
    sleep(2)

    def _on_shutdown():
        global SHUTDOWN
        SHUTDOWN = True

    rospy.on_shutdown(_on_shutdown)

    def control_loop():
        global SHUTDOWN
        GRIPPER_INDEX = 7
        rate = rospy.Rate(60)
        while not SHUTDOWN and not rospy.is_shutdown():
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                break

            if action is None:
                continue
            action_cp = action.copy()
            if action_cp[GRIPPER_INDEX] > 0.03:
                action_cp[GRIPPER_INDEX] = -1 # -1 is OPEN with deoxys control
            else:
                action_cp[GRIPPER_INDEX] = 1 # 1 is CLOSE with deoxys control

            print(f"action: {action}")
            try:
                franka_interface.control(
                    controller_type=controller_type,
                    action=action_cp,
                    controller_cfg=controller_cfg,
                )
            except Exception as e:
                print(f"[act_inference] control_loop exception: {e!r}")
                SHUTDOWN = True
                break

    control_thread = threading.Thread(target=control_loop, daemon=True)
    control_thread.start()

    prev_gripper_pos = None
    prev_time = None
    rate = rospy.Rate(15)

    try:
        while not rospy.is_shutdown():
            rate.sleep()
            rgb = rgb_subscriber.latest_rgb(args.camera_id)
            arm_q = franka_interface.last_q
            arm_dq = franka_interface.last_dq
            last_gripper_q = franka_interface.last_gripper_q
            if arm_q is None or arm_dq is None or last_gripper_q is None or rgb is None:
                print(
                    f"arm_q: {arm_q} is None, arm_dq: {arm_dq} is None, "
                    f"last_gripper_q: {last_gripper_q} is None"
                )  # rgb: {rgb} is None")
                try:
                    rate.sleep()
                except rospy.ROSInterruptException:
                    print("[act_inference] rate.sleep() exception")
                    break
                continue

            curr_time = time()
            gripper_pos = float(last_gripper_q) / 2.0

            if prev_gripper_pos is None:
                gripper_dq = 0.0
            else:
                dt = curr_time - prev_time
                gripper_dq = (gripper_pos - prev_gripper_pos) / dt if dt > 0 else 0.0

            prev_gripper_pos = gripper_pos
            prev_time = curr_time

            last_q = np.concatenate([arm_q, [gripper_pos, gripper_pos]])
            last_dq = np.concatenate([arm_dq, [gripper_dq, gripper_dq]])
            state = np.concatenate([last_q, last_dq]).astype(np.float32)

            action = inference_node.get_real_action(rgb, state)

            # 
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                break
    except (KeyboardInterrupt, rospy.ROSInterruptException):
        pass
    finally:
        SHUTDOWN = True
        try:
            control_thread.join(timeout=2.0)
        except Exception:
            pass
        try:
            franka_interface.close()
        except Exception:
            pass