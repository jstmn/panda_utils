import argparse
from pathlib import Path
import h5py
from deoxys.franka_interface import FrankaInterface
from panda_utils.deoxys_controller import DeoxysController

"""
This script replays joint configurations from a given trajectory file.

The h5 format is as follows:
/                        Group
/joint_states_dq         Dataset {62, 9}
/joint_states_q          Dataset {62, 9}

# Example usage:
python scripts/replay_joint_configurations.py \
    --h5_file /home/resl/Desktop/data/xinyi_data/test_realdeal2__01-15_20:53:17/traj_1.h5s

h5dump --width=200 -d /joint_states_q /home/resl/Desktop/data/xinyi_data/test_1__01-15_20:25:37/traj_0.h5
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5_file", type=str, required=True)
    args = parser.parse_args()

    with h5py.File(args.h5_file, "r") as f:
        joint_states_dq = f["joint_states_dq"][:]
        joint_states_q = f["joint_states_q"][:]

    print(joint_states_dq.shape)
    print(joint_states_q.shape)

    def should_stop():
        return False

    project_dir = Path(__file__).parent.parent
    franka_interface = FrankaInterface(f"{project_dir}/configs/charmander.yml")
    controller = DeoxysController(franka_interface, launch_viser=True, viser_use_visual=False)
    controller.joint_position_control(joint_states_q, tmax=100.0, should_stop=should_stop)
    print("Finished")
