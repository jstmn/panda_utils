#!/usr/bin/env python3
"""Turn real Franka demos + FoundationPose object poses into a robomimic `demo.hdf5` that
the mimicgen_aux ID90 training code can read.

WHAT IT JOINS
    per demo, three things that live in different places:
      1. `~/Desktop/data/<session>/traj_<i>.h5`   -- the logger's raw record: d435 colour,
         zedm colour, joint_states_q (7 arm + 2 finger).
      2. `.../assets/cam_1__<slug>/<slug>__poses_world.npy`  -- (T,4,4) T_base_object from
         `real2sim-toolkit/examples/track_object.py`.
      3. Jrl2 forward kinematics, to get the end-effector pose the logger never stored.

WHAT THE TRAINING CODE ACTUALLY WANTS
    Read off exps/id90/*.json rather than guessed:
      observation.modalities.obs.low_dim = [robot0_eef_pos, robot0_eef_quat,
                                            robot0_gripper_qpos]
      observation.modalities.obs.rgb     = [agentview_image, robot0_eye_in_hand_image]
      train.dataset_keys                 = [actions, rewards, dones]
      algo.aux_pose.obj_blocks           = [0]   for square_d2
    `obj_blocks` are offsets into `obs/object`, each block read as [pos(3) | quat(4)].
    So block 0 must be the manipulated object -- for Square, the nut.

MORE THAN ONE TRACKED OBJECT
    Repeat --object-slug once per object. `obs/object` becomes (T, 7*N) with the blocks in
    the order the flags were given, which is the order `obj_blocks` indexes, and
    `obs/aux_valid` becomes (T, N) -- one validity column per object rather than one ANDed
    mask, because the object waiting its turn is routinely occluded while the other is
    being carried and ANDing would discard the carried object's good labels.

    A single --object-slug keeps the old shapes exactly, (T, 7) and (T,), so datasets built
    before this reproduce byte for byte.

    Each object's "static" window is its OWN grasp, not the demo's first close and last
    open: `grasp_cycles` finds the closed intervals and `assign_cycles` works out which one
    carried which object from how far each moved. The pick order is therefore read from the
    data and may differ per demo.

ACTIONS ARE RECONSTRUCTED, NOT RECORDED
    `log_demonstration.py` never saved the teleop action: it builds one with
    `input2action`, hands it to `robot.control`, and drops it. Its `create_dataset`
    calls only cover joint states and images.

    So actions here are derived from what the arm actually did:
        dpos      = eef_pos[t+1] - eef_pos[t]                    (robot base frame)
        drot      = axis-angle of R[t]^T @ R[t+1]                (as OSC_POSE expects)
        grasp     = +1 while the fingers are closed, -1 while open
    That is a *better* BC label than the raw command would have been -- the policy should
    imitate the motion that happened, not a command the controller only partly tracked --
    but it is not the original command, and the last step is padded by repeating the
    previous action because there is no t+1 for it.

CONVENTIONS
    Quaternions are xyzw, matching robosuite/robomimic. `robot0_gripper_qpos` is
    [+f, -f] like robosuite's Panda, from the logger's single finger width.
    Images are centre-cropped to square before resizing, because a straight 1280x720 ->
    84x84 squash would distort every object; the crop still contains the whole workspace
    at this camera pose.

USAGE
    source ~/anaconda3/bin/activate r2st        # needs Jrl2 for FK

    python scripts/r2st/r2st_to_robomimic.py \
        --session ~/Desktop/data/square60 \
        --poses-root ~/Desktop/data/r2st/square60 \
        --object-slug black_square_object \
        --out ~/Desktop/data/robomimic/square_real30/demo.hdf5

    # two tracked objects: repeat --object-slug, block 0 first
    python scripts/r2st/r2st_to_robomimic.py \
        --session ~/Desktop/data/pickplace_test/pick_and_place__09-10_00-02-56 \
        --poses-root ~/Desktop/data/r2st/pickplace_s1 \
        --object-slug red_cylinder_object --object-slug green_cube \
        --out ~/Desktop/data/robomimic/pickplace/demo.hdf5

    # several recording sessions in one dataset: repeat the pair, in order. Demos are
    # concatenated and renumbered demo_0.., and each carries source_session/source_traj.
    python scripts/r2st/r2st_to_robomimic.py \
        --session ~/Desktop/data/square60/square__09-07_21-21-37 \
        --poses-root ~/Desktop/data/r2st/square60_v2 \
        --session ~/Desktop/data/square60/square__09-09_21-56-16 \
        --poses-root ~/Desktop/data/r2st/square40_0909 \
        --out ~/Desktop/data/robomimic/square_real100/demo.hdf5

    # sanity-check one demo without writing anything
    python scripts/r2st/r2st_to_robomimic.py ... --limit 1 --dry-run
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


def _quat_xyzw(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> xyzw quaternion (robosuite/robomimic order)."""
    from transforms3d.quaternions import mat2quat

    w, x, y, z = mat2quat(R)
    return np.array([x, y, z, w], dtype=np.float64)


def _axis_angle(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> axis-angle 3-vector (magnitude = angle)."""
    from transforms3d.axangles import mat2axangle

    axis, angle = mat2axangle(R)
    return np.asarray(axis, dtype=np.float64) * float(angle)


def _square_resize(img: np.ndarray, size: int, crop: bool) -> np.ndarray:
    import cv2

    if crop:
        h, w = img.shape[:2]
        s = min(h, w)
        y0, x0 = (h - s) // 2, (w - s) // 2
        img = img[y0:y0 + s, x0:x0 + s]
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def _fk_series(robot, qpos: np.ndarray, tcp_link: str) -> tuple[np.ndarray, np.ndarray]:
    """(T,3) positions and (T,3,3) rotations of `tcp_link` in the robot base frame."""
    names = list(robot.actuated_joint_names)
    pos = np.zeros((len(qpos), 3))
    rot = np.zeros((len(qpos), 3, 3))
    for t, q in enumerate(qpos):
        q = list(q) + [0.0] * (len(names) - len(q))
        T = np.asarray(robot.get_all_link_poses_non_batched({n: float(v) for n, v in zip(names, q)})[tcp_link])
        pos[t], rot[t] = T[:3, 3], T[:3, :3]
    return pos, rot


def build_actions(eef_pos: np.ndarray, eef_rot: np.ndarray, closed: np.ndarray) -> np.ndarray:
    """(T,7) OSC_POSE-style deltas: [dpos(3) | drot axis-angle(3) | grasp(1)]."""
    T = len(eef_pos)
    a = np.zeros((T, 7), dtype=np.float64)
    a[:-1, :3] = eef_pos[1:] - eef_pos[:-1]
    for t in range(T - 1):
        a[t, 3:6] = _axis_angle(eef_rot[t].T @ eef_rot[t + 1])
    # No t+1 exists for the final step; repeating the previous action keeps the array
    # aligned with the observations instead of injecting a spurious zero-motion sample.
    if T >= 2:
        a[-1, :6] = a[-2, :6]
    a[:, 6] = np.where(closed, 1.0, -1.0)
    return a


def grasp_cycles(closed: np.ndarray) -> list[tuple[int, int]]:
    """[(close_frame, open_frame)] for every interval the gripper is shut.

    `open_frame` is the first OPEN frame after the close, or T if it never reopens. A demo
    that starts with the gripper already shut counts as a cycle beginning at 0.
    """
    d = np.diff(closed.astype(int))
    closes = [int(i) + 1 for i in np.nonzero(d == 1)[0]]
    opens = [int(i) + 1 for i in np.nonzero(d == -1)[0]]
    if closed[0]:
        closes.insert(0, 0)
    return [(c, next((o for o in opens if o > c), len(closed))) for c in closes]


def assign_cycles(obj_pos: list[np.ndarray], cycles: list[tuple[int, int]]) -> list[int | None]:
    """Which grasp cycle carried each object, by how far that object moved during it.

    A two-object pick-and-place has two closed intervals, and which object goes first is a
    property of the demo, not of the task -- the operator may swap them, and on some rigs
    the order is randomised deliberately. Reading it out of the motion is both robust and
    free: during the cycle that carried it, an object travels the length of the transfer,
    while the other one sits still.

    Greedy on the largest displacement first, one cycle per object.
    """
    if not cycles:
        return [None] * len(obj_pos)
    pairs = []
    for oi, p in enumerate(obj_pos):
        for ci, (c, o) in enumerate(cycles):
            a, b = min(c, len(p) - 1), min(o, len(p) - 1)
            pairs.append((float(np.linalg.norm(p[b] - p[a])), oi, ci))
    pairs.sort(reverse=True)
    out: list[int | None] = [None] * len(obj_pos)
    taken: set[int] = set()
    for _, oi, ci in pairs:
        if out[oi] is None and ci not in taken:
            out[oi], _ = ci, taken.add(ci)
    return out


def aux_validity(obj_pos: np.ndarray, closed: np.ndarray, max_step_m: float,
                 static_tol_m: float, dilate: int,
                 cycle: tuple[int, int] | None = None,
                 eef_pos: np.ndarray | None = None, eef_rot: np.ndarray | None = None,
                 carry_tol_m: float | None = None, pre_ref_frames: int | None = None,
                 carry_anchor_tol_m: float | None = None) -> np.ndarray:
    """Per-frame trust in the tracked object pose, as uint8 (1 = usable as an aux target).

    FoundationPose is reliable while the object is plainly visible and moving smoothly,
    and unreliable across the moment the gripper uncovers it: measured over all 30 Square
    demos, every one carries 2-6 frames whose pose jumps more than 3 cm in a single 15 Hz
    step (up to 15 cm), and the post-release pose can drift up to 7.9 cm even though the
    nut is sitting still on the peg.

    Two physical facts pin down which frames are wrong, without any hand labelling:
      * a real object cannot move `max_step_m` between consecutive frames at this rate --
        0.03 m per 15 Hz step is 0.45 m/s, well above the teleop speeds actually used
        (measured max |dpos| was 7-10 mm/step);
      * the object is STATIC before the gripper first closes and after it finally opens,
        so within those windows any deviation from the window's median position is error.

    Neighbours of a bad frame are suspect too -- a jump means the pose was already sliding
    on the way in -- hence `dilate`.

    `cycle` is the (close, open) interval that carried THIS object, from `assign_cycles`.
    With one object it is the demo's only grasp and this reduces to the original behaviour.
    With two it matters: object B is still sitting untouched while object A is being
    carried, so B's "static" window runs to ITS OWN close, not to the first close in the
    demo. Passing None falls back to first-close/last-open, which is right only for one
    object.

    With `carry_tol_m` set, a third fact is enforced during the object's own grasp:
      * a grasped object moves RIGIDLY with the gripper, so its position expressed in the
        end-effector frame must stay where it was when the fingers closed.
    This catches what the other two cannot: an error that is smooth and not static. On the
    pick-and-place rig FoundationPose keeps the carried 5 cm cube correctly placed in the
    IMAGE but slides it ~10 cm along the camera ray to table height -- no single-frame jump,
    no static window, so it would otherwise be marked valid for the whole carry. Off by
    default so datasets built before this reproduce byte for byte.

    Both references above can themselves be wrong when tracking fails for long enough, and
    two opt-in options anchor them to frame 0 instead. Frame 0 is where FoundationPose
    registers, with the arm parked far away.
      * `pre_ref_frames`: the pre-grasp static window is compared against the median of the
        first N frames rather than its own median. On the pick-and-place cube, which drifts
        while the arm carries the other object over it, the drifted pose held most of the
        window in 5 of 101 demos. The median then sat 3-7 cm off, and the check kept the wrong
        frames and rejected the right ones.
      * `carry_anchor_tol_m`: if the object's position at grasp time (the carry reference)
        is further than this from its frame-0 position in xy, the whole carry is
        untrusted. The tolerance has to allow the approach bumping the object: the
        well-tracked cylinder moved up to 48 mm before its grasp, so use ~5 cm.

    WHAT THIS DOES NOT CATCH: an object that is static because it is HIDDEN. Both tests
    above are satisfied by a wrong pose that simply stops moving, which is exactly what an
    object placed inside a container looks like once the walls occlude it. On a shallow
    tray this is mild; for a deep box, gate on visibility as well.
    """
    T = len(obj_pos)
    valid = np.ones(T, dtype=np.uint8)

    step = np.linalg.norm(np.diff(obj_pos, axis=0), axis=1)
    bad = np.zeros(T, dtype=bool)
    jump = np.nonzero(step > max_step_m)[0]
    bad[jump] = True
    bad[np.minimum(jump + 1, T - 1)] = True

    if cycle is not None:
        windows = ((0, cycle[0]), (cycle[1], T))
    else:
        idx = np.nonzero(closed)[0]
        windows = ((0, int(idx[0])), (int(idx[-1]) + 1, T)) if len(idx) else ()
    if windows:
        for wi, (lo, hi) in enumerate(windows):
            if hi - lo >= 3:
                if wi == 0 and pre_ref_frames:
                    ref = np.median(obj_pos[: min(pre_ref_frames, hi)], axis=0)
                else:
                    ref = np.median(obj_pos[lo:hi], axis=0)
                bad[lo:hi] |= np.linalg.norm(obj_pos[lo:hi] - ref, axis=1) > static_tol_m

    if carry_tol_m is not None and cycle is not None and eef_pos is not None and eef_rot is not None:
        c, o = cycle
        if o - c >= 3:
            rel = np.einsum("tji,tj->ti", eef_rot[c:o], obj_pos[c:o] - eef_pos[c:o])
            # Reference from frames 5-10 after the close label: the fingers have finished closing
            # (median 5-7 frames on this rig) but the arm has not lifted yet (median 11-15).
            # Earlier is wrong in a way that looks like tracking error and is not: the closing
            # fingers CENTRE the object along their axis, which on Square moved the nut
            # 17-22 mm relative to the gripper in the demos where it was grasped off-centre.
            lo, hi = (5, 10) if o - c >= 10 else (0, o - c)
            ref = np.median(rel[lo:hi], axis=0)
            bad[c:o] |= np.linalg.norm(rel - ref, axis=1) > carry_tol_m
            if carry_anchor_tol_m is not None and c >= 3:
                anchor = np.median(obj_pos[: min(10, c)], axis=0)
                at_grasp = np.median(obj_pos[c + lo:c + hi], axis=0)
                if np.linalg.norm(at_grasp[:2] - anchor[:2]) > carry_anchor_tol_m:
                    bad[c:o] = True

    if dilate > 0 and bad.any():
        k = np.ones(2 * dilate + 1, dtype=int)
        bad = np.convolve(bad.astype(int), k, mode="same") > 0
    valid[bad] = 0
    return valid


def convert_demo(args, robot, traj_path: Path, poses_paths: list[Path]) -> dict:
    import h5py

    with h5py.File(traj_path, "r") as h:
        qpos = h["joint_states_q"][()]
        scene = h[f"{args.scene_camera}__color_image"]
        wrist_key = f"{args.wrist_camera}__color_image"
        has_wrist = wrist_key in h
        T = len(qpos)
        # One (T,4,4) track per object, in --object-slug order: that order IS the block
        # order the training config's `obj_blocks` indexes into, so it must not be sorted
        # or deduplicated behind the caller's back.
        poses = [np.load(q) for q in poses_paths]
        for q, arr in zip(poses_paths, poses):
            assert len(arr) == T, f"{traj_path.name}: {q.parent.name} has {len(arr)} poses vs {T} frames"

        agent = np.stack([_square_resize(scene[t], args.image_size, args.crop) for t in range(T)])
        wrist = (np.stack([_square_resize(h[wrist_key][t], args.image_size, args.crop) for t in range(T)])
                 if has_wrist else np.zeros_like(agent))

    arm = qpos[:, :7]
    finger = qpos[:, 7] if qpos.shape[1] > 7 else np.zeros(T)
    closed = finger < (finger.min() + finger.max()) / 2.0

    eef_pos, eef_rot = _fk_series(robot, qpos, args.tcp_link)
    eef_quat = np.stack([_quat_xyzw(R) for R in eef_rot])
    gripper_qpos = np.stack([finger, -finger], axis=1)

    n_obj = len(poses)
    obj = np.zeros((T, 7 * n_obj), dtype=np.float64)
    for b, arr in enumerate(poses):
        obj[:, 7*b:7*b+3] = arr[:, :3, 3]
        obj[:, 7*b+3:7*b+7] = np.stack([_quat_xyzw(P[:3, :3]) for P in arr])

    actions = build_actions(eef_pos, eef_rot, closed)

    # One validity column per object. NOT a single ANDed mask: the second object is
    # routinely occluded while the first is being carried, and ANDing would throw away
    # the first object's perfectly good labels for those frames.
    cycles = grasp_cycles(closed)
    if n_obj == 0:
        # --no-aux. Everything below this point describes tracked objects; with none, the
        # honest thing is to omit the keys rather than write zeros that read like poses.
        return dict(
            obs=dict(
                agentview_image=agent,
                robot0_eye_in_hand_image=wrist,
                robot0_eef_pos=eef_pos,
                robot0_eef_quat=eef_quat,
                robot0_gripper_qpos=gripper_qpos,
            ),
            actions=actions,
            rewards=np.concatenate([np.zeros(T - 1), [1.0]]),
            dones=np.concatenate([np.zeros(T - 1, dtype=np.int64), [1]]),
            num_samples=T,
            joint_qpos=arm,
            aux_valid=None,
            n_obj=0,
            cycles=cycles,
            assigned=[],
        )
    assigned = assign_cycles([arr[:, :3, 3] for arr in poses], cycles)
    cols = [aux_validity(obj[:, 7*b:7*b+3], closed, args.max_step_m, args.static_tol_m,
                         args.mask_dilate,
                         cycle=cycles[assigned[b]] if assigned[b] is not None else None,
                         eef_pos=eef_pos, eef_rot=eef_rot, carry_tol_m=args.carry_tol_m,
                         pre_ref_frames=args.pre_grasp_ref_frames,
                         carry_anchor_tol_m=args.carry_anchor_tol_m)
            for b in range(n_obj)]
    # Shape (T,) for a single object, so files built before multi-object support are
    # reproduced byte for byte; (T, n) once there is more than one.
    aux_valid = cols[0] if n_obj == 1 else np.stack(cols, axis=1)
    dones = np.zeros(T, dtype=np.int64)
    dones[-1] = 1
    rewards = np.zeros(T, dtype=np.float64)
    rewards[-1] = 1.0  # every recorded demo is a successful insertion

    return dict(
        obs=dict(
            agentview_image=agent,
            robot0_eye_in_hand_image=wrist,
            robot0_eef_pos=eef_pos,
            robot0_eef_quat=eef_quat,
            robot0_gripper_qpos=gripper_qpos,
            object=obj,
            # 1 where the tracked pose is trustworthy. Gate the aux loss on this: a wrong
            # pose is worse than a missing one, because it pushes gradient into the shared
            # visual encoder for an input that does not support it.
            aux_valid=aux_valid,
        ),
        actions=actions,
        rewards=rewards,
        dones=dones,
        num_samples=T,
        joint_qpos=arm,
        aux_valid=aux_valid,
        n_obj=n_obj,
        cycles=cycles,
        assigned=assigned,
    )


def main() -> int:
    import h5py

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--session", type=Path, required=True, action="append",
                   help="dir holding the logger's traj_*.h5; repeat to concatenate recording "
                        "sessions, pairing each with the --poses-root given in the same order")
    p.add_argument("--no-aux", action="store_true",
                   help="build without tracked object poses: obs/object and obs/aux_valid are "
                        "left out entirely rather than zero-filled, so a baseline arm can train "
                        "before tracking exists. An aux arm cannot use the result -- it has no "
                        "labels to gate, not empty ones")
    p.add_argument("--poses-root", type=Path, required=False, action="append",
                   help="dir with <i>/assets/cam_1__<slug>/<slug>__poses_world.npy per demo, "
                        "one per --session, in the same order")
    p.add_argument("--object-slug", action="append",
                   help="the --object-description track_object.py ran with, spaces as underscores. "
                        "Repeat once per tracked object; the ORDER is the block order that the "
                        "training config's algo.aux_pose.obj_blocks indexes into, so block 0 must "
                        "come first. Default: black_square_object (the Square task's nut)")
    p.add_argument("--out", type=Path, required=True, help="demo.hdf5 to write")
    p.add_argument("--scene-camera", default="d435")
    p.add_argument("--wrist-camera", default="zedm")
    p.add_argument("--image-size", type=int, default=84, help="square image side (config crops to 76)")
    p.add_argument("--crop", action="store_true", default=True,
                   help="centre-crop to square before resizing (default on)")
    p.add_argument("--no-crop", dest="crop", action="store_false")
    p.add_argument("--tcp-link", default="panda_hand_tcp")
    p.add_argument("--max-step-m", type=float, default=0.03,
                   help="object pose jump per frame above which the pose is untrusted (default 0.03)")
    p.add_argument("--static-tol-m", type=float, default=0.02,
                   help="allowed deviation while the object must be static, i.e. before the first "
                        "grasp and after the last release (default 0.02)")
    p.add_argument("--mask-dilate", type=int, default=2,
                   help="frames marked invalid either side of a bad frame (default 2)")
    p.add_argument("--carry-tol-m", type=float, default=None,
                   help="during an object's own grasp, max drift of its position in the end-effector "
                        "frame before the pose is untrusted (e.g. 0.02). Default off, which keeps "
                        "earlier datasets byte-identical")
    p.add_argument("--pre-grasp-ref-frames", type=int, default=None,
                   help="compare the pre-grasp static window against the median of the first N frames "
                        "(frame-0 registration) instead of the window's own median, e.g. 10. Default off")
    p.add_argument("--carry-anchor-tol-m", type=float, default=None,
                   help="with --carry-tol-m: untrust the whole carry if the object's xy at grasp time is "
                        "further than this from its frame-0 xy (e.g. 0.05). Default off")
    p.add_argument("--env-name", default="SquareReal",
                   help="env_args.env_name recorded as provenance (default SquareReal)")
    p.add_argument("--val-frac", type=float, default=0.1, help="fraction of demos in mask/valid")
    p.add_argument("--limit", type=int, default=None, help="convert at most N demos")
    p.add_argument("--dry-run", action="store_true", help="report per-demo stats, write nothing")
    args = p.parse_args()
    if not args.object_slug and not args.no_aux:
        args.object_slug = ["black_square_object"]
    if args.no_aux and not args.poses_root:
        args.poses_root = []
    elif not args.no_aux and not args.poses_root:
        raise SystemExit("--poses-root is required unless --no-aux is given")

    import warnings

    warnings.filterwarnings("ignore")
    from jrl2.robots import ALL_ROBOTS

    robot = [c() for c in ALL_ROBOTS if c().name == "panda"][0]

    # One (--session, --poses-root) pair per recording session, concatenated in the order
    # given. Sessions are kept explicit rather than discovered under a common parent: two
    # sessions both number their trajectories from 0, so the pairing is what says which
    # poses belong to which recording, and guessing it would silently mismatch them.
    if args.no_aux:
        args.poses_root = [None] * len(args.session)
        args.object_slug = []
    assert len(args.session) == len(args.poses_root), (
        f"{len(args.session)} --session but {len(args.poses_root)} --poses-root; "
        "give one --poses-root per --session, in the same order")

    jobs, n_recorded = [], 0
    for session, poses_root in zip(args.session, args.poses_root):
        session = session.expanduser()
        # Accept either the session directory itself (traj_*.h5 directly inside) or a parent
        # holding one such directory. Pointing at the wrong level is easy and the failure --
        # "no traj_*.h5" -- does not say which level was expected.
        trajs = sorted(session.glob("traj_*.h5"), key=lambda q: int(re.search(r"traj_(\d+)", q.name).group(1)))
        if not trajs:
            trajs = sorted(session.glob("*/traj_*.h5"), key=lambda q: int(re.search(r"traj_(\d+)", q.name).group(1)))
            sessions = {t.parent for t in trajs}
            assert len(sessions) <= 1, (
                f"{session} holds several recording sessions {sorted(s.name for s in sessions)}; "
                "point --session at one of them, and repeat --session/--poses-root per session, "
                "so each recording is paired with its own poses")
        assert trajs, f"no traj_*.h5 in {session} or in any directory directly under it"
        n_recorded += len(trajs)

        found = 0
        for t in trajs:
            i = int(re.search(r"traj_(\d+)", t.name).group(1))
            if args.no_aux:
                jobs.append((i, t, [], t.parent.name))
                found += 1
                continue
            pps = [(poses_root.expanduser() / str(i) / "assets" /
                    f"cam_1__{slug}" / f"{slug}__poses_world.npy") for slug in args.object_slug]
            missing = [slug for slug, q in zip(args.object_slug, pps) if not q.is_file()]
            if not missing:
                jobs.append((i, t, pps, t.parent.name))
                found += 1
            else:
                # Every tracked object must be present: a demo missing one of them would
                # otherwise land in the file with a silently zero-filled block.
                print(f"  skip {t.parent.name}/traj_{i}: no poses for {', '.join(missing)} "
                      f"under {poses_root.expanduser()/str(i)/'assets'}")
        print(f"  {trajs[0].parent.name}: {found}/{len(trajs)} demos"
              + ("" if args.no_aux else " with poses"))
    if args.limit:
        jobs = jobs[: args.limit]
    print(f"{len(jobs)} demos" + ("" if args.no_aux else " with poses")
          + f" (of {n_recorded} recorded)")

    out = args.out.expanduser()
    if not args.dry_run:
        out.parent.mkdir(parents=True, exist_ok=True)
    # Resume: reading 2-3 GB source files fills the page cache, and a watchdog that judges
    # memory by `free` rather than `available` can kill the run part-way. Reopening in
    # append mode and skipping demos already written means each restart makes progress
    # instead of replaying the same first N demos into the same wall.
    resume = (not args.dry_run) and out.is_file()
    f = None if args.dry_run else h5py.File(out, "a" if resume else "w")
    grp = None if f is None else (f["data"] if resume and "data" in f else f.create_group("data"))
    done = sorted(int(k.split("_")[1]) for k in grp.keys()) if grp is not None else []
    if done:
        print(f"resuming: {len(done)} demos already in {out.name}")
    total = 0 if not done else int(grp.attrs.get("total", 0))
    written = [f"demo_{i}" for i in done]
    print("objects (block order): "
          + ("none -- --no-aux, obs/object and obs/aux_valid are omitted"
             if args.no_aux else ", ".join(args.object_slug)))
    for n, (i, tpath, ppaths, sess_name) in enumerate(jobs):
        if n in done:
            continue
        d = convert_demo(args, robot, tpath, ppaths)
        T = d["num_samples"]
        total += T
        obj = d["obs"].get("object")
        span = float(np.linalg.norm(obj[-1, :3] - obj[0, :3])) if obj is not None else float("nan")
        av = d["aux_valid"]
        av_txt = ("no aux" if av is None else
                  f"aux_valid {100*av.mean():5.1f}%" if av.ndim == 1 else
                  "aux_valid " + "/".join(f"{100*av[:, b].mean():.1f}" for b in range(av.shape[1])) + "%")
        amax = np.abs(d["actions"][:, :3]).max()
        grasp = d["actions"][:, 6]
        n_closed = int((grasp > 0).sum())
        # transitions, not just the count: a demo with no open->closed->open cycle never
        # actually picked the object up, whatever its frame tally says.
        n_trans = int(np.abs(np.diff((grasp > 0).astype(int))).sum())
        print(f"  demo_{n:<3d} ({sess_name}/traj_{i:<2d}) T={T:4d}  "
              + ("" if obj is None else f"object moved {span:.3f} m  ")
              + f"max |dpos| {amax*1000:5.1f} mm/step  closed {n_closed:4d} fr, {n_trans} tr  "
              + f"{av_txt}  cycles {d['cycles']} -> {d['assigned']}")
        if f is None:
            continue
        g = grp.create_group(f"demo_{n}")
        g.attrs["source_traj"] = i
        # With several sessions concatenated, the trajectory index alone is ambiguous --
        # every session numbers its recordings from 0.
        g.attrs["source_session"] = sess_name
        g.attrs["object_slugs"] = json.dumps(args.object_slug)
        g.attrs["num_samples"] = T
        og = g.create_group("obs")
        for k, v in d["obs"].items():
            og.create_dataset(k, data=v, compression="gzip", compression_opts=1)
        for k in ("actions", "rewards", "dones"):
            g.create_dataset(k, data=d[k])
        g.create_dataset("joint_qpos", data=d["joint_qpos"])
        grp.attrs["total"] = total
        # Flush at every demo boundary. HDF5 buffers, so a process killed mid-write leaves
        # a file whose header will not reopen ("bad object header version number") and the
        # whole run has to start again -- which is exactly what happened here.
        f.flush()
        written.append(f"demo_{n}")

    if f is not None:
        grp.attrs["total"] = total
        # robomimic reads env_args to rebuild a rollout env. There is no simulator behind
        # real demos, so this records provenance and nothing more -- rollout must be
        # disabled in the training config (exp.rollout.enabled = false).
        grp.attrs["env_args"] = json.dumps({
            "env_name": args.env_name,
            "type": 1,
            "env_kwargs": {},
            "note": "real Franka demos; no simulator. Disable rollout in the training config.",
        })
        grp.attrs["aux_poses"] = "none" if args.no_aux else json.dumps(args.object_slug)
        if "mask" in f:
            del f["mask"]
        m = f.create_group("mask")
        n_val = max(1, int(round(len(written) * args.val_frac)))
        rng = np.random.default_rng(0)
        idx = rng.permutation(len(written))
        val = sorted(written[j] for j in idx[:n_val])
        train = sorted(written[j] for j in idx[n_val:])
        m.create_dataset("valid", data=np.array(val, dtype="S"))
        m.create_dataset("train", data=np.array(train, dtype="S"))
        f.close()
        print(f"\nwrote {out}  ({len(written)} demos, {total} samples, "
              f"{out.stat().st_size/1e9:.2f} GB)")
        print(f"  mask/train {len(train)}   mask/valid {len(val)}")
        print("  NOTE: set experiment.rollout.enabled=false -- there is no sim env for real data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
