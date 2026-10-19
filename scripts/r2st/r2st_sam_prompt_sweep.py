#!/usr/bin/env python3
"""Find the SAM 3 prompt that actually segments an object on this rig, and check that the
depth under it can support FoundationPose -- before committing hours of GPU tracking.

WHY THIS EXISTS
    `track_object.py` segments the object on frame 0 with SAM 3 and dies on
    `assert masks_t.shape[0] >= 1` if the prompt returns nothing. On this rig SAM does not
    know these objects by their task names: "square nut", "nut" and "metal nut" all
    returned ZERO masks, and the prompt that worked was "grey square object". There is no
    way to guess this, and finding out during a 60-demo batch costs the batch.

    Two things therefore have to be settled before recording a full session:

      1. WHICH PROMPT RETURNS THE RIGHT OBJECT. The prompt is not just a lookup key -- it
         names the asset directory (`<camera>__<prompt with underscores>`) that the mesh
         must be written into, so changing it later means regenerating meshes.
      2. WHETHER THE DEPTH UNDER THAT MASK IS USABLE. FoundationPose registers the mesh on
         frame 0 and consumes depth, so a hole in the depth exactly where the object is
         corrupts every later frame silently -- the run still "succeeds". A specular
         object is the case to worry about: an aluminium can reflects the D435's IR
         pattern away instead of scattering it back, which a matte black nut never did.

WHAT IT REPORTS, PER PROMPT
    mask pixels, SAM's score, the mask centroid, the fraction of the mask that carries
    valid depth, and -- through the hand-eye extrinsics -- where in the ROBOT BASE frame
    that object actually is. The last one is the discriminator when two objects score
    similarly: the thing sitting at x = 0.55 m on the table is the one being manipulated,
    whatever SAM called it.

USAGE
    source ~/anaconda3/bin/activate r2st

    python scripts/r2st/r2st_sam_prompt_sweep.py \
        --h5 ~/Desktop/data/can_smoke/can__09-10_.../traj_0.h5 \
        --prompt "red can" --prompt "coke can" --prompt "red cylinder object" \
        --prompt "green cube" --prompt "green block" --prompt "green object"

    # save a masked cut-out per prompt to eyeball which object each one grabbed
    python scripts/r2st/r2st_sam_prompt_sweep.py --h5 ... --prompt "red can" --save-dir /tmp/sweep
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

DEFAULT_EXTRINSICS = Path.home() / "Desktop/data/r2st/square_smoke/extrinsics_cam_1__720p.yaml"
# d435 at 1280x720, from r2st/constants.py. The sweep only needs these to turn a mask into
# a position, so a wrong entry here misplaces the readout without affecting tracking.
FX, FY, CX, CY = 912.07177734375, 911.8675537109375, 642.824951171875, 355.07000732421875


def load_frame(h5_path: Path, frame: int) -> tuple[np.ndarray, np.ndarray]:
    """(rgb, depth_mm) from either the logger's traj_*.h5 or a merged r2st h5."""
    import h5py

    with h5py.File(h5_path, "r") as h:
        if "d435__color_image" in h:
            return np.asarray(h["d435__color_image"][frame]), np.asarray(h["d435__depth_image"][frame])
        g = "obs/sensor_data/cam_1"
        assert f"{g}/rgb" in h, f"{h5_path} is neither a logger traj nor a merged h5; keys: {list(h.keys())}"
        return np.asarray(h[f"{g}/rgb"][frame]), np.asarray(h[f"{g}/depth"][frame])


def main() -> int:
    import cv2
    import yaml

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--h5", type=Path, required=True, help="a recorded traj_*.h5 (or merged_sensor_data.h5)")
    p.add_argument("--prompt", action="append", required=True, help="repeat once per description to try")
    p.add_argument("--frame", type=int, default=0,
                   help="frame to segment (default 0 -- the one FoundationPose registers on)")
    p.add_argument("--extrinsics", type=Path, default=DEFAULT_EXTRINSICS,
                   help=f"hand-eye YAML, to report positions in the robot base frame (default {DEFAULT_EXTRINSICS})")
    p.add_argument("--save-dir", type=Path, default=None, help="write a masked cut-out per prompt here")
    p.add_argument("--top-k", type=int, default=1, help="how many ranked masks to report per prompt (default 1)")
    p.add_argument("--repo", type=Path, default=Path.home()/"Desktop/real2sim-toolkit",
                   help="real2sim-toolkit checkout, for r2st imports")
    args = p.parse_args()

    sys.path.insert(0, str(args.repo.expanduser()/"src"))
    from r2st.core import SAM3Predictor
    from r2st.utils import ImageUtils

    rgb, depth = load_frame(args.h5.expanduser(), args.frame)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    T = None
    extr = args.extrinsics.expanduser()
    if extr.is_file():
        T = np.array(yaml.safe_load(extr.read_text())["matrix"], dtype=np.float64)
    else:
        print(f"[warn] no extrinsics at {extr}; base-frame positions will be omitted")

    if args.save_dir:
        args.save_dir.expanduser().mkdir(parents=True, exist_ok=True)

    print(f"frame {args.frame} of {args.h5.name}   {rgb.shape[1]}x{rgb.shape[0]}")
    print(f"{'prompt':<26}{'px':>7}{'score':>7}{'(u,v)':>17}{'depth':>7}   base x,y,z (m)")
    predictor = SAM3Predictor()
    rows = []
    for prompt in args.prompt:
        try:
            masks, scores, _ = ImageUtils.get_sam_masks_ranked(predictor, bgr, prompt)
        except AssertionError:
            # This is the failure mode the sweep exists to find: a prompt SAM has nothing
            # for. track_object.py would raise the same assertion, an hour into a batch.
            print(f"{prompt:<26}{'--':>7}{'--':>7}{'NO MASK':>17}")
            rows.append((prompt, 0, 0.0, None, 0))
            continue

        for k in range(min(args.top_k, len(masks))):
            m, sc = masks[k], float(scores[k])
            v, u = np.nonzero(m)
            px = int(m.sum())
            cu, cv_ = float(u.mean()), float(v.mean())

            d = depth[v, u].astype(np.float64) / 1000.0
            good = d > 0
            cover = float(good.mean())
            pos = ""
            if T is not None and good.any():
                zc = float(np.median(d[good]))
                pc = np.array([(cu - CX) * zc / FX, (cv_ - CY) * zc / FY, zc])
                pb = T[:3, :3] @ pc + T[:3, 3]
                pos = f"({pb[0]:+.4f}, {pb[1]:+.4f}, {pb[2]:+.4f})"

            label = prompt if k == 0 else f"  ^ rank {k+1}"
            print(f"{label:<26}{px:>7}{sc:>7.2f}{f'({cu:6.1f},{cv_:6.1f})':>17}{100*cover:>6.0f}%   {pos}")
            rows.append((prompt, px, cover, pb if (T is not None and good.any()) else None, k))

            if args.save_dir and k == 0:
                slug = prompt.replace(" ", "_")
                cut = bgr.copy()
                cut[~m] = (cut[~m] * 0.15).astype(cut.dtype)
                cv2.imwrite(str(args.save_dir.expanduser()/f"{slug}.png"), cut)

    print("\n  px      -- the nut ran 5,400-7,500 px at this range; a few hundred is a fragment,")
    print("             tens of thousands is the table or the arm, not the object.")
    print("  depth   -- fraction of the mask carrying valid depth. FoundationPose registers on")
    print("             this frame and consumes depth: below ~70% on a specular object, expect")
    print("             the registration to sit wrong and every later pose to inherit it.")
    print("  base    -- where that mask actually is. The object on the table is the one at")
    print("             x ~ 0.45-0.62 m, |y| < 0.2 m, z ~ 0.")

    # Deliberately NOT ranked by mask size: on this rig "black object" returns the peg at
    # 12,882 px while the nut it competes with is 7,169, so the biggest mask is the wrong
    # answer. Position on the table is what separates a manipulated object from the scenery
    # and the arm, so that is the filter.
    plausible = [r for r in rows
                 if r[4] == 0 and r[1] > 0 and r[3] is not None
                 and 0.35 < r[3][0] < 0.70 and abs(r[3][1]) < 0.25 and -0.02 < r[3][2] < 0.12
                 and 500 < r[1] < 40000]
    got = [r for r in rows if r[4] == 0 and r[1] > 0]
    print()
    if not got:
        print("No prompt returned a mask. Describe colour and shape rather than naming the object --")
        print("'grey square object' worked here where 'square nut' returned nothing.")
    elif not plausible:
        print(f"{len(got)} prompt(s) returned a mask, but none of them landed on the table in the")
        print("workspace. Check the base column: they are probably the arm, the table, or the box.")
    else:
        print("On the table, in the workspace -- candidates for a manipulated object:")
        for r in plausible:
            flag = "" if r[2] > 0.70 else "   <- depth coverage too low to register on"
            print(f"    cam_1__{r[0].replace(' ', '_'):<24} {r[1]:>6} px  depth {100*r[2]:>3.0f}%{flag}")
        print("\nPick by the base column and the saved cut-outs, not by mask size, then generate the")
        print("mesh under exactly that slug (r2st_make_pickplace_meshes.py --can-slug/--cube-slug).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
