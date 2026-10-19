#!/usr/bin/env python3
"""Generate GLB meshes for the pick-and-place task's two manipulated objects -- a drinks
can and a cube -- so `real2sim-toolkit/examples/track_object.py` can run without a Meshy
API key. Companion to `r2st_make_nut_meshes.py`.

WHY THIS EXISTS
    track_object.py builds the tracked object's mesh with Meshy AI
    (`MeshUtils.generate_with_meshy`, src/r2st/utils.py) and asserts out without
    MESHY_API_KEY. That helper short-circuits if a GLB already sits at

        <--output-dir>/<--camera>__<object description, spaces -> underscores>/model_glb.glb

    so writing one there skips Meshy entirely.

RUN THE PROMPT SWEEP FIRST
    The slug MUST equal the --object-description that SAM 3 actually answers to on this
    rig, and the task name is usually not it: "square nut", "nut" and "metal nut" all
    returned ZERO masks here, while "grey square object" worked. Find the working prompts
    with `r2st_sam_prompt_sweep.py`, then generate the meshes under those names. Doing it
    in the other order means regenerating under a second slug later.

PROPORTIONS MATTER, ABSOLUTE SIZE DOES NOT
    track_object.py calls `scale_glb_to_pointcloud`, which rescales the mesh onto the
    observed point cloud before FoundationPose sees it, so a constant scale error is
    absorbed and only the aspect ratio survives. Two consequences here:

      * the CAN's height/diameter is its only shape cue, because a body of revolution has
        no other. Measure the real can.
      * the CUBE is scale-free by construction -- 1:1:1 whatever its size -- so the
        placeholder is exactly right as long as it really is a cube. If the block is
        oblong, pass --cube-extents with the three real sides.

NEITHER OBJECT'S YAW IS SUPERVISED ON THIS TASK
    The can is a body of revolution: the same image is consistent with every yaw, so a
    yaw target read off the tracked pose is noise whose gradient lands in the shared
    visual encoder. This is the coffee_pod failure the simulated study already handles.
    An unmarked cube is 4-fold symmetric, which `rotation_fold` could fold to 90 deg, but
    this task drops rotation for both objects. In
    `mimicgen_aux/scripts/mg_make_train_configs.py` that is

        ROTATION_BLOCKS[<task>] = []      # which blocks KEEP rotation; none here
        OBJ_BLOCKS[<task>]      = [0, 1]  # can, cube -- both are manipulated

    FoundationPose will still emit a yaw for the can, and it will still swing. That is the
    pose being underdetermined, not a tracking bug to chase.

USAGE
    source ~/anaconda3/bin/activate r2st

    python scripts/r2st/r2st_make_pickplace_meshes.py \
        --outdir ~/Desktop/data/r2st/can_smoke/0/assets --camera cam_1 \
        --can-slug red_can --cube-slug green_cube

    # with real measurements (metres)
    python scripts/r2st/r2st_make_pickplace_meshes.py --outdir ... --camera cam_1 \
        --diameter 0.0663 --height 0.1225 --cube-extents 0.05 0.05 0.05
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import trimesh


def build_can(
    diameter: float,
    height: float,
    top_diameter: float,
    bottom_diameter: float,
    taper_height: float,
    sections: int,
) -> trimesh.Trimesh:
    """A cylinder tapering into a narrower lid and base, as a surface of revolution.

    Built with `revolve` from a half-profile rather than by unioning primitives: booleans
    on near-tangent cylinders leave sliver faces exactly where the silhouette is, and the
    silhouette is the entire shape signal for an object with no other features.
    """
    assert 0 < top_diameter <= diameter, f"top ({top_diameter}) must be 0 < t <= diameter ({diameter})"
    assert 0 < bottom_diameter <= diameter, f"bottom ({bottom_diameter}) must be 0 < b <= diameter ({diameter})"
    assert 2 * taper_height < height, f"tapers ({2 * taper_height}) must be shorter than the can ({height})"

    r, rt, rb, h, th = diameter / 2, top_diameter / 2, bottom_diameter / 2, height, taper_height
    # (radius, z) from the base centre up to the lid centre, z measured from the can's foot.
    profile = np.array([
        [0.0, 0.0],
        [rb,  0.0],
        [r,   th],
        [r,   h - th],
        [rt,  h],
        [0.0, h],
    ])
    can = trimesh.creation.revolve(profile, sections=sections)
    can.apply_translation(-can.centroid)
    return can


def build_cube(extents: tuple[float, float, float]) -> trimesh.Trimesh:
    """A rectangular block. Scale-free when the sides are equal, so the default is exact."""
    cube = trimesh.creation.box(extents=extents)
    cube.apply_translation(-cube.centroid)
    return cube


def write_glb(mesh: trimesh.Trimesh, outdir: Path, camera: str, slug: str) -> Path:
    asset_dir = outdir / f"{camera}__{slug}"
    asset_dir.mkdir(parents=True, exist_ok=True)
    path = asset_dir / "model_glb.glb"
    mesh.export(path)
    ext = mesh.extents
    print(
        f"  {slug:12s} -> {path}\n"
        f"      watertight={mesh.is_watertight}  volume={mesh.volume * 1e6:.1f} cm^3  "
        f"extents=({ext[0]*100:.1f}, {ext[1]*100:.1f}, {ext[2]*100:.1f}) cm  "
        f"faces={len(mesh.faces)}"
    )
    return path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--outdir", type=Path, required=True,
                   help="track_object.py's --output-dir, e.g. ~/Desktop/data/r2st/can_smoke/0/assets")
    p.add_argument("--camera", default="cam_1", help="track_object.py's --camera (default: cam_1)")
    p.add_argument("--can-slug", default="red_can",
                   help="--object-description with spaces as underscores. Must match the prompt SAM 3 "
                        "answers to -- run r2st_sam_prompt_sweep.py first")
    p.add_argument("--cube-slug", default="green_cube", help="ditto for the cube")

    p.add_argument("--diameter", type=float, default=0.0663, help="can body diameter, m (default 0.0663)")
    p.add_argument("--height", type=float, default=0.1225, help="can height, m (default 0.1225)")
    p.add_argument("--top-diameter", type=float, default=0.0570, help="lid diameter, m (default 0.0570)")
    p.add_argument("--bottom-diameter", type=float, default=0.0540, help="base diameter, m (default 0.0540)")
    p.add_argument("--taper-height", type=float, default=0.0085,
                   help="height over which the body tapers to lid/base, m (default 0.0085)")
    p.add_argument("--sections", type=int, default=64,
                   help="radial segments (default 64; the silhouette is the only shape cue, so do not "
                        "drop this far)")

    p.add_argument("--cube-extents", type=float, nargs=3, metavar=("X", "Y", "Z"), default=[0.05, 0.05, 0.05],
                   help="cube sides, m (default 0.05 cubed; a true cube is scale-free, so this only "
                        "matters if the block is oblong)")
    p.add_argument("--no-can", action="store_true", help="only write the cube mesh")
    p.add_argument("--no-cube", action="store_true", help="only write the can mesh")
    args = p.parse_args()

    outdir = args.outdir.expanduser()
    print(f"Writing GLBs under {outdir}")

    if not args.no_can:
        can = build_can(args.diameter, args.height, args.top_diameter, args.bottom_diameter,
                        args.taper_height, args.sections)
        write_glb(can, outdir, args.camera, args.can_slug)
        ratio = args.height / args.diameter
        print(f"      height / diameter = {ratio:.2f}  <- the one proportion that survives rescaling")
        if not 1.2 < ratio < 3.0:
            print("      WARNING: not a drinks-can aspect ratio. Check the units (metres).")

    if not args.no_cube:
        cube = build_cube(tuple(args.cube_extents))
        write_glb(cube, outdir, args.camera, args.cube_slug)
        e = np.asarray(args.cube_extents, dtype=float)
        if e.ptp() / e.mean() > 0.02:
            print(f"      note: sides differ by {100 * e.ptp() / e.mean():.0f}%, so this block is not a "
                  "cube and its proportions now carry information. Measure them.")

    print("\nThe can's proportions are a PLACEHOLDER (355 ml Korean coke) unless you measured it.")
    print("The cube's are exact if it is a cube, because scale_glb_to_pointcloud absorbs size.")
    print("\nReminder: this task supervises NO rotation. Set ROTATION_BLOCKS[<task>] = [] in")
    print("mimicgen_aux/scripts/mg_make_train_configs.py -- the can's yaw is not in the image,")
    print("and the cube's is dropped along with it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
