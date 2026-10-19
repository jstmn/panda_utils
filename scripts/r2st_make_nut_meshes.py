#!/usr/bin/env python3
"""Generate GLB meshes for the NutAssemblySquare square nut and its square peg, so
`real2sim-toolkit/examples/track_object.py` can run without a Meshy API key.

WHY THIS EXISTS
    track_object.py builds the tracked object's mesh with Meshy AI
    (`MeshUtils.generate_with_meshy`, src/r2st/utils.py). Without MESHY_API_KEY it
    asserts out. But that same helper short-circuits:

        existing = output_dir / "model_glb.glb"
        if existing.is_file():
            print(f"[info] Reusing existing Meshy GLB at {existing}")
            return existing

    so dropping a GLB at that exact path skips Meshy entirely. The asset dir is

        <--output-dir>/<--camera>__<object_description with spaces -> underscores>/

    e.g. `data/meshyai/cam_1__square_nut/model_glb.glb`.

ABSOLUTE SIZE BARELY MATTERS, PROPORTIONS DO
    track_object.py calls `scale_glb_to_pointcloud(glb_path, object_pts, ...)`, which
    rescales the mesh onto the observed point cloud before handing it to FoundationPose.
    So a mesh with the right *shape* and roughly right *aspect ratio* works even if every
    dimension is off by a constant factor. What it cannot fix is a wrong shape: a solid
    block where the real nut has a through-hole, or a missing handle, changes the
    silhouette FoundationPose matches against.

THE DEFAULTS ARE PLACEHOLDERS
    They follow robosuite's NutAssemblySquare proportions (square annulus + side handle;
    the square nut goes onto peg1, which is a BOX, not a cylinder -- see
    mimicgen/envs/robosuite/nut_assembly.py:341, where peg1's horizontal radius is
    norm(size[0:2]) while peg2's is size[0]). They are NOT measurements of this rig's
    physical parts. Measure the real nut and pass the numbers in; the closer the
    proportions, the better the pose.

USAGE
    source ~/anaconda3/bin/activate r2st

    # both meshes, straight into the asset dirs track_object.py will look in
    python scripts/r2st_make_nut_meshes.py \
        --outdir ~/Desktop/real2sim-toolkit/data/meshyai --camera cam_1

    # with real measurements (metres)
    python scripts/r2st_make_nut_meshes.py \
        --outdir ~/Desktop/real2sim-toolkit/data/meshyai --camera cam_1 \
        --nut-outer 0.058 --nut-hole 0.038 --nut-thickness 0.012 \
        --handle-length 0.048 --handle-width 0.018

    # then, with the FoundationPose gRPC server up in the container:
    uv run python examples/track_object.py --h5-path ... --camera cam_1 \
        --camera-model-id d435 --object-description "square nut" ...
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import trimesh


def build_square_nut(
    outer: float,
    hole: float,
    thickness: float,
    handle_length: float,
    handle_width: float,
) -> trimesh.Trimesh:
    """A square annulus (plate with a square through-hole) plus a handle off one side.

    The through-hole is the whole point: it is what makes the nut's silhouette
    distinctive from every angle, and what FoundationPose keys on.
    """
    assert hole < outer, f"hole ({hole}) must be smaller than outer ({outer})"
    plate = trimesh.creation.box(extents=(outer, outer, thickness))
    # Cut generously in z so the boolean produces a clean through-hole rather than a
    # coplanar-face mess at the top and bottom.
    cutter = trimesh.creation.box(extents=(hole, hole, thickness * 3.0))
    ring = plate.difference(cutter)

    handle = trimesh.creation.box(extents=(handle_length, handle_width, thickness))
    # Overlap the ring wall by a quarter of the handle length so the union is solid.
    handle.apply_translation((outer / 2.0 + handle_length / 2.0 - handle_length / 4.0, 0.0, 0.0))

    nut = ring.union(handle)
    nut.apply_translation(-nut.centroid)
    return nut


def build_square_peg(shaft: float, height: float, base: float, base_thickness: float) -> trimesh.Trimesh:
    """The peg the nut drops onto.

    Default is a plain rectangular column, which is both what robosuite's peg1 is (a
    single box geom) and what this rig's physical peg looks like in the D435 view: it
    stands on the table with no base plate. Pass a base > shaft to add a plinth.
    """
    post = trimesh.creation.box(extents=(shaft, shaft, height))
    if base > shaft and base_thickness > 0.0:
        post.apply_translation((0.0, 0.0, base_thickness + height / 2.0))
        plate = trimesh.creation.box(extents=(base, base, base_thickness))
        plate.apply_translation((0.0, 0.0, base_thickness / 2.0))
        post = plate.union(post)
    post.apply_translation(-post.centroid)
    return post


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
                   help="track_object.py's --output-dir, e.g. ~/Desktop/real2sim-toolkit/data/meshyai")
    p.add_argument("--camera", default="cam_1", help="track_object.py's --camera (default: cam_1)")
    p.add_argument("--nut-slug", default="square_nut",
                   help="--object-description with spaces replaced by underscores (default: square_nut)")
    p.add_argument("--peg-slug", default="square_peg", help="ditto for the peg (default: square_peg)")

    p.add_argument("--nut-outer", type=float, default=0.058, help="nut outer width, m (default 0.058)")
    p.add_argument("--nut-hole", type=float, default=0.038, help="through-hole width, m (default 0.038)")
    p.add_argument("--nut-thickness", type=float, default=0.012, help="plate thickness, m (default 0.012; measured ~15 mm above the table from D435 depth)")
    p.add_argument("--handle-length", type=float, default=0.048, help="handle length, m (default 0.048)")
    p.add_argument("--handle-width", type=float, default=0.018, help="handle width, m (default 0.018)")

    p.add_argument("--peg-shaft", type=float, default=0.032, help="peg shaft width, m (default 0.032)")
    p.add_argument("--peg-height", type=float, default=0.090, help="peg height, m (default 0.090; measured ~89 mm above the table from D435 depth)")
    p.add_argument("--peg-base", type=float, default=0.0,
                   help="optional plinth width, m. 0 (default) = plain column, matching this rig")
    p.add_argument("--peg-base-thickness", type=float, default=0.0, help="plinth thickness, m (default 0.0)")
    p.add_argument("--no-peg", action="store_true", help="only write the nut mesh")
    args = p.parse_args()

    outdir = args.outdir.expanduser()
    print(f"Writing GLBs under {outdir}")

    nut = build_square_nut(
        args.nut_outer, args.nut_hole, args.nut_thickness, args.handle_length, args.handle_width
    )
    write_glb(nut, outdir, args.camera, args.nut_slug)

    if not args.no_peg:
        peg = build_square_peg(args.peg_shaft, args.peg_height, args.peg_base, args.peg_base_thickness)
        write_glb(peg, outdir, args.camera, args.peg_slug)

    # The nut must clear the peg or the task is impossible; flag it rather than let it
    # surface as a mysterious tracking failure later.
    if not args.no_peg and args.nut_hole <= args.peg_shaft:
        print(f"\n  WARNING: hole ({args.nut_hole:.3f} m) <= peg shaft ({args.peg_shaft:.3f} m); "
              "the nut cannot fit over the peg. Check your measurements.")

    print("\nThese are PLACEHOLDER proportions, not measurements of this rig's parts.")
    print("scale_glb_to_pointcloud rescales them onto the observed cloud, so absolute size")
    print("is forgiving, but shape is not. Re-run with real numbers when you have them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
