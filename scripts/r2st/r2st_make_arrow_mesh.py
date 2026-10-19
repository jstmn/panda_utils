#!/usr/bin/env python3
"""Generate the GLB for the drawer task's arrow so `real2sim-toolkit/examples/track_object.py`
can run without a Meshy API key. Companion to `r2st_make_nut_meshes.py` and
`r2st_make_pickplace_meshes.py`.

WHY THIS EXISTS
    track_object.py builds the tracked object's mesh with Meshy AI
    (`MeshUtils.generate_with_meshy`, src/r2st/utils.py:296) and asserts out when
    MESHY_API_KEY is unset, which it is on this machine. That helper short-circuits if a GLB
    already sits at

        <--outdir>/<--camera>__<object description, spaces -> underscores>/model_glb.glb

    so writing one there skips Meshy entirely.

WHERE THE NUMBERS COME FROM
    Length, width and thickness are measured: 245 x 60 x 20 mm.

    The proportions are not guessed either. The SAM mask from
    `r2st_sam_prompt_sweep.py --prompt "arrow shaped object"` on
    drawer__09-13_22-55-44/traj_0.h5 frame 0 is 6,646 px at 0.956 m, and its width profile
    along the principal axis reads (tail -> tip, mm):

        40 41 41 41 42 41 | 47 66 68 | 61 51 41 32 21

    A constant ~41 mm shaft over the first 43% of the length, shoulders at the widest point,
    then a taper to the tip. Hence --head-frac 0.55 and a shaft 0.61 of the full width. The
    profile's peak reads 68 mm against a measured 60 because the arrow is viewed obliquely
    and its 20 mm side face is in the silhouette; the ratios survive that, the absolute
    width does not, so the measured 60 is what is used.

    Sanity check on the result: this polygon's area is 8,011 mm^2, 54% of the 245x60
    bounding box. The mask projects to 7,302 mm^2, 50% of it, and projection of a tilted
    object can only shrink the figure. Consistent.

WHY THE SHAPE MATTERS MORE HERE THAN FOR A CUBE
    FoundationPose fits the mesh to depth, so a mesh that is the wrong size pushes the pose
    along the camera ray -- the failure that put the stack_cube red cube 49 mm below where
    physics demands. An arrow has the opposite problem from that cube in the good sense: it
    is long and asymmetric, so its yaw is observable, which is the whole reason it is worth
    tracking as an aux-pose target.

USAGE
    source ~/anaconda3/bin/activate r2st

    python scripts/r2st/r2st_make_arrow_mesh.py \
        --outdir ~/Desktop/real2sim-toolkit/data/meshyai720 \
        --camera cam_1 --slug arrow_shaped_object

    # the slug MUST be the prompt that SAM answers to, with spaces as underscores;
    # r2st_sam_prompt_sweep.py prints the candidates as cam_1__<slug>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import trimesh


def build_arrow(length: float, width: float, thickness: float,
                head_frac: float, shaft_frac: float) -> trimesh.Trimesh:
    """A flat arrow: rectangular shaft, triangular head, extruded by `thickness`.

    x runs tail -> tip, y across the width, z is the thickness. Centred on its centroid at
    the end, as the other mesh scripts do, because track_object.py registers the mesh's own
    frame against the mask and a mesh sitting off its origin shows up as a constant offset in
    every pose it reports.
    """
    head = length * head_frac
    hw, sw = width / 2.0, width * shaft_frac / 2.0
    outline = np.array([
        [0.0, -sw],                 # tail, one corner
        [length - head, -sw],       # where the shaft meets the head
        [length - head, -hw],       # shoulder
        [length, 0.0],              # tip
        [length - head, hw],        # shoulder
        [0.0, sw],                  # tail, other corner
    ])
    poly = trimesh.path.polygons.Polygon(outline)
    mesh = trimesh.creation.extrude_polygon(poly, height=thickness)
    mesh.apply_translation(-mesh.centroid)
    return mesh


def write_glb(mesh: trimesh.Trimesh, outdir: Path, camera: str, slug: str) -> Path:
    asset_dir = outdir / f"{camera}__{slug}"
    asset_dir.mkdir(parents=True, exist_ok=True)
    path = asset_dir / "model_glb.glb"
    mesh.export(path)
    ext = mesh.extents
    print(
        f"  {slug} -> {path}\n"
        f"      watertight={mesh.is_watertight}  volume={mesh.volume * 1e6:.1f} cm^3  "
        f"extents=({ext[0]*1000:.0f}, {ext[1]*1000:.0f}, {ext[2]*1000:.0f}) mm  "
        f"faces={len(mesh.faces)}"
    )
    return path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--outdir", type=Path, required=True,
                   help="track_object.py's --output-dir; the asset dir is created under it")
    p.add_argument("--camera", default="cam_1", help="track_object.py's --camera (default: cam_1)")
    p.add_argument("--slug", default="arrow_shaped_object",
                   help="the SAM prompt with spaces as underscores (default: arrow_shaped_object)")
    p.add_argument("--length", type=float, default=0.245, help="tail to tip, m (measured 0.245)")
    p.add_argument("--width", type=float, default=0.060, help="across the shoulders, m (measured 0.060)")
    p.add_argument("--thickness", type=float, default=0.020, help="m (measured 0.020)")
    p.add_argument("--head-frac", type=float, default=0.55,
                   help="head length as a fraction of the total (default 0.55, from the mask "
                        "width profile)")
    p.add_argument("--shaft-frac", type=float, default=0.61,
                   help="shaft width as a fraction of the full width (default 0.61, ditto)")
    cli = p.parse_args()

    mesh = build_arrow(cli.length, cli.width, cli.thickness, cli.head_frac, cli.shaft_frac)
    print(f"arrow {cli.length*1000:.0f} x {cli.width*1000:.0f} x {cli.thickness*1000:.0f} mm, "
          f"head {cli.head_frac:.2f} of length, shaft {cli.shaft_frac:.2f} of width")
    # The sweep's own numbers, so a changed shape can be checked against the mask it has to match.
    silhouette = cli.volume_area = float(mesh.volume / cli.thickness)
    box = cli.length * cli.width
    print(f"  silhouette {silhouette*1e6:.0f} mm^2, {silhouette/box*100:.0f}% of the "
          f"{cli.length*1000:.0f}x{cli.width*1000:.0f} box "
          f"(mask on frame 0 projects to 7302 mm^2, 50%)")
    write_glb(mesh, cli.outdir, cli.camera, cli.slug)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
