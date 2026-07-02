"""
Render the Rösch–MacAdam colour solid (the set of ALL physically realizable
object colours under an illuminant, a.k.a. the MacAdam limits / optimal colours)
as slow turntable videos, in two coordinate systems:

    solid_srgb.mp4      linear sRGB axes, with the sRGB unit cube for scale
    solid_ciecam02.mp4  CIECAM02 appearance axes (CAM02-UCS: a', b', J')

Same physical solid, two spaces — showing how "all real colours" dwarfs a device
gamut, and how its shape changes between a linear device space and a perceptual
appearance space. Data: colour-science solid_RoschMacAdam under D65.

Run:  python solids.py
"""
import numpy as np
import colour
import imageio.v2 as imageio
from colour.volume import solid_RoschMacAdam
from scipy.spatial import ConvexHull
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

BG, FG = "#0d1117", "#c9d1d9"
FRAMES, FPS = 480, 20         # 480 frames @ 20 fps = one very slow 24 s turntable loop


def build(step_nm):
    """Boundary mesh of the solid at a given spectral sampling (finer = smoother)."""
    shape = colour.SpectralShape(360, 780, step_nm)
    cmfs = colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"].copy().align(shape)
    illum = colour.SDS_ILLUMINANTS["D65"].copy().align(shape)
    XYZ = np.unique(np.round(solid_RoschMacAdam(cmfs, illum), 6), axis=0)  # Y in [0,1]
    tris = ConvexHull(XYZ).simplices
    tcol = np.clip(colour.XYZ_to_sRGB(XYZ), 0, 1)[tris].mean(axis=1)       # face colour
    return XYZ, tris, tcol


def style(ax, title, labels):
    ax.set_title(title, color=FG, fontsize=13, pad=0)
    ax.set_xlabel(labels[0], color=FG, fontsize=10)
    ax.set_ylabel(labels[1], color=FG, fontsize=10)
    ax.set_zlabel(labels[2], color=FG, fontsize=10)
    ax.set_facecolor(BG)
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.set_pane_color((1, 1, 1, 0.03))
        a.line.set_color((1, 1, 1, 0.2))
        a.label.set_color(FG)
    ax.tick_params(colors="#7d8794", labelsize=8)


def turntable(coords, tris, tcol, out, title, labels, elev=18, cube=False, alpha=1.0):
    fig = plt.figure(figsize=(6.0, 5.6), dpi=100, facecolor=BG)
    ax = fig.add_subplot(111, projection="3d", facecolor=BG)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=0.94)

    faces = np.concatenate([tcol, np.full((len(tcol), 1), alpha)], axis=1)  # RGBA
    ax.add_collection3d(Poly3DCollection(coords[tris], facecolors=faces, edgecolors="none"))

    lo, hi = coords.min(0), coords.max(0)
    if cube:  # sRGB unit cube [0,1]^3 wireframe, drawn on top
        r = [0, 1]
        for s in r:
            for t in r:
                ax.plot([s, s], [t, t], r, color="#ffe066", lw=2.0, zorder=10)
                ax.plot([s, s], r, [t, t], color="#ffe066", lw=2.0, zorder=10)
                ax.plot(r, [s, s], [t, t], color="#ffe066", lw=2.0, zorder=10)
        lo, hi = np.minimum(lo, 0), np.maximum(hi, 1)

    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect((hi - lo))
    style(ax, title, labels)

    with imageio.get_writer(out, fps=FPS, codec="libx264", quality=6,
                            macro_block_size=8,
                            output_params=["-pix_fmt", "yuv420p"]) as w:
        for azim in np.linspace(-62, -62 + 360, FRAMES, endpoint=False):
            ax.view_init(elev=elev, azim=azim)
            fig.canvas.draw()
            w.append_data(np.asarray(fig.canvas.buffer_rgba())[..., :3])
    plt.close(fig)
    print(f"  wrote {out}")


def main():
    # sRGB solid: semi-transparent so the gamut cube shows through -> coarse mesh
    # (facets are hidden by the transparency; keeps the cube clearly visible).
    XYZ, tris, tcol = build(5)
    rgb = colour.XYZ_to_sRGB(XYZ, apply_cctf_encoding=False)   # linear RGB
    turntable(rgb, tris, tcol, "solid_srgb.mp4",
              "Rösch–MacAdam solid in linear sRGB  (yellow cube = sRGB gamut)",
              ("R", "G", "B"), cube=True, alpha=0.28)

    # CIECAM02 solid: opaque -> fine mesh for smooth, wireframe-free shading.
    XYZ, tris, tcol = build(2)
    XYZ_w = colour.xy_to_XYZ(
        colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D65"]) * 100
    spec = colour.XYZ_to_CIECAM02(XYZ * 100, XYZ_w, L_A=64, Y_b=20)
    Jab = np.nan_to_num(colour.JMh_CIECAM02_to_CAM02UCS(
        np.stack([spec.J, spec.M, spec.h], axis=-1)))          # J', a', b'
    turntable(Jab[:, [1, 2, 0]], tris, tcol, "solid_ciecam02.mp4",
              "Rösch–MacAdam solid in CIECAM02  (CAM02-UCS: a', b', J')",
              ("a'", "b'", "J'"))


if __name__ == "__main__":
    main()
