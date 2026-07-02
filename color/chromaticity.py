"""
Figures for the "XYZ -> xyY -> chromaticity horseshoe" build-up:

    horseshoe.png    CIE 1931 chromaticity diagram + Planckian locus (transparent bg)
    xyY_solid.mp4    the visible/object-colour solid in xyY (Y vertical), as a very
                     slow turntable (24 s per rotation = 1/4 the colour-solid speed),
                     with the spectral-locus horseshoe on the xy floor

Run:  python chromaticity.py
"""
import numpy as np
import colour
import colour.plotting as cp
import imageio.v2 as imageio
from colour.volume import solid_RoschMacAdam
from scipy.spatial import ConvexHull
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

BG, FG = "#0d1117", "#c9d1d9"
XYZ_FRAMES, XYZ_FPS = 480, 20          # 24 s per rotation (1/4 the colour-solid speed)
HALO = [pe.withStroke(linewidth=2.4, foreground="#0d1117")]   # keeps light text legible


# ---------------------------------------------------------------- horseshoe
def horseshoe():
    plt.rcParams.update({"font.size": 11})
    fig, ax = cp.plot_planckian_locus_in_chromaticity_diagram_CIE1931(
        ["A", "D50", "D65"], show=False,
        planckian_locus_labels=[2000, 3000, 4000, 6000, 10000],   # sparser -> less crowding
        planckian_locus_iso_temperature_lines_D_uv=0.06,          # bar length; also pushes labels out
        planckian_locus_colours="#e6edf3")
    ax.set_title("")
    ax.set_xlim(-0.05, 0.88); ax.set_ylim(-0.05, 0.86)
    ax.set_xlabel("x", color=FG, fontsize=18); ax.set_ylabel("y", color=FG, fontsize=18)
    ax.xaxis.label.set_path_effects(HALO); ax.yaxis.label.set_path_effects(HALO)
    ax.tick_params(colors=FG, labelsize=13)
    # transparent background; light text + dark halo so labels read on a dark slide
    fig.patch.set_alpha(0); ax.patch.set_alpha(0)
    for t in ax.texts:
        t.set_color(FG); t.set_fontsize(11); t.set_path_effects(HALO)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_path_effects(HALO)
    for ln in ax.lines:                                   # spectral locus, CCT rays
        if ln.get_color() in ("k", "black", "#000000", (0, 0, 0)):
            ln.set_color(FG)
    for sp in ax.spines.values():
        sp.set_color(FG)
    fig.savefig("horseshoe.png", dpi=130, bbox_inches="tight",
                transparent=True, pad_inches=0.12)
    plt.close(fig); plt.rcParams.update({"font.size": 10})
    print("  wrote horseshoe.png")


# ---------------------------------------------------------------- xyY solid
def _subdivide_triangles(coords, tris, vcol, max_area):
    """Recursively subdivide triangles exceeding max_area.
    Returns (new_coords, new_tris, new_vcol) with vertex colors interpolated."""
    new_verts = [np.array(c, dtype=float) for c in coords]
    new_vcol = [np.array(c, dtype=float) for c in vcol]
    new_tris = []
    queue = list(tris)
    while queue:
        t = queue.pop()
        a, b, c = t
        v0, v1, v2 = new_verts[a], new_verts[b], new_verts[c]
        area = np.linalg.norm(np.cross(v1 - v0, v2 - v0), ord=2) * 0.5
        if area > max_area:
            d01 = np.linalg.norm(v1 - v0, ord=2)
            d12 = np.linalg.norm(v2 - v1, ord=2)
            d02 = np.linalg.norm(v2 - v0, ord=2)
            if d01 >= d12 and d01 >= d02:
                mid = (v0 + v1) * 0.5
                mid_c = (new_vcol[a] + new_vcol[b]) * 0.5
                idx = len(new_verts)
                new_verts.append(mid); new_vcol.append(mid_c)
                queue.append((a, idx, c))
                queue.append((idx, b, c))
            elif d12 >= d01 and d12 >= d02:
                mid = (v1 + v2) * 0.5
                mid_c = (new_vcol[b] + new_vcol[c]) * 0.5
                idx = len(new_verts)
                new_verts.append(mid); new_vcol.append(mid_c)
                queue.append((a, b, idx))
                queue.append((a, idx, c))
            else:
                mid = (v0 + v2) * 0.5
                mid_c = (new_vcol[a] + new_vcol[c]) * 0.5
                idx = len(new_verts)
                new_verts.append(mid); new_vcol.append(mid_c)
                queue.append((a, b, idx))
                queue.append((idx, b, c))
        else:
            new_tris.append(t)
    return np.array(new_verts), np.array(new_tris), np.array(new_vcol)


def _xyz_to_xyy_jacobian(XYZ):
    """Jacobian of the XYZ -> xyY transform. Shape (N, 3, 3)."""
    S = XYZ.sum(1, keepdims=True)
    S = np.maximum(S, 1e-12)
    S2 = S * S
    # Actually let me recompute properly
    X, Y, Z = XYZ[:, 0:1], XYZ[:, 1:2], XYZ[:, 2:3]
    J = np.zeros((XYZ.shape[0], 3, 3))
    # dx/dX = Y/S^2, dx/dY = -X*Y/S^2, dx/dZ = -X*Z/S^2
    J[:, 0, 0] = Y / S2
    J[:, 0, 1] = -X * Y / S2
    J[:, 0, 2] = -X * Z / S2
    # dy/dX = Z/S^2, dy/dY = -Y*Z/S^2, dy/dZ = -Y*(X+Y)/S^2 = -Y*(S-Z)/S^2 ... let me just do it directly
    # y = Z/S, dy/dX = -Z*X/S^2, dy/dY = -Z*Y/S^2, dy/dZ = (S-Z)/S^2 = (X+Y)/S^2
    J[:, 1, 0] = -X * Z / S2
    J[:, 1, 1] = -Y * Z / S2
    J[:, 1, 2] = (X + Y) / S2
    # dY/dX = 0, dY/dY = 1, dY/dZ = 0
    J[:, 2, 1] = 1.0
    return J

# ---------------------------------------------------------------- xyY solid
def build_xyY(step_nm=1):
    shape = colour.SpectralShape(360, 780, step_nm)       # 1 nm -> dense, smooth mesh
    cmfs = colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"].copy().align(shape)
    illum = colour.SDS_ILLUMINANTS["D65"].copy().align(shape)
    XYZ = np.unique(np.round(solid_RoschMacAdam(cmfs, illum), 6), axis=0)
    tris = ConvexHull(XYZ).simplices
    vcol = np.clip(colour.XYZ_to_sRGB(XYZ), 0, 1)
    tcol = vcol[tris].mean(axis=1)
    s = XYZ.sum(1, keepdims=True)
    xy = np.where(s > 1e-6, XYZ[:, :2] / np.maximum(s, 1e-6), [0.3127, 0.3290])
    return np.column_stack([xy, XYZ[:, 1]]), tris, tcol, vcol, XYZ


def xyY_turntable():
    coords, tris, tcol, vcol, XYZ = build_xyY(step_nm=1)

    # --- Subdivide large flat facets ---
    areas = np.array([np.linalg.norm(np.cross(coords[t[1]] - coords[t[0]],
                 coords[t[2]] - coords[t[0]]), ord=2) * 0.5 for t in tris])
    median_area = np.median(areas)
    subdivided_coords, subdivided_tris, subdivided_vcol = _subdivide_triangles(
        coords, tris, vcol, max_area=median_area * 4)
    sub_tcol = subdivided_vcol[subdivided_tris].mean(axis=1)
    coords = subdivided_coords
    tris = subdivided_tris
    tcol = sub_tcol

    # --- Compute outward face normals in xyY space ---
    # Cross product of edges in xyY, outward by centroid test.
    # Using the actual solid centroid (not display-mean) for robustness
    # near the xyY fold where the old disp.mean(0) guess failed.
    tv = coords[tris]
    N = np.cross(tv[:, 1] - tv[:, 0], tv[:, 2] - tv[:, 0])
    lengths = np.linalg.norm(N, axis=1, keepdims=True)
    lengths = np.maximum(lengths, 1e-12)
    N = N / lengths
    centroid = coords.mean(0)
    face_centers = tv.mean(1)
    flip = np.einsum("ij,ij->i", N, face_centers - centroid) < 0
    N[flip] *= -1

    # --- Render with 2x supersampling ---
    ASPECT = (0.75, 0.85, 0.9)
    TARGET_W, TARGET_H = 600, 560
    SS = 2  # supersampling factor

    fig = plt.figure(figsize=(TARGET_W / 50 * SS, TARGET_H / 50 * SS),
                     dpi=50, facecolor=BG)
    ax = fig.add_subplot(111, projection="3d", facecolor=BG)
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.98)
    wl = np.arange(380, 701, 5)
    L = colour.XYZ_to_xy(colour.wavelength_to_XYZ(wl)); L = np.vstack([L, L[0]])
    ax.plot(L[:, 0], L[:, 1], 0, color="#8b949e", lw=1.6)
    ax.set_xlim(0, 0.75); ax.set_ylim(0, 0.85); ax.set_zlim(0, 1)
    ax.set_box_aspect(ASPECT)
    ax.set_xlabel("x", color=FG, fontsize=17, labelpad=8)
    ax.set_ylabel("y", color=FG, fontsize=17, labelpad=8)
    ax.set_zlabel("Y", color=FG, fontsize=17, labelpad=8)
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.set_pane_color((1, 1, 1, 0.03)); a.line.set_color((1, 1, 1, 0.2)); a.label.set_color(FG)
    ax.tick_params(colors="#8b949e", labelsize=12)

    elev, az0, solid = 22, -58, None
    er = np.radians(elev)
    with imageio.get_writer("xyY_solid.mp4", fps=XYZ_FPS, codec="libx264",
                            quality=6, macro_block_size=8,
                            output_params=["-pix_fmt", "yuv420p"]) as w:
        for az in np.linspace(az0, az0 + 360, XYZ_FRAMES, endpoint=False):
            ax.view_init(elev=elev, azim=az)
            ar = np.radians(az)
            eye = np.array([np.cos(er) * np.cos(ar), np.cos(er) * np.sin(ar), np.sin(er)])
            front = N @ eye > 0
            if solid is not None:
                solid.remove()
            solid = Poly3DCollection(coords[tris[front]], facecolors=tcol[front],
                                     edgecolors="none", antialiased=False)
            ax.add_collection3d(solid)
            fig.canvas.draw()
            # 2x supersampled render -> downsample by average-pooling 2x2 blocks
            buf = np.asarray(fig.canvas.buffer_rgba())
            bh, bw = buf.shape[:2]
            downsampled = (buf.reshape(bh // SS, SS, bw // SS, SS, 4)
                           .mean(axis=(1, 3)).astype(np.uint8))[..., :3]
            w.append_data(downsampled)
    plt.close(fig)
    print("  wrote xyY_solid.mp4")


if __name__ == "__main__":
    horseshoe()
    xyY_turntable()
