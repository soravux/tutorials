"""
ColorChecker calibration demo.

Detects an X-Rite ColorChecker Classic in an image, fits several correction
models from the detected patches to the measured reference colours, and writes,
for every model:

    corrected_<stem>_<model>.png            corrected full image
    corrected_<stem>_<model>_clamped.png    same, but input clamped to the
                                            training range first (no extrapolation)
    comparison_<stem>_<model>.png           input vs corrected, plus a patch grid
                                            of nested squares: outer = input,
                                            middle = reference, inner = corrected

The numpy models are exactly those from the slide deck (least squares in linear
light): linear 3x3, polynomial (deg 2), root-polynomial (Finlayson). OpenCV's
cv2.ccm.ColorCorrectionModel is included as a library comparison.

Run:  python calibrate.py [image]      (default: example_image.jpg)
"""
import argparse
import os
import numpy as np
import colour
from PIL import Image, ImageDraw, ImageFont
from scipy.optimize import linear_sum_assignment
from colour_checker_detection import detect_colour_checkers_segmentation as detect

# ---------------------------------------------------------------- colour math
decode = lambda s: colour.cctf_decoding(s, function="sRGB")   # encoded -> linear
encode = lambda l: colour.cctf_encoding(l, function="sRGB")   # linear  -> encoded
to_lab = lambda s: colour.XYZ_to_Lab(colour.sRGB_to_XYZ(np.clip(s, 0, 1)))  # D65
slug = lambda name: name.replace(" ", "_")


def delta_e(corrected_srgb, ref_lab):
    return colour.delta_E(to_lab(corrected_srgb), ref_lab, method="CIE 1976")


def features(rgb, mode):
    """Design features for an array of linear RGB, shape (..., 3)."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    if mode == "linear":
        cols = [r, g, b]
    elif mode == "root":  # root-polynomial, degree 2 (exposure-invariant)
        cols = [r, g, b, np.sqrt(np.abs(r * g)),
                np.sqrt(np.abs(g * b)), np.sqrt(np.abs(b * r))]
    else:  # polynomial, degree 2
        cols = [r, g, b, r * r, g * g, b * b, r * g, g * b, b * r]
    return np.stack(cols, axis=-1)


# ---------------------------------------------------------------- image I/O (Pillow only)
def load_image(path):
    """Read an image as encoded-sRGB float in [0, 1]."""
    return np.asarray(Image.open(path).convert("RGB")).astype(np.float32) / 255.0


def to_pil(arr):
    return Image.fromarray((np.clip(arr, 0, 1) * 255 + 0.5).astype(np.uint8))


def save_png(path, arr):
    to_pil(arr).save(path)


# ---------------------------------------------------------------- reference
def reference():
    """Measured ColorChecker24 reference (After Nov 2014), reading order.

    The chart is measured under D50; sRGB and our CIELAB use D65. We therefore
    chromatically adapt the reference white from D50 to D65 first, so the neutral
    patches come out achromatic and there is no global colour cast.

    Returns target_lin (linear sRGB, unclipped), ref_lab (CIELAB, D65), and
    ref_disp (clipped encoded sRGB, for on-screen swatches).
    """
    cc = colour.CCS_COLOURCHECKERS["ColorChecker24 - After November 2014"]
    XYZ = colour.xyY_to_XYZ(np.array(list(cc.data.values())))
    D65 = colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D65"]
    XYZ = colour.chromatic_adaptation(
        XYZ, colour.xy_to_XYZ(cc.illuminant), colour.xy_to_XYZ(D65))  # D50 -> D65
    target_lin = colour.XYZ_to_sRGB(XYZ, apply_cctf_encoding=False)
    return target_lin, colour.XYZ_to_Lab(XYZ), np.clip(colour.XYZ_to_sRGB(XYZ), 0, 1)


# ---------------------------------------------------------------- models
class NumpyModel:
    """Least-squares correction in linear light: target_lin ~= features(src_lin) @ W."""

    def __init__(self, mode, src_lin, target_lin, clamp):
        self.mode = mode
        self.lo, self.hi = clamp                      # per-channel encoded-sRGB range
        F = features(src_lin, mode)                   # (24, k)
        self.W, *_ = np.linalg.lstsq(F, target_lin, rcond=None)  # (k, 3)

    def apply(self, img, clamp=False):
        if clamp:
            img = np.clip(img, self.lo, self.hi)
        return encode(np.clip(features(decode(img), self.mode) @ self.W, 0, 1))


class OpenCVModel:
    """cv2.ccm.ColorCorrectionModel fit to the same measured reference."""

    def __init__(self, src_srgb, ref_srgb, clamp):
        import cv2
        self.lo, self.hi = clamp
        m = cv2.ccm.ColorCorrectionModel(
            src_srgb.reshape(24, 1, 3).astype(np.float64),
            ref_srgb.reshape(24, 1, 3).astype(np.float64),
            cv2.ccm.COLOR_SPACE_sRGB)
        m.setCCM_TYPE(cv2.ccm.CCM_3x3)
        m.setLinear(cv2.ccm.LINEARIZATION_GAMMA)
        m.setLinearGamma(2.2)
        m.setLinearDegree(3)
        m.run()
        self.model = m

    def apply(self, img, clamp=False):
        if clamp:
            img = np.clip(img, self.lo, self.hi)
        return np.clip(self.model.infer(img.astype(np.float64)), 0, 1)


def build_models(detected, target_lin, ref_disp):
    clamp = (detected.min(0), detected.max(0))        # training range, encoded sRGB
    src_lin = decode(detected)
    models = {
        "Linear 3x3": NumpyModel("linear", src_lin, target_lin, clamp),
        "Polynomial": NumpyModel("poly", src_lin, target_lin, clamp),
        "Root-poly":  NumpyModel("root", src_lin, target_lin, clamp),
    }
    try:
        models["OpenCV ccm"] = OpenCVModel(detected, ref_disp, clamp)
    except Exception as exc:                          # pragma: no cover
        print(f"[warn] OpenCV ccm unavailable: {exc}")
    return models


def patch_errors(detected, model, ref_lab):
    corrected = model.apply(detected.reshape(24, 1, 3)).reshape(24, 3)
    return delta_e(corrected, ref_lab), corrected


# ---------------------------------------------------------------- correspondence
def find_correspondence(detected, target_lin, ref_lab, ref_disp):
    """Match detected swatches to reference patches regardless of chart orientation.

    A rotated/flipped chart means the detector's reading order need not match the
    reference order. We white-balance with the (achromatic) neutral patches, then
    propose an optimal assignment (Hungarian) in CIELAB, keeping whichever of
    {identity, Hungarian} gives the lower linear-fit residual (so an already
    correct chart is left untouched). Returns perm with detected[i] -> ref[perm[i]].
    """
    det_lin, ref_lin = decode(detected), decode(ref_disp)
    dn = np.argsort(detected.max(1) - detected.min(1))[:6]   # 6 least-chromatic
    rn = np.argsort(ref_disp.max(1) - ref_disp.min(1))[:6]   # reference neutrals
    dn = dn[np.argsort(det_lin[dn].sum(1))]                  # order by luminance
    rn = rn[np.argsort(ref_lin[rn].sum(1))]
    gain = ref_lin[rn].sum(0) / np.maximum(det_lin[dn].sum(0), 1e-6)
    cost = np.linalg.norm(
        to_lab(encode(np.clip(det_lin * gain, 0, 1)))[:, None] - ref_lab[None], axis=2)
    rows, cols = linear_sum_assignment(cost)
    hungarian = cols[np.argsort(rows)]

    def residual(perm):
        m = NumpyModel("linear", det_lin, target_lin[perm], (0, 1))
        corrected = m.apply(detected.reshape(24, 1, 3)).reshape(24, 3)
        return delta_e(corrected, ref_lab[perm]).mean()

    return min([np.arange(24), hungarian], key=residual)


# ---------------------------------------------------------------- comparison figure
def _font(size):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        try:
            return ImageFont.load_default(size)
        except Exception:
            return ImageFont.load_default()


def comparison(path, title, input_img, corrected_img, input_sw, ref_sw, corr_sw, dE):
    """Compose input | corrected plus a 3-square patch grid, with Pillow."""
    TW, M, CELL = 520, 24, 96
    thumb = lambda a: (lambda im: im.resize((TW, round(im.height * TW / im.width))))(to_pil(a))
    li, ri = thumb(input_img), thumb(corrected_img)
    grid_w, grid_h = 6 * CELL, 4 * CELL
    W = max(2 * TW + 3 * M, grid_w + 2 * M)
    top = 64
    H = top + li.height + 34 + grid_h + 60
    cv = Image.new("RGB", (W, H), (244, 244, 246))
    d = ImageDraw.Draw(cv)
    f_title, f_lab, f_sm = _font(22), _font(16), _font(11)

    d.text((M, 16), title, fill=(20, 20, 22), font=f_title)
    rx = W - TW - M
    d.text((M, 42), "Input", fill=(70, 70, 75), font=f_lab)
    d.text((rx, 42), "Corrected", fill=(70, 70, 75), font=f_lab)
    cv.paste(li, (M, top)); cv.paste(ri, (rx, top))
    for x in (M, rx):
        d.rectangle([x, top, x + TW, top + li.height], outline=(180, 180, 185))

    col = lambda s: tuple(int(round(c * 255)) for c in np.clip(s, 0, 1))
    gx, gy = (W - grid_w) // 2, top + li.height + 34
    for i in range(24):
        x0, y0 = gx + (i % 6) * CELL, gy + (i // 6) * CELL
        for frac, sw in ((0.94, input_sw[i]), (0.60, ref_sw[i]), (0.32, corr_sw[i])):
            pad = CELL * (1 - frac) / 2
            d.rectangle([x0 + pad, y0 + pad, x0 + CELL - pad, y0 + CELL - pad], fill=col(sw))
        d.text((x0 + 5, y0 + 3), f"{dE[i]:.1f}", fill=(15, 15, 15), font=f_sm)
    d.text((gx, gy + grid_h + 8),
           "nested squares:  outer = input    middle = reference    inner = corrected",
           fill=(90, 90, 95), font=f_lab)
    cv.save(path)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", nargs="?", default="example_image.jpg")
    args = ap.parse_args()
    stem = os.path.splitext(os.path.basename(args.image))[0]

    print(f"Reading {args.image} and detecting ColorChecker ...")
    img = load_image(args.image)
    detected = np.asarray(detect(img, additional_data=True)[0].swatch_colours)
    print(f"  detected {len(detected)} swatches")

    target_lin, ref_lab, ref_disp = reference()
    perm = find_correspondence(detected, target_lin, ref_lab, ref_disp)
    if not np.array_equal(perm, np.arange(24)):
        print("  re-oriented chart; correspondence permutation applied")
    target_lin, ref_lab, ref_disp = target_lin[perm], ref_lab[perm], ref_disp[perm]

    models = build_models(detected, target_lin, ref_disp)
    base_dE = delta_e(detected, ref_lab)
    print(f"\n  uncorrected   mean dE = {base_dE.mean():5.2f}   max = {base_dE.max():5.2f}")

    for name, m in models.items():
        dE, corr_sw = patch_errors(detected, m, ref_lab)
        print(f"  {name:12s}  mean dE = {dE.mean():5.2f}   max = {dE.max():5.2f}")

        reg = m.apply(img)
        save_png(f"corrected_{stem}_{slug(name)}.png", reg)
        save_png(f"corrected_{stem}_{slug(name)}_clamped.png", m.apply(img, clamp=True))
        title = (f"{stem}  —  {name}    mean ΔE {dE.mean():.2f} "
                 f"(was {base_dE.mean():.2f}),  max {dE.max():.2f}")
        comparison(f"comparison_{stem}_{slug(name)}.png", title,
                   img, reg, detected, ref_disp, corr_sw, dE)

    print(f"\n  wrote corrected_/comparison_ PNGs for {stem} (regular + clamped)")


if __name__ == "__main__":
    main()
