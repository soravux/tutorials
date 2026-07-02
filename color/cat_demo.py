"""
Chromatic-adaptation demo for the CAT slide.

Takes lvsn.jpg (an sRGB / D65 image) and Bradford-adapts its adopted white from
D65 to other illuminants — so the scene's neutrals shift toward each illuminant's
cast. This is the visible effect of "moving the white point":

    lvsn_D65.jpg   original (resized)
    lvsn_D50.jpg   adapted D65 -> D50   (subtle warm)
    lvsn_A.jpg     adapted D65 -> A      (strong tungsten orange)
    lvsn_FL2.jpg   adapted D65 -> F2     (fluorescent green)

Run:  python cat_demo.py
"""
import numpy as np
import colour
from PIL import Image

CCS = colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]


def white_XYZ(name):
    xy = CCS[name] if name in CCS else colour.XYZ_to_xy(
        colour.sd_to_XYZ(colour.SDS_ILLUMINANTS[name]) / 100)
    return colour.xy_to_XYZ(xy)


def main():
    im = Image.open("lvsn.jpg").convert("RGB")
    w = 640
    im = im.resize((w, round(im.height * w / im.width)), Image.LANCZOS)
    im.save("lvsn_D65.jpg", quality=88)

    XYZ = colour.sRGB_to_XYZ(np.asarray(im) / 255.0)     # decode + to XYZ (D65)
    D65 = white_XYZ("D65")

    for name in ("D50", "A", "FL2"):
        M = colour.adaptation.matrix_chromatic_adaptation_VonKries(
            D65, white_XYZ(name), transform="Bradford")     # adapt D65 -> name
        out = np.clip(colour.XYZ_to_sRGB(XYZ @ M.T), 0, 1)
        Image.fromarray((out * 255 + 0.5).astype("uint8")).save(
            f"lvsn_{name}.jpg", quality=88)
        print(f"  wrote lvsn_{name}.jpg")


if __name__ == "__main__":
    main()
