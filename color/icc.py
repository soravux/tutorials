"""
Build a custom ICC profile from scratch, then use it with Pillow.

An ICC matrix/TRC display profile is, at its core, just a few numbers:

    * the white point          (wtpt)   -- the PCS is always D50
    * the three primaries      (rXYZ, gXYZ, bXYZ)  -- each colorant's XYZ,
                                          Bradford-adapted to D50
    * one tone curve / channel (rTRC, gTRC, bTRC)  -- the OETF/EOTF, here a gamma

This script writes those tags into a valid ICC binary by hand (no colour-
management library), so you can see there is no magic: "taking the profile into
account" is exactly applying that matrix and those curves. We then hand the
bytes to Pillow (LittleCMS) to (a) embed the profile in a PNG and (b) transform
a pixel between our profile and sRGB to prove it is understood.

Run:  python icc.py
"""
import io
import struct
import numpy as np
from PIL import Image, ImageCms

# ---------------------------------------------------------------- colour math
BRADFORD = np.array([[ 0.8951,  0.2664, -0.1614],
                     [-0.7502,  1.7135,  0.0367],
                     [ 0.0389, -0.0685,  1.0296]])
D50_xy, D65_xy = (0.3457, 0.3585), (0.3127, 0.3290)
xy_to_XYZ = lambda x, y: np.array([x / y, 1.0, (1 - x - y) / y])


def rgb_to_xyz_matrix(primaries, white_xy):
    """RGB->XYZ matrix for given primary chromaticities and white point."""
    P = np.array([[*xy_to_XYZ(*primaries["R"])],
                  [*xy_to_XYZ(*primaries["G"])],
                  [*xy_to_XYZ(*primaries["B"])]]).T          # columns = primaries
    S = np.linalg.solve(P, xy_to_XYZ(*white_xy))             # per-primary scaling
    return P * S                                             # white -> R=G=B=1


def adapt_to_d50(M, src_white_xy):
    """Bradford-adapt a matrix whose white is src_white to the D50 PCS white."""
    src, dst = BRADFORD @ xy_to_XYZ(*src_white_xy), BRADFORD @ xy_to_XYZ(*D50_xy)
    A = np.linalg.inv(BRADFORD) @ np.diag(dst / src) @ BRADFORD
    return A @ M


# ---------------------------------------------------------------- ICC tag types
s15f16 = lambda x: struct.pack(">i", round(x * 65536))       # s15Fixed16
u8f8 = lambda g: struct.pack(">H", round(g * 256))           # u8Fixed8


def xyz_tag(v):
    return b"XYZ \x00\x00\x00\x00" + s15f16(v[0]) + s15f16(v[1]) + s15f16(v[2])


def curve_tag(gamma):
    # curveType with a single u8Fixed8 gamma value
    return b"curv\x00\x00\x00\x00" + struct.pack(">I", 1) + u8f8(gamma)


def desc_tag(text):
    # textDescriptionType (ICC v2): ASCII + empty Unicode + empty ScriptCode
    a = text.encode("ascii") + b"\x00"
    return (b"desc\x00\x00\x00\x00" + struct.pack(">I", len(a)) + a
            + struct.pack(">I", 0) + struct.pack(">I", 0)
            + struct.pack(">H", 0) + struct.pack(">B", 0) + b"\x00" * 67)


def text_tag(text):
    return b"text\x00\x00\x00\x00" + text.encode("ascii") + b"\x00"


def build_icc(primaries, white_xy=D65_xy, gamma=2.2, desc="Custom RGB"):
    """Assemble a minimal valid matrix/TRC RGB display profile (ICC v2.4)."""
    M = adapt_to_d50(rgb_to_xyz_matrix(primaries, white_xy), white_xy)
    D50 = xy_to_XYZ(*D50_xy)
    tags = [(b"desc", desc_tag(desc)),
            (b"wtpt", xyz_tag(D50)),
            (b"rXYZ", xyz_tag(M[:, 0])), (b"gXYZ", xyz_tag(M[:, 1])),
            (b"bXYZ", xyz_tag(M[:, 2])),
            (b"rTRC", curve_tag(gamma)), (b"gTRC", curve_tag(gamma)),
            (b"bTRC", curve_tag(gamma)),
            (b"cprt", text_tag("Public Domain"))]

    table_size = 4 + 12 * len(tags)
    offset = 128 + table_size
    blob, entries = b"", []
    for sig, payload in tags:
        entries.append((sig, offset, len(payload)))
        pad = (-len(payload)) % 4
        blob += payload + b"\x00" * pad
        offset += len(payload) + pad

    size = 128 + table_size + len(blob)
    header = bytearray(128)
    struct.pack_into(">I", header, 0, size)                  # profile size
    struct.pack_into(">I", header, 8, 0x02400000)           # version 2.4
    header[12:16], header[16:20], header[20:24] = b"mntr", b"RGB ", b"XYZ "
    header[36:40] = b"acsp"                                  # required signature
    header[68:80] = s15f16(D50[0]) + s15f16(D50[1]) + s15f16(D50[2])  # PCS illum.
    table = struct.pack(">I", len(tags))
    table += b"".join(s + struct.pack(">II", o, n) for s, o, n in entries)
    return bytes(header) + table + blob


# ---------------------------------------------------------------- demo
SRGB = {"R": (0.640, 0.330), "G": (0.300, 0.600), "B": (0.150, 0.060)}
ADOBE = {"R": (0.640, 0.330), "G": (0.210, 0.710), "B": (0.150, 0.060)}
# A deliberately nonsensical profile: primaries rotated (R<-B<-G<-R), a wildly
# off white point, and an inverted-ish gamma. Physically meaningless -- its only
# job is to prove that a viewer really does re-interpret the pixels through it.
FUNKY = {"R": (0.150, 0.060), "G": (0.640, 0.330), "B": (0.300, 0.600)}
FUNKY_WHITE = (0.24, 0.20)
# A saturated green: sRGB and AdobeRGB green primaries diverge the most, so a
# near-primary green shows the largest visible difference between the profiles.
PATCH_RGB = (40, 200, 70)


def show(name, profile_bytes, rgb=PATCH_RGB):
    prof = ImageCms.ImageCmsProfile(io.BytesIO(profile_bytes))
    srgb = ImageCms.createProfile("sRGB")
    # what does this RGB triple MEAN when read through `prof`? express it in sRGB.
    t = ImageCms.buildTransform(prof, srgb, "RGB", "RGB",
                                renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC)
    out = ImageCms.applyTransform(Image.new("RGB", (1, 1), rgb), t).getpixel((0, 0))
    print(f"  {name:16s} {ImageCms.getProfileName(prof).strip()[:34]:34s}"
          f"  {rgb} -> sRGB {out}")
    return prof


def save_tagged(img, path, profile_bytes):
    """Save `img` with an embedded ICC profile WITHOUT touching pixel values."""
    fmt = "JPEG" if path.lower().endswith((".jpg", ".jpeg")) else "PNG"
    kw = {"quality": 100, "subsampling": 0} if fmt == "JPEG" else {}
    img.save(path, format=fmt, icc_profile=profile_bytes, **kw)
    print(f"    wrote {path:22s} ({len(profile_bytes)} B profile embedded)")


def assert_same_pixels(path_a, path_b):
    """Prove two files carry byte-identical pixels (only the profile differs)."""
    a = np.asarray(Image.open(path_a).convert("RGB"))
    b = np.asarray(Image.open(path_b).convert("RGB"))
    same = a.shape == b.shape and np.array_equal(a, b)
    print(f"    pixels {path_a} vs {path_b}: "
          f"{'IDENTICAL' if same else 'DIFFER'}")


def main():
    print("Building custom ICC profiles from scratch ...\n")
    srgb_icc = build_icc(SRGB, gamma=2.2, desc="My sRGB-like profile")
    adobe_icc = build_icc(ADOBE, gamma=2.2, desc="My AdobeRGB-like profile")
    funky_icc = build_icc(FUNKY, white_xy=FUNKY_WHITE, gamma=0.55,
                          desc="Totally Broken Funky profile")

    print(f"  sRGB-like profile : {len(srgb_icc)} bytes")
    print(f"  Adobe-like profile: {len(adobe_icc)} bytes")
    print(f"  Funky profile     : {len(funky_icc)} bytes\n")
    print(f"Reading the SAME triple {PATCH_RGB} through each profile:")
    show("sRGB-like", srgb_icc)
    show("AdobeRGB-like", adobe_icc)
    show("Funky", funky_icc)
    print("\n  -> identical numbers, different colours: the profile decides meaning.\n")

    # ---- 1) a uniform patch: same pixels, two profiles -------------------
    print("Uniform patch (same pixels, different profiles):")
    patch = Image.new("RGB", (256, 256), PATCH_RGB)
    save_tagged(patch, "patch_srgb.png", srgb_icc)
    save_tagged(patch, "patch_adobe.png", adobe_icc)
    assert_same_pixels("patch_srgb.png", "patch_adobe.png")
    print("  -> open both: a viewer that honours ICC shows two different colours.\n")

    # ---- 2) the lvsn photo: same pixels, three profiles ------------------
    print("lvsn.jpg (same pixels, three profiles):")
    photo = Image.open("lvsn.jpg").convert("RGB")
    save_tagged(photo, "lvsn_srgb.jpg", srgb_icc)
    save_tagged(photo, "lvsn_adobe.jpg", adobe_icc)
    save_tagged(photo, "lvsn_funky.jpg", funky_icc)
    assert_same_pixels("lvsn_srgb.jpg", "lvsn_adobe.jpg")
    assert_same_pixels("lvsn_srgb.jpg", "lvsn_funky.jpg")
    print("  -> all three carry identical pixels; the funky one should look wild.")


if __name__ == "__main__":
    main()
