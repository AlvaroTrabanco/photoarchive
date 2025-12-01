#!/usr/bin/env python3
"""
XMP Archive Harvester — v2 (fixed)
----------------------------------

Reads Lightroom Classic metadata from .XMP sidecars (and associated image files)
or embedded XMP in JPG/TIFF/DNG (via exiftool). Produces CSV/JSON (and optional
static site) plus thumbnails.

New flags:
  --jobs N           (0=auto [CPU], 1=sequential)
  --thumbs-scope     selected|strict
                     selected = all selected records (default)
                     strict   = only records with print_id OR negative_sleeve
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional
import xml.etree.ElementTree as ET
import subprocess, shutil
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
import tempfile

def _decode_raw_to_temp_jpeg_with_sips(src: Path, max_edge: int = 0) -> Optional[Path]:
    if not shutil.which("sips"):
        return None
    try:
        tmpdir = Path(tempfile.gettempdir())
        tmp = tmpdir / (src.stem + ".sips.jpg")
        cmd = ["sips", src.as_posix(), "--setProperty", "format", "jpeg", "--out", tmp.as_posix()]
        if max_edge and max_edge > 0:
            cmd[1:1] = ["-Z", str(max_edge)]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if res.returncode == 0 and tmp.exists() and tmp.stat().st_size > 0:
            return tmp
    except Exception:
        pass
    return None

from PIL import Image, ImageOps

# -------------------- config & helpers --------------------

NS = {
    "x": "adobe:ns:meta/",
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "dc": "http://purl.org/dc/elements/1.1/",
    "lr": "http://ns.adobe.com/lightroom/1.0/",
    "xmp": "http://ns.adobe.com/xap/1.0/",
    "crs": "http://ns.adobe.com/camera-raw-settings/1.0/",
    "tiff": "http://ns.adobe.com/tiff/1.0/",
}

IMAGE_EXTS = [
    ".jpg", ".jpeg", ".tif", ".tiff", ".dng", ".nef", ".cr2", ".cr3", ".arw", ".orf", ".raf", ".rw2"
]

# RAW formats we need to decode via sips for reliable orientation/crops
RAW_EXTS = {'.nef', '.cr2', '.cr3', '.arw', '.orf', '.raf', '.rw2', '.dng'}

def _apply_tiff_orientation(img: Image.Image, orientation: Optional[int]) -> Image.Image:
    """
    Apply TIFF/EXIF Orientation (1..8).
    1=normal, 2=mirror-H, 3=180, 4=mirror-V, 5=transpose, 6=90 CW, 7=transverse, 8=90 CCW
    """
    if not orientation or orientation == 1:
        return img
    try:
        if   orientation == 2: return img.transpose(Image.FLIP_LEFT_RIGHT)
        elif orientation == 3: return img.transpose(Image.ROTATE_180)
        elif orientation == 4: return img.transpose(Image.FLIP_TOP_BOTTOM)
        elif orientation == 5: return img.transpose(Image.TRANSPOSE)
        elif orientation == 6: return img.transpose(Image.ROTATE_270)  # 90 CW
        elif orientation == 7: return img.transpose(Image.TRANSVERSE)
        elif orientation == 8: return img.transpose(Image.ROTATE_90)   # 90 CCW
    except Exception:
        return img
    return img



from math import radians, sin, cos

def _apply_manual_perspective(im: Image.Image,
                              pv: Optional[float], ph: Optional[float],
                              px: Optional[float], py: Optional[float],
                              pscale: Optional[float], paspect: Optional[float],
                              prot: Optional[float]) -> Image.Image:
    """
    Approximate LR Manual Transform with a projective 'quad' mapping.
    This is heuristic, aimed at thumbnails; tweak KEYSTONE_STRENGTH to match taste.
    Order: keystone (pv/ph) → aspect/scale → offset → rotate (small).
    """
    W, H = im.size
    if W < 2 or H < 2:
        return im

    # Normalize sliders to [-1,1] where applicable
    def norm100(v): return None if v is None else (max(-100.0, min(100.0, v)) / 100.0)
    n_pv = norm100(pv)
    n_ph = norm100(ph)
    n_px = norm100(px)
    n_py = norm100(py)
    n_as = norm100(paspect)

    # Scale (% → factor). Default 100 → 1.0
    s = 1.0
    if pscale is not None:
        try:
            s = float(pscale) / 100.0
        except Exception:
            s = 1.0
        s = max(0.01, min(4.0, s))

    # Heuristic strengths (tune these):
    KEYSTONE_STRENGTH = 0.40   # how aggressively pv/ph pinch the trapezoid
    SHIFT_PIXELS = 0.20        # fraction of half-size for X/Y shifts at 100

    # Base quad is full image rectangle in source (UL, LL, LR, UR) — order for QUAD:
    # Pillow's QUAD expects: (UL, LL, LR, UR) in source coords.
    ul = [0.0, 0.0]
    ll = [0.0, float(H)]
    lr = [float(W), float(H)]
    ur = [float(W), 0.0]

    # Apply vertical keystone: move top inward/outward symmetrically
    if n_pv is not None and abs(n_pv) > 1e-6:
        ax = (W * KEYSTONE_STRENGTH) * n_pv
        ul[0] += ax
        ur[0] -= ax

    # Apply horizontal keystone: move left/right vertically in opposite directions
    if n_ph is not None and abs(n_ph) > 1e-6:
        ay = (H * KEYSTONE_STRENGTH) * n_ph
        ul[1] += ay
        ll[1] -= ay
        ur[1] -= ay
        lr[1] += ay

    # Aspect (LR's slider stretches/compresses width vs height a bit).
    # Positive aspect: widen; negative: tighten.
    if n_as is not None and abs(n_as) > 1e-6:
        k = 1.0 + 0.25 * n_as  # gentle ±25%
        cx, cy = W * 0.5, H * 0.5
        for p in (ul, ll, lr, ur):
            p[0] = cx + (p[0] - cx) * k

    # Scale about center
    if abs(s - 1.0) > 1e-6:
        cx, cy = W * 0.5, H * 0.5
        for p in (ul, ll, lr, ur):
            p[0] = cx + (p[0] - cx) * s
            p[1] = cy + (p[1] - cy) * s

    # Shifts (X/Y) in percent of half-size
    if n_px is not None and abs(n_px) > 1e-6:
        dx = (W * 0.5) * SHIFT_PIXELS * n_px
        for p in (ul, ll, lr, ur): p[0] += dx
    if n_py is not None and abs(n_py) > 1e-6:
        dy = (H * 0.5) * SHIFT_PIXELS * n_py
        for p in (ul, ll, lr, ur): p[1] += dy

    # Small extra rotate (Transform panel's rotate; often redundant with CropAngle)
    if prot is not None and abs(prot) > 1e-6:
        th = radians(float(prot))
        c, s_ = cos(th), sin(th)
        cx, cy = W * 0.5, H * 0.5
        def rot(p):
            x, y = p
            x0, y0 = x - cx, y - cy
            return [cx + x0 * c - y0 * s_, cy + x0 * s_ + y0 * c]
        ul = rot(ul); ll = rot(ll); lr = rot(lr); ur = rot(ur)

    # Perform projective mapping: map this source quad to a rectangle of same size.
    quad = tuple(ul + ll + lr + ur)
    return im.transform((W, H), Image.QUAD, quad, resample=Image.BICUBIC)


def _map_box_by_orientation(box, orientation):
    """Map normalized (L,T,R,B) into the coordinate system after applying
    a TIFF/EXIF orientation. Returns a normalized box with L<=R, T<=B."""
    if not box or not orientation or orientation == 1:
        return box
    L, T, R, B = map(float, box)

    def order(x1, x2): 
        return (min(x1, x2), max(x1, x2))

    # mapping formulas on normalized [0..1] coordinates
    if orientation == 2:     # mirror horizontal
        L, R = 1-R, 1-L
    elif orientation == 3:   # rotate 180
        L, R = 1-R, 1-L
        T, B = 1-B, 1-T
    elif orientation == 4:   # mirror vertical
        T, B = 1-B, 1-T
    elif orientation == 5:   # transpose (flip across main diagonal): (x,y)->(y,x)
        L, T, R, B = T, L, B, R
    elif orientation == 6:   # 90 CW: (x,y)->(y,1-x)
        L, T, R, B = T, 1-R, B, 1-L
    elif orientation == 7:   # transverse (flip across anti-diagonal): (x,y)->(1-y,1-x)
        L, T, R, B = 1-B, 1-R, 1-T, 1-L
    elif orientation == 8:   # 90 CCW: (x,y)->(1-y,x)
        L, T, R, B = 1-B, L, 1-T, R

    # clamp + re-order just in case
    L, T, R, B = [max(0.0, min(1.0, v)) for v in (L, T, R, B)]
    L, R = order(L, R); T, B = order(T, B)
    return (L, T, R, B)




@dataclass
class Record:
    file_path: str
    xmp_path: str
    title: Optional[str] = None
    description: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    hierarchical_keywords: List[str] = field(default_factory=list)
    print_id: Optional[str] = None
    negative_sleeve: Optional[str] = None
    frame_number: Optional[str] = None
    film: Optional[str] = None
    thumb_path: Optional[str] = None
    crop_left: Optional[float] = None
    crop_top: Optional[float] = None
    crop_right: Optional[float] = None
    crop_bottom: Optional[float] = None
    crop_angle: Optional[float] = None
    exif_normalize_for_crop: bool = True
    exif_orientation: Optional[int] = None
    negative_scan: bool = False

    # Lightroom Transform sliders
    persp_vertical: Optional[float] = None
    persp_horizontal: Optional[float] = None
    persp_rotate: Optional[float] = None
    persp_scale: Optional[float] = None
    persp_aspect: Optional[float] = None
    persp_x: Optional[float] = None
    persp_y: Optional[float] = None

    def to_row(self) -> dict:
        return {
            "file_path": self.file_path,
            "xmp_path": self.xmp_path,
            "title": self.title or "",
            "description": self.description or "",
            "keywords": "; ".join(self.keywords) if self.keywords else "",
            "keywords_text": "; ".join(self.keywords) if self.keywords else "",
            "keywords_list": list(self.keywords) if self.keywords else [],
            "hierarchical_keywords": "; ".join(self.hierarchical_keywords) if self.hierarchical_keywords else "",
            "print_id": self.print_id or "",
            "negative_sleeve": self.negative_sleeve or "",
            "frame_number": self.frame_number or "",
            "film": self.film or "",
            "thumb_path": self.thumb_path or "",
            "negative_scan": bool(self.negative_scan),
            "missing": bool(self.negative_scan),
        }



def _crs_float(desc: ET.Element, tag: str) -> Optional[float]:
    val = desc.get(f"{{{NS['crs']}}}{tag}")
    if val is None:
        node = desc.find(f"crs:{tag}", NS)
        if node is not None and node.text:
            val = node.text
    if val is None:
        return None
    try:
        return float(val)
    except Exception:
        return None
    


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Harvest Lightroom .XMP metadata into CSV/JSON and optional static site."
    )
    p.add_argument("--root", required=True, help="Root directory to scan recursively")
    p.add_argument("--csv", required=True, help="Output CSV file path")
    p.add_argument("--sqlite", default=None, help="Optional SQLite output path")
    p.add_argument("--json", default=None, help="Optional JSON catalog output path")
    p.add_argument("--site", default=None, help="Optional output directory for a static viewer site")
    p.add_argument("--select", choices=["all", "auto", "keyword", "prefix"], default="auto",
                   help="Subset selection strategy for archive membership")
    p.add_argument("--require-keyword", action="append", default=[],
                   help="Keyword token that must be present (repeatable)")
    p.add_argument("--require-prefix", action="append", default=[],
                   help="Keyword prefix; at least one token starting with this must be present (repeatable)")
    p.add_argument("--log", default="INFO", help="Logging level (DEBUG, INFO, WARNING)")
    p.add_argument("--thumbs", default=None, help="Optional directory to write thumbnails")
    p.add_argument("--thumb-size", type=int, default=512, help="Max thumbnail edge in px (default 512)")
    p.add_argument("--jobs", type=int, default=0,
                   help="Worker threads for parsing (0=auto: CPU count; 1=sequential)")

    # NEW: control when index.html is written
    p.add_argument(
        "--html",
        choices=["force", "auto", "skip"],
        default="auto",
        help="When building --site: force=always rewrite index.html; auto=write only if missing; skip=never write"
    )

    # NEW: thumbs scope
    p.add_argument("--thumbs-scope", choices=["selected", "strict"], default="selected",
                   help="Which items get thumbnails: 'selected' (default) or 'strict' (print_id or negative_sleeve)")
    return p.parse_args()

def make_thumbnail(
    image_path: Path,
    thumbs_dir: Path,
    max_px: int,
) -> Optional[Path]:
    """
    Build a thumbnail with **no** Lightroom crop / transform logic.
    We assume we're working with already-exported JPGs that look correct.

    Behaviour:
      - Try `sips -Z` first for speed (macOS).
      - Fallback to Pillow: open → resize → save.
    """
    try:
        thumbs_dir.mkdir(parents=True, exist_ok=True)
        out = thumbs_dir / (image_path.stem + ".jpg")

        # Fast path via sips (only for non-RAW; here we expect JPGs anyway)
        if shutil.which("sips") and image_path.suffix.lower() not in RAW_EXTS:
            cmd = [
                "sips",
                "-Z",
                str(max_px),
                image_path.as_posix(),
                "--setProperty",
                "format",
                "jpeg",
                "--out",
                out.as_posix(),
            ]
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if res.returncode == 0 and out.exists() and out.stat().st_size > 0:
                return out

        # Fallback: Pillow (no EXIF auto-rotation, no crop)
        with Image.open(image_path.as_posix()) as im:
            im.thumbnail((max_px, max_px))
            im.convert("RGB").save(out.as_posix(), "JPEG", quality=85)

        return out if out.exists() else None

    except Exception as e:
        logging.debug("make_thumbnail failed for %s: %s", image_path, e)
        return None
def find_associated_image(xmp_path: Path) -> Optional[Path]:
    stem = xmp_path.stem
    parent = xmp_path.parent
    for ext in IMAGE_EXTS:
        candidate = parent / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None

def extract_dc_bag_items(desc: ET.Element, bag_name: str) -> List[str]:
    items: List[str] = []
    if ":" in bag_name:
        prefix, local = bag_name.split(":", 1)
        qname = f"{{{NS[prefix]}}}{local}"
    else:
        qname = bag_name
    bag = desc.find(f".//{qname}/rdf:Bag", NS)
    if bag is None:
        return items
    for li in bag.findall("rdf:li", NS):
        if li.text:
            items.append(li.text.strip())
    return items

def extract_lr_crop(desc: ET.Element) -> dict:
    """
    Read Lightroom crop from XMP:
      crs:HasCrop (bool-ish), crs:CropLeft/Top/Right/Bottom (0..1), crs:CropAngle (degrees)
    Returns {} if no crop found.
    """
    crs_ns = NS.get("crs") or "http://ns.adobe.com/camera-raw-settings/1.0/"
    NS["crs"] = crs_ns

    def _get_float(qname: str) -> Optional[float]:
        el = desc.get(qname)
        if el is None:
            return None
        try:
            return float(el)
        except Exception:
            return None

    def _get_bool(qname: str) -> Optional[bool]:
        el = desc.get(qname)
        if el is None:
            return None
        v = str(el).strip().lower()
        if v in ("true", "1", "yes"):
            return True
        if v in ("false", "0", "no"):
            return False
        return None

    has_crop = _get_bool(f"{{{crs_ns}}}HasCrop")

    def _child_text(local: str) -> Optional[str]:
        el = desc.find(f"crs:{local}", NS)
        return el.text.strip() if (el is not None and el.text) else None

    def _float_any(local: str) -> Optional[float]:
        v = _get_float(f"{{{crs_ns}}}{local}")
        if v is not None:
            return v
        t = _child_text(local)
        if t is None:
            return None
        try:
            return float(t)
        except Exception:
            return None

    left   = _float_any("CropLeft")
    top    = _float_any("CropTop")
    right  = _float_any("CropRight")
    bottom = _float_any("CropBottom")
    angle  = _float_any("CropAngle")

    vals = [left, top, right, bottom, angle]
    if (has_crop is False) or all(v is None for v in vals[:-1]):
        return {}

    out = {}
    if left   is not None: out["left"] = left
    if top    is not None: out["top"] = top
    if right  is not None: out["right"] = right
    if bottom is not None: out["bottom"] = bottom
    if angle  is not None: out["angle"] = angle
    return out

def extract_embedded_xmp(image_path: Path) -> Optional[ET.Element]:
    if not shutil.which("exiftool"):
        logging.debug("ExifTool not found; cannot read embedded XMP from %s", image_path)
        return None
    try:
        res = subprocess.run(["exiftool", "-b", "-XMP", image_path.as_posix()],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if res.returncode != 0 or not res.stdout.strip():
            logging.debug("No embedded XMP found in %s", image_path)
            return None
        try:
            root = ET.fromstring(res.stdout)
            return root
        except ET.ParseError:
            wrapped = b'<?xml version="1.0"?>\n<x:xmpmeta xmlns:x="adobe:ns:meta/">' + res.stdout + b"</x:xmpmeta>"
            return ET.fromstring(wrapped)
    except Exception as e:
        logging.warning("Failed to extract embedded XMP from %s: %s", image_path, e)
        return None

def _get_crs_float_from_desc(desc: ET.Element, tag: str) -> Optional[float]:
    val = desc.get(f"{{{NS['crs']}}}{tag}")
    if val is None:
        node = desc.find(f"crs:{tag}", NS)
        if node is not None and node.text:
            val = node.text
    if val is None:
        return None
    try:
        return float(val)
    except Exception:
        return None

def read_lr_crop_and_angle_from_root(xmp_root: ET.Element) -> tuple[Optional[tuple[float,float,float,float]], Optional[float]]:
    desc = xmp_root.find(".//rdf:RDF/rdf:Description", NS)
    if desc is None:
        return (None, None)

    L = _get_crs_float_from_desc(desc, "CropLeft")
    T = _get_crs_float_from_desc(desc, "CropTop")
    R = _get_crs_float_from_desc(desc, "CropRight")
    B = _get_crs_float_from_desc(desc, "CropBottom")
    angle = _get_crs_float_from_desc(desc, "CropAngle")

    crop = None
    if None not in (L, T, R, B):
        L = max(0.0, min(1.0, L))
        T = max(0.0, min(1.0, T))
        R = max(0.0, min(1.0, R))
        B = max(0.0, min(1.0, B))
        if R > L and B > T:
            crop = (L, T, R, B)

    return (crop, angle)

def find_xmp_for_image(image_path: Path) -> Optional[Path]:
    sidecar = image_path.with_suffix(".xmp")
    return sidecar if sidecar.exists() else None

def get_lr_cropangle_for_path(path: Path) -> tuple[Optional[tuple[float,float,float,float]], Optional[float]]:
    if path.suffix.lower() == ".xmp":
        try:
            root = ET.parse(path).getroot()
            return read_lr_crop_and_angle_from_root(root)
        except Exception:
            return (None, None)

    sidecar = find_xmp_for_image(path)
    if sidecar:
        try:
            root = ET.parse(sidecar).getroot()
            crop, ang = read_lr_crop_and_angle_from_root(root)
            if crop or ang is not None:
                return (crop, ang)
        except Exception:
            pass

    root = extract_embedded_xmp(path)
    if root is not None:
        return read_lr_crop_and_angle_from_root(root)

    return (None, None)

def read_lr_crop_from_root(xmp_root: ET.Element) -> Optional[tuple[float, float, float, float]]:
    desc = xmp_root.find(".//rdf:RDF/rdf:Description", NS)
    if desc is None:
        return None

    def _getf(tag: str) -> Optional[float]:
        el = desc.get(f"{{{NS['crs']}}}{tag}")
        if el is None:
            node = desc.find(f"crs:{tag}", NS)
            if node is not None and node.text:
                el = node.text
        if el is None:
            return None
        try:
            return float(el)
        except Exception:
            return None

    L = _getf("CropLeft")
    T = _getf("CropTop")
    R = _getf("CropRight")
    B = _getf("CropBottom")

    if None in (L, T, R, B):
        return None

    L = max(0.0, min(1.0, L))
    T = max(0.0, min(1.0, T))
    R = max(0.0, min(1.0, R))
    B = max(0.0, min(1.0, B))
    if R <= L or B <= T:
        return None
    return (L, T, R, B)

def get_lr_crop_for_path(path: Path) -> Optional[tuple[float, float, float, float]]:
    if path.suffix.lower() == ".xmp":
        try:
            root = ET.parse(path).getroot()
            return read_lr_crop_from_root(root)
        except Exception:
            return None

    sidecar = find_xmp_for_image(path)
    if sidecar:
        try:
            root = ET.parse(sidecar).getroot()
            crop = read_lr_crop_from_root(root)
            if crop:
                return crop
        except Exception:
            pass

    root = extract_embedded_xmp(path)
    if root is not None:
        return read_lr_crop_from_root(root)

    return None

def _assign_keywords(rec: Record) -> None:
    """
    Normalize keywords (keep as list) and derive print_id from tokens.
    """
    # 1) normalize keywords list from dc:subject bag
    kw = rec.keywords or []
    kw = [k.strip() for k in kw if isinstance(k, str) and k.strip()]
    rec.keywords = kw

    # 2) collect candidates to infer print_id (from keywords + hierarchical segments)
    candidates: List[str] = list(kw)
    for hk in (rec.hierarchical_keywords or []):
        for seg in str(hk).split("|"):
            seg = seg.strip()
            if seg:
                candidates.append(seg)

    # 3) derive print_id if missing; supports bare IDs and "PrintID: ..."
    if not rec.print_id:
        for k in candidates:
            m = re.match(r"(?i)(?:print[_ -]?id[:=]\s*)?([A-Za-z]*\d{4,}-\d{1,3})$", k)
            if m:
                rec.print_id = m.group(1).strip()
                break

NEG_SCAN_TOKENS = {"35mmscan", "6x6mmscan"}

def _derive_film_sleeve_frame(rec: Record) -> None:
    # 1) from hierarchical keywords like "Archive|Film|Kodak Tri-X 400"
    for hk in (rec.hierarchical_keywords or []):
        parts = [p.strip() for p in str(hk).split("|") if p and p.strip()]
        for idx, p in enumerate(parts):
            p_low = p.lower()
            if p_low == "film" and rec.film is None and idx + 1 < len(parts):
                rec.film = " | ".join(parts[idx + 1:]) or parts[idx + 1]
            if (p_low in ("sleeve", "negativesleeve", "negative sleeve")) and rec.negative_sleeve is None and idx + 1 < len(parts):
                rec.negative_sleeve = parts[idx + 1]
            if p_low == "frame" and rec.frame_number is None and idx + 1 < len(parts):
                rec.frame_number = parts[idx + 1]

    # 2) fallbacks from flat tokens like "NegativeSleeve:57", "Frame:5", "Film: Kodak Tri-X 400"
    for t in (rec.keywords or []):
        s = str(t).strip()
        if rec.film is None:
            m = re.match(r"(?i)^film[:=]\s*(.+)$", s)
            if m: rec.film = m.group(1).strip()
        if rec.negative_sleeve is None:
            m = re.match(r"(?i)^(?:neg(?:ative)?[_\s-]?sleeve|sleeve)[:=]\s*([A-Za-z0-9._ -]+)$", s)
            if m: rec.negative_sleeve = m.group(1).strip()
        if rec.frame_number is None:
            m = re.match(r"(?i)^frame[:=]\s*([A-Za-z0-9._ -]+)$", s)
            if m: rec.frame_number = m.group(1).strip()

def _derive_negative_scan_flag(rec: Record) -> None:
    # flat tokens
    toks = [t.strip().lower().replace(" ", "") for t in (rec.keywords or [])]
    # hierarchical tokens → check each segment too
    for hk in (rec.hierarchical_keywords or []):
        for seg in hk.split("|"):
            toks.append(seg.strip().lower().replace(" ", ""))

    if any(t in NEG_SCAN_TOKENS for t in toks):
        rec.negative_scan = True

def parse_xmp_from_root(xmp_root: ET.Element,
                        image_path_for_record: Optional[Path],
                        xmp_path_for_record: Path) -> Record:
    desc = xmp_root.find(".//rdf:RDF/rdf:Description", NS)
    if desc is None:
        return Record(file_path=str(image_path_for_record or ""),
                      xmp_path=str(xmp_path_for_record))

    # Title / description
    title = None
    title_alt = desc.find("dc:title/rdf:Alt/rdf:li", NS)
    if title_alt is not None and title_alt.text:
        title = title_alt.text.strip()
    else:
        title_simple = desc.find("dc:title", NS)
        if title_simple is not None and title_simple.text:
            title = title_simple.text.strip()

    description = None
    desc_alt = desc.find("dc:description/rdf:Alt/rdf:li", NS)
    if desc_alt is not None and desc_alt.text:
        description = desc_alt.text.strip()

    # Keywords
    keywords = extract_dc_bag_items(desc, "dc:subject")
    h_keywords = extract_dc_bag_items(desc, "lr:hierarchicalSubject")

    # Crop (if present)
    lr_crop = extract_lr_crop(desc)

    # Build record
    rec = Record(
        file_path=str(image_path_for_record or ""),
        xmp_path=str(xmp_path_for_record),
        title=title,
        description=description,
        keywords=keywords,
        hierarchical_keywords=h_keywords,
    )

    # Orientation from XMP (TIFF/EXIF semantics 1..8)
    try:
        o = desc.get(f"{{{NS['tiff']}}}Orientation")
        if o is None:
            node = desc.find("tiff:Orientation", NS)
            if node is not None and node.text:
                o = node.text
        if o is not None:
            rec.exif_orientation = int(str(o).strip())
    except Exception:
        pass

    # Apply crop values if present
    if lr_crop:
        rec.crop_left   = lr_crop.get("left")
        rec.crop_top    = lr_crop.get("top")
        rec.crop_right  = lr_crop.get("right")
        rec.crop_bottom = lr_crop.get("bottom")
        rec.crop_angle  = lr_crop.get("angle")

    # --- Lightroom manual Transform sliders (best-effort; read regardless of crop) ---
    rec.persp_vertical   = _crs_float(desc, "PerspectiveVertical")
    rec.persp_horizontal = _crs_float(desc, "PerspectiveHorizontal")
    rec.persp_rotate     = _crs_float(desc, "PerspectiveRotate")
    rec.persp_scale      = _crs_float(desc, "PerspectiveScale")
    rec.persp_aspect     = _crs_float(desc, "PerspectiveAspect")
    rec.persp_x          = _crs_float(desc, "PerspectiveX")
    rec.persp_y          = _crs_float(desc, "PerspectiveY")

    # Derivations
    _assign_keywords(rec)
    _derive_negative_scan_flag(rec)
    _derive_film_sleeve_frame(rec)
    return rec

def parse_xmp(xmp_or_image_file: Path) -> Record:
    if xmp_or_image_file.suffix.lower() == ".xmp":
      try:
          tree = ET.parse(xmp_or_image_file)
          root = tree.getroot()
      except ET.ParseError as e:
          logging.warning("XML parse error in %s: %s", xmp_or_image_file, e)
          return Record(file_path=str(find_associated_image(xmp_or_image_file) or ""), xmp_path=str(xmp_or_image_file))

      image = find_associated_image(xmp_or_image_file)
      rec = parse_xmp_from_root(root, image, xmp_or_image_file)

      # ✅ NEW: respect EXIF orientation even when crop is present
      # Lightroom doesn’t store 90° rotations in XMP for sidecar-based raws;
      # use EXIF orientation flag from the image itself.
      rec.exif_normalize_for_crop = False

      return rec

    root = extract_embedded_xmp(xmp_or_image_file)
    if root is None:
        return Record(file_path=str(xmp_or_image_file), xmp_path=str(xmp_or_image_file))
    rec = parse_xmp_from_root(root, xmp_or_image_file, xmp_or_image_file)
    rec.exif_normalize_for_crop = True  # embedded: DO EXIF-normalize for crop
    return rec

def walk_targets(root: Path) -> Iterable[Path]:
    sidecar_keys = set()
    for p in root.rglob("*.xmp"):
        if p.is_file():
            sidecar_keys.add((str(p.parent.resolve()), p.stem))
            yield p

    exts = tuple(e.lower() for e in IMAGE_EXTS)
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() in exts:
            # skip temp decodes like _DSC9171.sips.jpg / .sips.jpeg
            if p.name.endswith(".sips.jpg") or p.name.endswith(".sips.jpeg"):
                continue
            key = (str(p.parent.resolve()), p.stem)
            if key not in sidecar_keys:
                yield p

def write_csv(rows: List[Record], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].to_row().keys()) if rows else [
        "file_path","xmp_path","title","description","keywords",
        "hierarchical_keywords","print_id","negative_sleeve","frame_number","film",
        "thumb_path","negative_scan","missing"
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r.to_row())

def write_sqlite(rows: List[Record], out_sqlite: Path) -> None:
    out_sqlite.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(out_sqlite)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS photos (
            id INTEGER PRIMARY KEY,
            file_path TEXT,
            xmp_path TEXT,
            title TEXT,
            description TEXT,
            keywords TEXT,
            hierarchical_keywords TEXT,
            print_id TEXT,
            negative_sleeve TEXT,
            frame_number TEXT,
            film TEXT,
            thumb_path TEXT,
            negative_scan INTEGER,
            missing INTEGER
        )
        """
    )
    cur.execute("DELETE FROM photos")
    cur.executemany(
        """
        INSERT INTO photos (
            file_path, xmp_path, title, description, keywords, hierarchical_keywords,
            print_id, negative_sleeve, frame_number, film, thumb_path, negative_scan, missing
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                r.file_path, r.xmp_path, r.title, r.description,
                "; ".join(r.keywords), "; ".join(r.hierarchical_keywords),
                r.print_id, r.negative_sleeve, r.frame_number, r.film, r.thumb_path,
                1 if r.negative_scan else 0,
                1 if r.negative_scan else 0,
            )
            for r in rows
        ],
    )
    conn.commit(); conn.close()

def write_json(rows: List[Record], out_json: Path) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump([r.to_row() for r in rows], f, ensure_ascii=False, indent=2)

def write_index_html(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    BUILD_TAG = "ARCHIVE-HTML v2.4 — grid/list + sort + status + editor + clear filters"
    html = """<!doctype html>
<!-- ARCHIVE-HTML v2.4 — grid/list + sort + status + editor + clear filters -->
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Photo Archive</title>
<style>
  *{box-sizing:border-box}
  :root {
    --gap:16px; --border:#ddd; --muted:#666; --bg:#fff; --bg2:#fafafa;
    --shadow:0 8px 24px rgba(0,0,0,.12);
    --card-min: 260px; /* zoomable grid cell */
  }
  body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Ubuntu;margin:24px;background:var(--bg2)}
  header.toolbar{display:flex;gap:12px;align-items:center;margin-bottom:16px;flex-wrap:wrap}
  input,select,button,textarea{padding:6px 8px;font:inherit}
  .cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(var(--card-min),1fr));gap:var(--gap)}
  .cards.cards--list{display:flex;flex-direction:column}
  .card{border:1px solid var(--border);border-radius:12px;padding:12px;background:var(--bg)}
  .card--active{outline:2px solid orange;outline-offset:2px}
  .muted{color:var(--muted);font-size:12px}
  .row{margin:6px 0}
  a{word-break:break-all}
  .pill{display:inline-block;border:1px solid #ccc;border-radius:999px;padding:2px 8px;margin-right:6px;font-size:12px}
  .pill--missing{border-color:#e99;background:#fff4f4;color:#b40000}
  .list-row{display:flex;gap:12px;align-items:flex-start}
  .thumb{max-width:100%;border-radius:8px;margin-bottom:8px;cursor:pointer}
  .list-row .thumb{max-width:160px}
  .thumb--placeholder{display:flex;align-items:center;justify-content:center;
                      aspect-ratio:4/3;border:2px dashed #ddd;border-radius:8px;background:#fafafa;
                      font-size:12px;color:#888}
  .group{display:flex;gap:8px;align-items:center}
  .ms{position:relative}
  .ms__button{display:flex;align-items:center;gap:8px;border:1px solid var(--border);background:var(--bg);
              border-radius:10px;padding:6px 10px;cursor:pointer}
  .ms__button-badge{background:#eee;border-radius:999px;padding:2px 8px;font-size:12px}
  .ms__popover{position:absolute;top:calc(100% + 8px);left:0;z-index:20;width:280px;background:var(--bg);
               border:1px solid var(--border);border-radius:12px;box-shadow:var(--shadow);padding:10px}
  .ms__popover[hidden]{display:none}
  .ms__search{width:100%;border:1px solid var(--border);border-radius:8px;padding:6px 8px}
  .ms__list{margin-top:8px;max-height:220px;overflow:auto;border:1px solid var(--border);
            border-radius:10px;padding:6px}
  .ms__item{display:flex;align-items:center;gap:8px;padding:4px 2px}
  .ms__actions{display:flex;gap:8px;margin-top:8px}
  .ms__chips{display:flex;flex-wrap:wrap;gap:6px;margin-top:8px}
  .chip{display:flex;align-items:center;gap:6px;background:#f0f0f0;border-radius:999px;padding:2px 8px;font-size:12px}
  .chip button{border:none;background:transparent;cursor:pointer;padding:0 2px;font-size:14px;line-height:1}

  /* Modal */
  .modal-backdrop{position:fixed;inset:0;background:rgba(0,0,0,.35);display:none;align-items:center;justify-content:center;z-index:50}
  .modal{width:min(980px,95vw);max-height:90vh;overflow:auto;background:var(--bg);border:1px solid var(--border);border-radius:12px;box-shadow:var(--shadow);padding:16px}
  .modal header{display:flex;justify-content:space-between;align-items:center;margin-bottom:8px}
  .modal h2{margin:0}
  .modal .rows{display:flex;flex-direction:column;gap:12px;margin-top:8px}
  .series-row{border:1px solid var(--border);border-radius:10px;padding:12px;background:#fff}
  .series-grid{display:grid;gap:8px;grid-template-columns:repeat(6,1fr)}
  .series-grid label{display:flex;flex-direction:column;font-size:12px;color:#444}
  .series-grid input, .series-grid textarea{width:100%;border:1px solid var(--border);border-radius:8px}
  .series-grid textarea{grid-column:1/-1;min-height:60px;resize:vertical}
  .row-actions{display:flex;gap:8px;margin-top:8px}
  .modal footer{display:flex;gap:8px;justify-content:flex-end;margin-top:12px}
  .btn-danger{background:#ffecec;border:1px solid #ffb3b3}
  .btn-primary{background:#111;border:1px solid #111}
  .btn{border:1px solid var(--border);background:#fff;border-radius:10px;padding:6px 10px;cursor:pointer}
  .note{font-size:12px;color:#555}
</style>
</head>
<body>
<header class="toolbar">
  <h1 style="margin:0">Photo Archive</h1>

  <!-- Film multi-select -->
  <div class="ms" id="filmMs">
    <button type="button" class="ms__button" id="filmMsBtn" aria-haspopup="listbox" aria-expanded="false">
      <span>Pick films</span>
      <span class="ms__button-badge" id="filmMsCount">All</span>
    </button>
    <div class="ms__popover" id="filmMsPop" hidden>
      <input id="filmMsSearch" class="ms__search" placeholder="Search films…" />
      <div class="ms__actions">
        <button id="filmMsSelectAll" type="button">Select all (filtered)</button>
        <button id="filmMsClear" type="button">Clear</button>
      </div>
      <div id="filmMsList" class="ms__list" role="listbox" aria-multiselectable="true"></div>
      <div class="ms__chips" id="filmMsChips"></div>
    </div>
  </div>

  <!-- Sleeves multi-select -->
  <div class="ms" id="sleevesMs">
    <button type="button" class="ms__button" id="msBtn" aria-haspopup="listbox" aria-expanded="false">
      <span>Pick sleeve</span>
      <span class="ms__button-badge" id="msCount">All</span>
    </button>
    <div class="ms__popover" id="msPop" hidden>
      <input id="msSearch" class="ms__search" placeholder="Search sleeves…" />
      <div class="ms__actions">
        <button id="msSelectAll" type="button">Select all (filtered)</button>
        <button id="msClear" type="button">Clear</button>
      </div>
      <div id="msList" class="ms__list" role="listbox" aria-multiselectable="true"></div>
      <div class="ms__chips" id="msChips"></div>
    </div>
  </div>

  <!-- Series multi-select -->
  <div class="ms" id="seriesMs">
    <button type="button" class="ms__button" id="seriesMsBtn" aria-haspopup="listbox" aria-expanded="false">
      <span>Pick series</span>
      <span class="ms__button-badge" id="seriesMsCount">All</span>
    </button>
    <div class="ms__popover" id="seriesMsPop" hidden>
      <input id="seriesMsSearch" class="ms__search" placeholder="Search series…" />
      <div class="ms__actions">
        <button id="seriesMsSelectAll" type="button">Select all (filtered)</button>
        <button id="seriesMsClear" type="button">Clear</button>
      </div>
      <div id="seriesMsList" class="ms__list" role="listbox" aria-multiselectable="true"></div>
      <div class="ms__chips" id="seriesMsChips"></div>
    </div>
  </div>

  <!-- View + Sort -->
  <div class="group">
    <label for="viewMode">View</label>
    <select id="viewMode">
      <option value="grid" selected>Grid</option>
      <option value="list">List</option>
    </select>
  </div>

  <div class="group">
    <label for="sortKey">Sort</label>
    <select id="sortKey">
      <option value="print_id">PrintID (A→Z)</option>
      <option value="film">Film (A→Z)</option>
      <option value="title">Title (A→Z)</option>
      <option value="negative_sleeve">Sleeve (A→Z)</option>
    </select>
  </div>

  <!-- Status -->
  <div class="group">
    <label for="status">Status</label>
    <select id="status">
      <option value="">All</option>
      <option value="digitized">Digitized only</option>
      <option value="missing">Missing only</option>
      <option value="only_neg">Negative scans only</option>
    </select>
  </div>

  <!-- Zoom -->
  <div class="group" style="align-items:center">
    <label for="zoom">Zoom</label>
    <input id="zoom" type="range" min="180" max="520" step="20" value="260" />
  </div>

  <button id="toggleThumbs">Hide thumbnails</button>
  <button id="clearFilters" class="btn">Clear filters</button>
  <button id="openEditor" class="btn">Edit archive…</button>

  <button id="openEditor" class="btn">Edit archive…</button>
  <span id="archiveCount" class="muted" style="margin-left:auto">0/0</span>
</header>

<div id="cards" class="cards"></div>

<!-- Modal: Archive editor -->
<div id="editorBackdrop" class="modal-backdrop" aria-hidden="true">
  <div class="modal" role="dialog" aria-modal="true" aria-labelledby="editorTitle">
    <header>
      <h2 id="editorTitle">Archive editor</h2>
    </header>

    <p class="note">Edit your archive series (e.g. <code>2022-1..120</code>, <code>M2024-01..36</code>). Changes are staged in this editor until you click <strong>Save</strong>.</p>

    <div class="rows" id="seriesRows"></div>

    <div class="row-actions">
      <button id="addSeries" class="btn">+ Add series</button>
      <input type="file" id="importSeries" accept="application/json" style="display:none">
      <button id="importSeriesBtn" class="btn">Import JSON…</button>
      <button id="resetSeries" class="btn">Reset draft to file</button>
    </div>

    <footer>
      <button id="saveSeries" class="btn btn-primary">Save</button>
      <button id="discardSeries" class="btn">Close without saving</button>
    </footer>
  </div>
</div>

<!-- Lightbox for enlarged photo -->
<div id="lightboxBackdrop" class="modal-backdrop" aria-hidden="true">
  <div class="modal" id="lightboxModal" aria-modal="true" role="dialog">
    <header style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
      <div>
        <button id="lightboxPrev" class="btn">&larr; Prev</button>
        <button id="lightboxNext" class="btn">Next &rarr;</button>
      </div>
      <button id="lightboxClose" class="btn">× Close</button>
    </header>
    <img id="lightboxImage"
         alt=""
         style="max-width:100%;height:auto;display:block;margin:0 auto 4px;" />
  </div>
</div>

<script>
console.log("ARCHIVE-HTML v2.4 — grid/list + sort + status + editor + clear filters loaded");

(async function(){
  // -------- helpers
  async function getJSON(url){
    const res = await fetch(url + (url.includes('?')?'':'?ts=') + Date.now(), {cache:'no-store'});
    if(!res.ok) throw new Error(url+' → '+res.status);
    const txt = await res.text();
    if(!txt.trim()) return null;
    return JSON.parse(txt);
  }
  function dl(filename, dataObj){
    const blob = new Blob([JSON.stringify(dataObj, null, 2)], {type:'application/json'});
    const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = filename;
    document.body.appendChild(a); a.click(); a.remove(); setTimeout(()=>URL.revokeObjectURL(a.href), 1000);
  }

  const STORAGE_KEY = 'series_overrides_v1';

  // -------- load data
  const catalog = (await getJSON('catalog.json')) || [];
  const seriesFile = (await getJSON('series.json')) || {series:[]};
  const overrides = (function(){ try { return JSON.parse(localStorage.getItem(STORAGE_KEY) || 'null'); } catch(e){ return null; }})();
  let registry = (overrides && overrides.series) ? overrides : seriesFile;

  // -------- expand placeholders for missing
  function expandSeries(s){
    const prefix = String((s.prefix != null ? s.prefix : (s.id != null ? s.id : '')));
    const first  = Number(s.first_number == null ? 1 : s.first_number);
    const last   = Number(s.last_number  == null ? first : s.last_number);
    const fmt = String(s.format || '{prefix}-{n}');
    const ids = [];
    for(let n=first; n<=last; n++){
      let id = fmt.replace('{prefix}', prefix);
      id = id.replace(/\{n(?::0(\d+))?\}/, function(_,w){ return String(n).padStart(w?Number(w):0,'0'); });
      id = id.replace('{n}', String(n));
      ids.push(id);
    }
    return ids;
  }

  // Group all items by print_id
  const byId = new Map();
  for (const r of catalog) {
    const id = String(r.print_id || '');
    if (!id) continue;
    if (!byId.has(id)) byId.set(id, []);
    byId.get(id).push(r);
  }

  // A print is "fulfilled" if ANY record for that id is NOT a negative scan
  function isFulfilledPrint(id){
    const arr = byId.get(id) || [];
    return arr.some(r => !r.negative_scan);  // uses the field from catalog.json
  }

  // Create placeholders ONLY for series IDs that are NOT fulfilled
  const placeholders = [];
  for (const s of (registry.series || [])){
    for (const id of expandSeries(s)){
      if (!isFulfilledPrint(id)) {
        placeholders.push({
          file_path:'', xmp_path:'', title:s.title||'', description:s.description||'',
          keywords:'', keywords_text:'', keywords_list:[], hierarchical_keywords:'',
          print_id:id, negative_sleeve:'', frame_number:'', film:'', thumb_path:'',
          missing:true, series_id:s.id || s.prefix || ''
        });
      }
    }
  }

  const HELPER_URL = "http://127.0.0.1:8787";
  async function helper(action, absPath){
    try{
      const res = await fetch(`${HELPER_URL}/api/${action}`, {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({path: absPath})
      });
      const out = await res.json();
      if(!out.ok) throw new Error(out.error || 'Helper error');
    }catch(e){
      alert(`Could not ${action}:
${e.message}

Is filehelper.py running?`);
    }
  }

  const items = catalog.concat(placeholders);

  const totalCount = items.length;
  const countEl = document.getElementById('archiveCount');
  function setCount(n){ if(countEl) countEl.textContent = `${n}/${totalCount}`; }
  setCount(totalCount); // initial: before any filters apply

  // Backfill + normalize fields for the UI
  for (const i of items) {
    // Treat negative scans as "Missing"
    if (i.negative_scan) i.missing = true;

    // Build a token pool
    const pool = []
      .concat(Array.isArray(i.keywords_list) ? i.keywords_list : [])
      .concat((i.keywords_text || '').split(/[;,]/))
      .concat((i.hierarchical_keywords || '').split(/[;]+/)); // keep pipe groups intact for parsing

    // 1) From hierarchical paths like "Archive|Film|Kodak Tri-X 400"
    if (!i.film || !i.negative_sleeve || !i.frame_number) {
      for (const raw of (i.hierarchical_keywords || '').split(/[;]+/)) {
        const parts = String(raw || '').split('|').map(s => s.trim()).filter(Boolean);
        const idxFilm = parts.findIndex(p => /^film$/i.test(p));
        if (!i.film && idxFilm >= 0 && parts[idxFilm + 1]) {
          i.film = parts.slice(idxFilm + 1).join(' | '); // keep full remainder if nested
        }
        const idxSleeve = parts.findIndex(p => /^(negative)?sleeve$/i.test(p) || /^negative\s*sleeve$/i.test(p));
        if (!i.negative_sleeve && idxSleeve >= 0 && parts[idxSleeve + 1]) {
          i.negative_sleeve = parts[idxSleeve + 1];
        }
        const idxFrame = parts.findIndex(p => /^frame$/i.test(p));
        if (!i.frame_number && idxFrame >= 0 && parts[idxFrame + 1]) {
          i.frame_number = parts[idxFrame + 1];
        }
      }
    }

    // 2) Fallbacks from flat tokens like "Frame:5" or "NegativeSleeve:57"
    if (!i.film || !i.negative_sleeve || !i.frame_number) {
      for (const raw of pool) {
        const t = String(raw || '').trim();
        if (!i.film) {
          const mFilm = t.match(/^(?:film[:=]\s*)(.+)$/i);
          if (mFilm) i.film = mFilm[1].trim();
        }
        if (!i.negative_sleeve) {
          const mSleeve = t.match(/^(?:neg(?:ative)?[_\s-]?sleeve|sleeve)\s*[:=]\s*([A-Za-z0-9._ -]+)$/i);
          if (mSleeve) i.negative_sleeve = mSleeve[1].trim();
        }
        if (!i.frame_number) {
          const mFrame = t.match(/^frame\s*[:=]\s*([A-Za-z0-9._ -]+)$/i);
          if (mFrame) i.frame_number = mFrame[1].trim();
        }
      }
    }
  }

  // Ensure every real item gets a series_id from its print_id prefix
  (function assignSeriesIds(){
    const defs = (registry.series || []);
    function idOrPrefix(s){ return (s.id && String(s.id).trim()) || (s.prefix || ''); }
    for (const it of items){
      if (it.series_id || !it.print_id) continue;
      const pid = String(it.print_id);
      for (const s of defs){
        const pref = idOrPrefix(s);
        if (pref && pid.startsWith(pref)){
          it.series_id = idOrPrefix(s);
          break;
        }
      }
    }
  })();

  function toFileURL(absPath, opts) {
    opts = opts || {};
    var folder = !!opts.folder;

    if (!absPath) return "";
    // Windows-safe: convert backslashes → slashes without regex pitfalls
    var p = String(absPath).split("\\\\").join("/");

    if (folder) p = p.replace(/\/[^\/]+$/, "/");   // strip filename → keep trailing slash
    if (!p.startsWith("/")) p = "/" + p;           // ensure leading slash
    return "file://" + encodeURI(p);               // file:///… (properly encoded)
  }

  // -------- UI references
  const $ = s => document.querySelector(s);
  function norm(v){ return String(v == null ? '' : v).toLowerCase(); }
  function naturalSort(a,b){ return String(a).localeCompare(String(b), undefined, {numeric:true, sensitivity:'base'}); }

  const cards = $('#cards');
  const statusSel = $('#status');
  const viewMode = $('#viewMode');
  const sortKey = $('#sortKey');
  const toggleThumbsBtn = $('#toggleThumbs');
  const zoom = $('#zoom');
  const clearFiltersBtn = $('#clearFilters');

  // Lightbox state: list of items with thumbs + current index in that list
  let lightboxItems = [];
  let lightboxCurrentIndex = -1;

  // Lightbox DOM
  const lightboxBackdrop = document.getElementById('lightboxBackdrop');
  const lightboxImage = document.getElementById('lightboxImage');
  const lightboxClose = document.getElementById('lightboxClose');
  const lightboxPrev = document.getElementById('lightboxPrev');
  const lightboxNext = document.getElementById('lightboxNext');


    function openLightboxForIndex(idx) {
      if (!lightboxItems || !lightboxItems.length) return;
      if (idx < 0 || idx >= lightboxItems.length) return;

      lightboxCurrentIndex = idx;

      const item = lightboxItems[lightboxCurrentIndex];
      const title = (item.print_id || item.title || 'Untitled');
      const src = item.thumb_path ? (item.thumb_path + '?ts=' + Date.now()) : '';

      if (!src) return;

      lightboxImage.src = src;
      lightboxImage.alt = title;
      lightboxBackdrop.style.display = 'flex';
      lightboxBackdrop.setAttribute('aria-hidden', 'false');

      // Re-render so the correct card gets the orange outline
      render();
    }

    function closeLightbox() {
      lightboxBackdrop.style.display = 'none';
      lightboxBackdrop.setAttribute('aria-hidden', 'true');
      lightboxImage.src = '';
      lightboxCurrentIndex = -1;
      // Re-render to clear any active card
      render();
    }

    function showPrev() {
      if (!lightboxItems || lightboxItems.length === 0) return;
      if (lightboxCurrentIndex <= 0) return;
      openLightboxForIndex(lightboxCurrentIndex - 1);
    }

    function showNext() {
      if (!lightboxItems || lightboxItems.length === 0) return;
      if (lightboxCurrentIndex < 0 || lightboxCurrentIndex >= lightboxItems.length - 1) return;
      openLightboxForIndex(lightboxCurrentIndex + 1);
    }

  lightboxClose.addEventListener('click', closeLightbox);
  lightboxPrev.addEventListener('click', showPrev);
  lightboxNext.addEventListener('click', showNext);

  // Click outside the modal closes the lightbox
  lightboxBackdrop.addEventListener('click', (e) => {
    if (e.target === lightboxBackdrop) {
      closeLightbox();
    }
  });

  // film & sleeves & series multiselect DOM
  const filmMs = $('#filmMs'), filmMsBtn = $('#filmMsBtn'), filmMsPop = $('#filmMsPop'),
        filmMsSearch = $('#filmMsSearch'), filmMsList = $('#filmMsList'),
        filmMsCount = $('#filmMsCount'), filmMsChips = $('#filmMsChips'),
        filmMsSelectAll = $('#filmMsSelectAll'), filmMsClear = $('#filmMsClear');

  const ms=$('#sleevesMs'), msBtn=$('#msBtn'), msPop=$('#msPop'), msSearch=$('#msSearch'),
        msList=$('#msList'), msCount=$('#msCount'), msChips=$('#msChips'),
        msSelectAll=$('#msSelectAll'), msClear=$('#msClear');

  const seriesMs = $('#seriesMs'), seriesMsBtn = $('#seriesMsBtn'), seriesMsPop = $('#seriesMsPop'),
        seriesMsSearch = $('#seriesMsSearch'), seriesMsList = $('#seriesMsList'),
        seriesMsCount = $('#seriesMsCount'), seriesMsChips = $('#seriesMsChips'),
        seriesMsSelectAll = $('#seriesMsSelectAll'), seriesMsClear = $('#seriesMsClear');

  // Build list of series names from registry (fall back id/prefix)
  const allSeries = Array
    .from(new Set((registry.series || []).map(function(s){
      return String(s.id != null && String(s.id).trim() ? s.id
           : (s.prefix != null ? s.prefix : ''));
    }).filter(function(x){ return !!x; })))
    .sort(naturalSort);

  const selectedSeries = new Set(); let filteredSeries = allSeries.slice();

  function updateSeriesCount(){
    seriesMsCount.textContent = selectedSeries.size===0 ? 'All'
      : (Array.from(selectedSeries).sort(naturalSort).slice(0,2).join(', ')
        + (selectedSeries.size>2?(' +'+(selectedSeries.size-2)):''));
  }

  function renderSeriesList(){
    seriesMsList.innerHTML = '';
    for (var k=0; k<filteredSeries.length; k++){
      const sName = filteredSeries[k];
      const id='series_'+btoa(unescape(encodeURIComponent(sName))).replace(/=/g,'');
      const row=document.createElement('label'); row.className='ms__item'; row.htmlFor=id;
      const cb=document.createElement('input'); cb.type='checkbox'; cb.id=id; cb.checked=selectedSeries.has(sName);
      cb.addEventListener('change', function(name, box){
        return function(){
          if (box.checked) selectedSeries.add(name); else selectedSeries.delete(name);
          renderSeriesChips(); updateSeriesCount(); render();
        };
      }(sName, cb));
      const txt=document.createElement('span'); txt.textContent=sName;
      row.append(cb,txt); seriesMsList.append(row);
    }
  }

  function renderSeriesChips(){
    seriesMsChips.innerHTML='';
    const list = Array.from(selectedSeries).sort(naturalSort);
    for (var i=0; i<list.length; i++){
      const name = list[i];
      const chip=document.createElement('span'); chip.className='chip';
      chip.innerHTML = name + ' <button>×</button>';
      chip.querySelector('button').addEventListener('click', function(nm){
        return function(){
          selectedSeries.delete(nm);
          renderSeriesList(); renderSeriesChips(); updateSeriesCount(); render();
        };
      }(name));
      seriesMsChips.append(chip);
    }
  }

  function openSeriesMs(){ seriesMsPop.hidden=false; seriesMsBtn.setAttribute('aria-expanded','true'); seriesMsSearch.focus(); }
  function closeSeriesMs(){ seriesMsPop.hidden=true; seriesMsBtn.setAttribute('aria-expanded','false'); }

  seriesMsBtn.addEventListener('click', function(){ seriesMsPop.hidden ? openSeriesMs() : closeSeriesMs(); });
  document.addEventListener('click', function(e){ if(!seriesMs.contains(e.target)) closeSeriesMs(); });
  seriesMsSearch.addEventListener('input', function(){
    const q = norm(seriesMsSearch.value);
    filteredSeries = allSeries.filter(function(s){ return norm(s).indexOf(q) !== -1; });
    renderSeriesList();
  });
  seriesMsSelectAll.addEventListener('click', function(){
    for (var i=0; i<filteredSeries.length; i++) selectedSeries.add(filteredSeries[i]);
    renderSeriesList(); renderSeriesChips(); updateSeriesCount(); render();
  });
  seriesMsClear.addEventListener('click', function(){
    selectedSeries.clear(); renderSeriesList(); renderSeriesChips(); updateSeriesCount(); render();
  });

  renderSeriesList(); updateSeriesCount();

  // Films
  const allFilms = Array.from(new Set(items.map(i=>i.film).filter(Boolean))).sort(naturalSort);
  const selectedFilms = new Set(); let filteredFilms = allFilms.slice();
  function updateFilmCount(){ filmMsCount.textContent = selectedFilms.size===0 ? 'All'
    : (Array.from(selectedFilms).sort(naturalSort).slice(0,2).join(', ') + (selectedFilms.size>2?` +${selectedFilms.size-2}`:'')); }
  function renderFilmList(){
    filmMsList.innerHTML='';
    for (const f of filteredFilms){
      const id='film_'+btoa(unescape(encodeURIComponent(f))).replace(/=/g,'');
      const row=document.createElement('label'); row.className='ms__item'; row.htmlFor=id;
      const cb=document.createElement('input'); cb.type='checkbox'; cb.id=id; cb.checked=selectedFilms.has(f);
      cb.addEventListener('change',()=>{ if(cb.checked)selectedFilms.add(f); else selectedFilms.delete(f);
        renderFilmChips(); updateFilmCount(); render(); });
      const txt=document.createElement('span'); txt.textContent=f;
      row.append(cb,txt); filmMsList.append(row);
    }
  }
  function renderFilmChips(){
    filmMsChips.innerHTML='';
    for(const f of Array.from(selectedFilms).sort(naturalSort)){
      const chip=document.createElement('span'); chip.className='chip';
      chip.innerHTML=`${f} <button>×</button>`;
      chip.querySelector('button').addEventListener('click',()=>{
        selectedFilms.delete(f); renderFilmList(); renderFilmChips(); updateFilmCount(); render();
      });
      filmMsChips.append(chip);
    }
  }
  function openFilmMs(){ filmMsPop.hidden=false; filmMsBtn.setAttribute('aria-expanded','true'); filmMsSearch.focus(); }
  function closeFilmMs(){ filmMsPop.hidden=true; filmMsBtn.setAttribute('aria-expanded','false'); }
  filmMsBtn.addEventListener('click',()=> filmMsPop.hidden ? openFilmMs() : closeFilmMs());
  document.addEventListener('click',e=>{ if(!filmMs.contains(e.target)) closeFilmMs(); });
  filmMsSearch.addEventListener('input',()=>{ const q=norm(filmMsSearch.value);
    filteredFilms = allFilms.filter(s=>norm(s).includes(q)); renderFilmList(); });
  filmMsSelectAll.addEventListener('click',()=>{ for(const f of filteredFilms) selectedFilms.add(f);
    renderFilmList(); renderFilmChips(); updateFilmCount(); render(); });
  filmMsClear.addEventListener('click',()=>{ selectedFilms.clear(); renderFilmList(); renderFilmChips(); updateFilmCount(); render(); });
  renderFilmList(); updateFilmCount();

  // Sleeves
  const allSleeves=Array.from(new Set(items.map(i=>i.negative_sleeve).filter(Boolean))).sort(naturalSort);
  const selectedSleeves=new Set(); let filteredSleeves=allSleeves.slice();
  function updateMsCount(){ msCount.textContent = selectedSleeves.size===0 ? 'All'
    : (Array.from(selectedSleeves).sort(naturalSort).slice(0,2).join(', ') + (selectedSleeves.size>2?` +${selectedSleeves.size-2}`:'')); }
  function renderMsList(){
    msList.innerHTML='';
    for(const s of filteredSleeves){
      const id='sleeve_'+btoa(unescape(encodeURIComponent(s))).replace(/=/g,'');
      const row=document.createElement('label'); row.className='ms__item'; row.htmlFor=id;
      const cb=document.createElement('input'); cb.type='checkbox'; cb.id=id; cb.checked=selectedSleeves.has(s);
      cb.addEventListener('change',()=>{ if(cb.checked)selectedSleeves.add(s); else selectedSleeves.delete(s);
        renderMsChips(); updateMsCount(); render(); });
      const txt=document.createElement('span'); txt.textContent=s;
      row.append(cb,txt); msList.append(row);
    }
  }
  function renderMsChips(){
    msChips.innerHTML='';
    for(const s of Array.from(selectedSleeves).sort(naturalSort)){
      const chip=document.createElement('span'); chip.className='chip';
      chip.innerHTML=`${s} <button>×</button>`;
      chip.querySelector('button').addEventListener('click',()=>{
        selectedSleeves.delete(s); renderMsList(); renderMsChips(); updateMsCount(); render();
      });
      msChips.append(chip);
    }
  }
  function openMs(){ msPop.hidden=false; msBtn.setAttribute('aria-expanded','true'); msSearch.focus(); }
  function closeMs(){ msPop.hidden=true; msBtn.setAttribute('aria-expanded','false'); }
  msBtn.addEventListener('click',()=> msPop.hidden ? openMs() : closeMs());
  document.addEventListener('click',e=>{ if(!ms.contains(e.target)) closeMs(); });
  msSearch.addEventListener('input',()=>{ const q=norm(msSearch.value);
    filteredSleeves=allSleeves.filter(s=>norm(s).includes(q)); renderMsList(); });
  msSelectAll.addEventListener('click',()=>{ for(const s of filteredSleeves) selectedSleeves.add(s);
    renderMsList(); renderMsChips(); updateMsCount(); render(); });
  msClear.addEventListener('click',()=>{ selectedSleeves.clear(); renderMsList(); renderMsChips(); updateMsCount(); render(); });
  renderMsList(); updateMsCount();

  // Zoom + toggles
  function applyZoom(px){ document.documentElement.style.setProperty('--card-min', px + 'px'); }
  applyZoom(zoom.value); zoom.addEventListener('input',()=> applyZoom(zoom.value));
  let showThumbs = true;
  toggleThumbsBtn.addEventListener('click',()=>{
    showThumbs=!showThumbs; toggleThumbsBtn.textContent=showThumbs?'Hide thumbnails':'Show thumbnails'; render();
  });

  // Filter + render
  function matches(i){
    if(selectedFilms.size>0 && !selectedFilms.has(i.film || '')) return false;
    if(selectedSleeves.size>0 && !selectedSleeves.has(i.negative_sleeve || '')) return false;

    // SERIES: allow either explicit series_id OR a print_id prefix match
    if (selectedSeries.size > 0) {
      const sid = String(i.series_id || '');
      const pid = String(i.print_id || '');
      let inSeries = (sid && selectedSeries.has(sid));

      // Fallback: if item has no series_id, match by prefix: e.g. "2023" ⇢ "2023-23"
      if (!inSeries && pid) {
        for (const s of selectedSeries) {
          if (pid.startsWith(String(s))) { inSeries = true; break; }
        }
      }
      if (!inSeries) return false;
    }

    if(statusSel.value==='digitized' && i.missing) return false;
    if(statusSel.value==='missing' && !i.missing) return false;
    if(statusSel.value==='only_neg' && !i.negative_scan) return false;
    return true;
  }

  // --- Smarter natural sort (handles 2022-1, 2022-10, 2022-2 correctly) ---
  function sortItems(arr){
    const key = sortKey.value;
    const collator = new Intl.Collator(undefined, {numeric:true, sensitivity:'base'});
    return arr.slice().sort((a,b)=>{
      const A = (a && a[key]) ? String(a[key]).toLowerCase() : '';
      const B = (b && b[key]) ? String(b[key]).toLowerCase() : '';
      return collator.compare(A,B);
    });
  }
  
function render(){
  cards.classList.toggle('cards--list', viewMode.value === 'list');
  const filtered = sortItems(items.filter(matches));
  setCount(filtered.length);
  cards.innerHTML = '';

  // Build lightboxItems as "only items with thumbs", in the same visual order
  lightboxItems = filtered.filter(i => i.thumb_path);

  let thumbIndex = 0; // index inside lightboxItems

  filtered.forEach((i) => {
    const card = document.createElement('div');
    card.className = 'card';
    const title = (i.print_id || i.title || 'Untitled');
    const row = document.createElement('div');
    if (viewMode.value === 'list') row.className = 'list-row';

    let thisLightboxIndex = -1;

    if (i.thumb_path) {
      const img = document.createElement('img');
      img.className = 'thumb';
      img.src = i.thumb_path + '?ts=' + Date.now();
      img.alt = title;
      img.style.display = showThumbs ? 'block' : 'none';

      // this card corresponds to lightboxItems[thumbIndex]
      thisLightboxIndex = thumbIndex;

      // open lightbox at this index (in the thumbs-only list)
      img.addEventListener('click', () => {
        openLightboxForIndex(thisLightboxIndex);
      });

      thumbIndex += 1;
      row.append(img);
    } else if (i.missing) {
      const ph = document.createElement('div');
      ph.className = 'thumb thumb--placeholder';
      ph.textContent = 'No scan yet';
      row.append(ph);
    }

    const content = document.createElement('div');
    const h = document.createElement('div');
    h.className = 'row';
    h.innerHTML = '<strong>' + title + '</strong>';
    content.append(h);

    const meta = document.createElement('div');
    meta.innerHTML = '<div class="row">'
      + (i.missing ? '<span class="pill pill--missing">Missing</span>' : '')
      + (i.negative_scan && !i.missing ? '<span class="pill">Neg scan</span>' : '')
      + (i.negative_sleeve ? '<span class="pill">Sleeve: ' + i.negative_sleeve + '</span>' : '')
      + (i.frame_number ? '<span class="pill">Frame: ' + i.frame_number + '</span>' : '')
      + (i.film ? '<span class="pill">Film: ' + i.film + '</span>' : '')
      + '</div>';
    content.append(meta);

    if (i.description) {
      const p = document.createElement('div');
      p.className = 'row muted';
      p.textContent = i.description;
      content.append(p);
    }

    if (i.file_path && !i.missing) {
      const rowEl = document.createElement('div');
      rowEl.className = 'row';

      async function action(cmd, path){
        try{
          const res = await fetch('/action', {
            method: 'POST',
            headers: {'Content-Type':'application/json'},
            body: JSON.stringify({cmd, path})
          });
          if(!res.ok){
            const t = await res.text();
            throw new Error(t || res.status);
          }
        }catch(e){
          alert(`Could not ${cmd}: ${e}`);
        }
      }

      const btnReveal = document.createElement('button');
      btnReveal.className = 'btn';
      btnReveal.type = 'button';
      btnReveal.textContent = 'Reveal in Finder';
      btnReveal.addEventListener('click', ()=> action('reveal', i.file_path));

      const btnOpen = document.createElement('button');
      btnOpen.className = 'btn';
      btnOpen.type = 'button';
      btnOpen.textContent = 'Open file';
      btnOpen.addEventListener('click', ()=> action('open', i.file_path));

      const copyBtn = document.createElement('button');
      copyBtn.className = 'btn';
      copyBtn.type = 'button';
      copyBtn.textContent = 'Copy path';
      copyBtn.addEventListener('click', async () => {
        try {
          await navigator.clipboard.writeText(i.file_path);
          copyBtn.textContent = 'Copied!';
          setTimeout(() => (copyBtn.textContent = 'Copy path'), 1000);
        } catch {}
      });

      rowEl.append(btnReveal, document.createTextNode(' '), btnOpen, document.createTextNode(' '), copyBtn);
      content.append(rowEl);
    }

    // highlight card if it’s the currently open lightbox image
    if (thisLightboxIndex !== -1 && thisLightboxIndex === lightboxCurrentIndex) {
      card.classList.add('card--active');
    }

    row.append(content);
    card.append(row);
    cards.append(card);
  });
}
  statusSel.addEventListener('change', render);
  viewMode.addEventListener('change', render);
  sortKey.addEventListener('change', render);

  // Clear all filters
  clearFiltersBtn.addEventListener('click', ()=>{
    selectedFilms.clear();
    selectedSleeves.clear();
    selectedSeries.clear();

    statusSel.value = '';
    viewMode.value = 'grid';
    sortKey.value = 'print_id';

    renderFilmList();  renderFilmChips();  updateFilmCount();
    renderMsList();    renderMsChips();    updateMsCount();
    renderSeriesList();renderSeriesChips();updateSeriesCount();

    render();
  });

  render();

  // -------------- ARCHIVE EDITOR (modal) ----------------
  const editorBackdrop = $('#editorBackdrop');
  const seriesRows = $('#seriesRows');
  const addSeriesBtn = $('#addSeries');
  const resetSeriesBtn = $('#resetSeries');
  const saveAndCloseBtn = $('#saveSeries');
  const closeNoSaveBtn = $('#discardSeries');
  const importSeriesBtn = $('#importSeriesBtn');
  const importSeriesInput = $('#importSeries');

  // working copy used while the modal is open
  let working = null;

  function deepClone(o){ return JSON.parse(JSON.stringify(o)); }

  function openModal(){ editorBackdrop.style.display='flex'; }
  function closeModal(){ editorBackdrop.style.display='none'; }

    // ESC + arrow keys: prioritize lightbox, then block closing editor with Esc
  document.addEventListener('keydown', function(e){
    if (lightboxBackdrop.style.display === 'flex') {
      if (e.key === 'Escape' || e.key === 'Esc') {
        e.preventDefault();
        closeLightbox();
        return;
      }
      if (e.key === 'ArrowLeft') {
        e.preventDefault();
        showPrev();
        return;
      }
      if (e.key === 'ArrowRight') {
        e.preventDefault();
        showNext();
        return;
      }
    }

    if (e.key === 'Escape' || e.key === 'Esc') {
      if (editorBackdrop.style.display === 'flex') {
        e.preventDefault(); // don't let Esc close the editor
      }
    }
  });

  function emptySeries(){
    return { id:'', title:'', description:'', prefix:'', format:'', first_number:1, last_number:1 };
  }

  function renderEditor(data){
    seriesRows.innerHTML='';
    (data.series || []).forEach(function(s, idx){ seriesRows.appendChild(makeSeriesRow(s, idx)); });
  }

  function makeSeriesRow(s, idx){
    const wrap = document.createElement('div');
    wrap.className = 'series-row';
    wrap.innerHTML = `
      <div class="series-grid">
        <label>Id<input value="${(s.id==null?'':s.id)}" data-k="id"></label>
        <label>Prefix<input value="${(s.prefix==null?'':s.prefix)}" data-k="prefix" placeholder="e.g. 2022 or M2024"></label>
        <label>Format<input value="${(s.format==null?'':s.format)}" data-k="format" placeholder="{prefix}-{n} or {prefix}-{n:02d}"></label>
        <label>First #<input type="number" min="1" value="${Number(s.first_number==null?1:s.first_number)}" data-k="first_number"></label>
        <label>Last #<input type="number" min="1" value="${Number(s.last_number==null?1:s.last_number)}" data-k="last_number"></label>
        <label>Title<input value="${(s.title==null?'':s.title)}" data-k="title"></label>
        <label>Description<textarea data-k="description">${(s.description==null?'':s.description)}</textarea></label>
      </div>
      <div class="row-actions">
        <button class="btn btn-danger" data-action="remove">Remove</button>
      </div>
    `;

    // remove row (on working copy only)
    wrap.querySelector('[data-action="remove"]').addEventListener('click', function(){
      const arr = working.series || [];
      arr.splice(idx, 1);
      working.series = arr;
      renderEditor(working);
    });

    // bind inputs to working copy (do NOT persist yet)
    wrap.querySelectorAll('[data-k]').forEach(function(el){
      el.addEventListener('input', function(){
        const key = el.getAttribute('data-k');
        let val = el.value;
        if (key === 'first_number' || key === 'last_number') {
          val = Number(val || 0);
        }
        working.series[idx][key] = val;
      });
    });

    return wrap;
  }

  // open editor with a fresh working copy
  $('#openEditor').addEventListener('click', function(){
    working = deepClone(registry);
    working.series = Array.isArray(working.series) ? working.series : [];
    renderEditor(working);
    openModal();
  });

  // add series (working copy only)
  addSeriesBtn.addEventListener('click', function(){
    working.series = working.series || [];
    working.series.push(emptySeries());
    renderEditor(working);
  });

  // reset working copy to file on disk (series.json from server)
  resetSeriesBtn.addEventListener('click', async function(){
    try {
      const res = await fetch('series.json?ts='+Date.now(), {cache:'no-store'});
      const fileData = res.ok ? await res.json() : {series:[]};
      working = deepClone(fileData || {series:[]});
    } catch {
      working = {series:[]};
    }
    renderEditor(working);
  });

  // import JSON into the working copy
  importSeriesBtn.addEventListener('click', function(){ importSeriesInput.click(); });
  importSeriesInput.addEventListener('change', async function(){
    const f = importSeriesInput.files[0];
    if(!f) return;
    try{
      const txt = await f.text();
      const data = JSON.parse(txt);
      if(!data || !Array.isArray(data.series)) throw new Error('Invalid JSON: missing "series" array');
      working = deepClone(data);
      renderEditor(working);
    }catch(e){ alert(e.message); }
    finally{ importSeriesInput.value=''; }
  });

  // persist overrides to localStorage
  function persistOverrides(data){
    localStorage.setItem(STORAGE_KEY, JSON.stringify(data));
  }

  // Save (persist working → overrides + download) and close
  saveAndCloseBtn.addEventListener('click', function(){
    registry = deepClone(working);          // promote working copy
    persistOverrides(registry);              // commit to localStorage

    // download a fresh series.json
    const blob = new Blob([JSON.stringify(registry, null, 2)], {type:'application/json'});
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'series.json';
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(function(){ URL.revokeObjectURL(a.href); }, 1000);

    closeModal();
  });

  // Close without saving: discard working, keep current registry/overrides as-is
  closeNoSaveBtn.addEventListener('click', function(){
    working = null;
    closeModal();
  });

})();
</script>
</body>
</html>"""

    # only replace our token; never call .format()
    html = html.replace("__BUILD_TAG__", BUILD_TAG)
    (out_dir / "index.html").write_text(html, encoding="utf-8")


def write_series_scaffold_if_absent(out_dir: Path) -> None:
    """Create a minimal series.json if it doesn't exist yet."""
    series_path = out_dir / "series.json"
    if series_path.exists():
        return
    scaffold = {
        "series": [
            {
                "id": "2022",
                "title": "2022 main series",
                "description": "Edit me: describe this series",
                "prefix": "2022",        # used if 'format' not provided
                "first_number": 1,       # inclusive
                "last_number": 12,       # inclusive (set this to 'where you are' now)
                # OR provide a format, which wins over 'prefix':
                # "format": "{prefix}-{n:02d}"   # supports {n} / {n:02d} and {prefix}
            },
            {
                "id": "M2024",
                "title": "Medium format 2024",
                "description": "Edit me too",
                "prefix": "M2024",
                "first_number": 1,
                "last_number": 8
            }
        ]
    }
    series_path.write_text(json.dumps(scaffold, ensure_ascii=False, indent=2), encoding="utf-8")

def build_static_site(rows: List[Record] | List[dict], out_dir: Path, html_mode: str = "auto") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Ensure we have a series.json scaffold for the viewer/editor
    write_series_scaffold_if_absent(out_dir)

    # Normalize rows → list[dict]
    data = rows if (rows and isinstance(rows[0], dict)) else [r.to_row() for r in rows]

    # Make sure neg scans are marked missing in catalog.json
    for r in data:
        if r.get("negative_scan"):
            r["missing"] = True

    # Always refresh catalog.json
    cat_path = out_dir / "catalog.json"
    with cat_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logging.info("Wrote catalog.json → %s", cat_path)

    # HTML policy
    index_path = out_dir / "index.html"
    if html_mode == "force":
        write_index_html(out_dir)
        logging.info("index.html: FORCE rewrite at %s", index_path)
    elif html_mode == "auto":
        if not index_path.exists():
            write_index_html(out_dir)
            logging.info("index.html: AUTO wrote (did not exist) at %s", index_path)
        else:
            logging.info("index.html: AUTO skipped (exists) at %s", index_path)
    else:
        logging.info("index.html: SKIP per --html=skip (existing file untouched)")

def is_archive_member(rec: Record, mode: str, require_keyword: List[str], require_prefix: List[str]) -> bool:
    toks = (rec.keywords or []) + (rec.hierarchical_keywords or [])
    if mode == "all":
        return True
    if mode == "auto":
        # Archive = only prints you've numbered.
        return bool(rec.print_id)
    if mode == "keyword":
        if not require_keyword:
            return False
        s = set(toks)
        return any(k in s for k in require_keyword)
    if mode == "prefix":
        if not require_prefix:
            return False
        for t in toks:
            for pref in require_prefix:
                if t.startswith(pref):
                    return True
        return False
    return False

# -------------------- main --------------------

# -------------------- main --------------------

def main():
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log.upper(), logging.INFO),
                        format="%(levelname)s: %(message)s")

    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        raise SystemExit(f"Root not found: {root}")

    targets = list(walk_targets(root))
    total_targets = len(targets)
    logging.info("Found %d candidates (.xmp or images): %d", total_targets, total_targets)

    rows: List[Record] = []

    jobs = args.jobs or 0
    if total_targets == 0:
        logging.info("No targets to process.")
    elif jobs == 1:
        import time
        last_print = 0.0
        for idx, path in enumerate(targets, 1):
            try:
                rec = parse_xmp(path)
            except Exception as e:
                logging.exception("parse_xmp failed for %s: %s", path, e)
                rec = Record(file_path="", xmp_path=str(path))
            if not rec.file_path:
                logging.debug("No associated image found for %s", path)
            rows.append(rec)
            now = time.time()
            if (idx % 200 == 0) or (idx == total_targets) or (now - last_print) >= 0.5:
                print(f"\rParsing metadata… {idx}/{total_targets} ({(idx/total_targets)*100:.1f}%)",
                      end="", flush=True)
                last_print = now
        print()
    else:
        if jobs <= 0:
            jobs = max(2, os.cpu_count() or 4)
        logging.info("Parsing with %d threads …", jobs)
        ordered: List[Optional[Record]] = [None] * total_targets

        def _work(i: int, p: Path):
            try:
                r = parse_xmp(p)
            except Exception as e:
                logging.exception("parse_xmp failed for %s: %s", p, e)
                r = Record(file_path="", xmp_path=str(p))
            return i, r

        import time
        last_print = 0.0
        done = 0
        with ThreadPoolExecutor(max_workers=jobs) as ex:
            futs = [ex.submit(_work, i, p) for i, p in enumerate(targets, 1)]
            for fut in as_completed(futs):
                i, rec = fut.result()
                ordered[i - 1] = rec
                done += 1
                now = time.time()
                if (done % 200 == 0) or (done == total_targets) or (now - last_print) >= 0.5:
                    print(f"\rParsing metadata… {done}/{total_targets} ({(done/total_targets)*100:.1f}%)",
                          end="", flush=True)
                    last_print = now
        print()
        rows = [r for r in ordered if r is not None]

    # Selection
    selected: List[Record] = []
    for r in rows:
        if is_archive_member(r, args.select, args.require_keyword, args.require_prefix):
            selected.append(r)
    logging.info("Selected %d / %d records (mode=%s)", len(selected), len(rows), args.select)

    # Thumbnails (optional) — skip existing, keep crop normalized
    if args.thumbs:
        import time
        thumbs_dir = Path(args.thumbs).expanduser().resolve()
        thumbs_dir.mkdir(parents=True, exist_ok=True)

        if args.thumbs_scope == "strict":
            thumb_candidates = [r for r in selected if (r.print_id or r.negative_sleeve)]
        else:
            thumb_candidates = list(selected)

            todo = []
            already = 0
            for r in thumb_candidates:
                if not r.file_path:
                    continue
                p = Path(r.file_path)
                if not p.exists():
                    continue

                out_thumb = thumbs_dir / (p.stem + ".jpg")
                if out_thumb.exists():
                    r.thumb_path = f"thumbs/{out_thumb.name}"
                    already += 1
                    continue

                # queue work item – no crop / angle / perspective anymore
                todo.append((r, p, out_thumb))

            total = len(todo)
            logging.info(
                "Generating thumbnails (scope=%s): %d to build, %d already present",
                args.thumbs_scope, total, already
            )

            done = 0
            last_print = 0.0
            for r, p, out_thumb in todo:
                t = make_thumbnail(p, thumbs_dir, args.thumb_size)
                done += 1
                if t:
                    r.thumb_path = f"thumbs/{out_thumb.name}"
                now = time.time()
                if (now - last_print) >= 0.5 or done == total:
                    pct = (done / total * 100) if total else 100.0
                    print(f"\r  → {done}/{total} ({pct:.1f}%)", end="", flush=True)
                    last_print = now
        if total:
            print()
        logging.info("Thumbnails complete.")

    # Outputs
    out_csv = Path(args.csv).expanduser().resolve()
    write_csv(selected, out_csv)
    logging.info("Wrote CSV → %s (%d rows)", out_csv, len(selected))

    if args.sqlite:
        out_sqlite = Path(args.sqlite).expanduser().resolve()
        write_sqlite(selected, out_sqlite)
        logging.info("Wrote SQLite → %s", out_sqlite)

    if args.json:
        out_json = Path(args.json).expanduser().resolve()
        write_json(selected, out_json)
        logging.info("Wrote JSON → %s", out_json)

    if args.site:
        out_site = Path(args.site).expanduser().resolve()
        build_static_site(selected, out_site, html_mode=args.html)
        logging.info("Built static site → %s (html=%s)", out_site, args.html)

if __name__ == "__main__":
    main()