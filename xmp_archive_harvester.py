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

# -------------------- config & helpers --------------------

NS = {
    "x": "adobe:ns:meta/",
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "dc": "http://purl.org/dc/elements/1.1/",
    "lr": "http://ns.adobe.com/lightroom/1.0/",
    "xmp": "http://ns.adobe.com/xap/1.0/",
    "crs": "http://ns.adobe.com/camera-raw-settings/1.0/",
}

IMAGE_EXTS = [
    ".jpg", ".jpeg", ".tif", ".tiff", ".dng", ".nef", ".cr2", ".cr3", ".arw", ".orf", ".raf"
]

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

    def to_row(self) -> dict:
        return {
            "file_path": self.file_path,
            "xmp_path": self.xmp_path,
            "title": self.title or "",
            "description": self.description or "",
            # legacy string for compatibility
            "keywords": "; ".join(self.keywords) if self.keywords else "",
            # explicit forms for searching (frontend can use either)
            "keywords_text": "; ".join(self.keywords) if self.keywords else "",
            "keywords_list": list(self.keywords) if self.keywords else [],
            "hierarchical_keywords": "; ".join(self.hierarchical_keywords) if self.hierarchical_keywords else "",
            "print_id": self.print_id or "",
            "negative_sleeve": self.negative_sleeve or "",
            "frame_number": self.frame_number or "",
            "film": self.film or "",
            "thumb_path": self.thumb_path or "",
        }

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Harvest Lightroom .XMP metadata into CSV/JSON and optional static site.")
    p.add_argument("--root", required=True, help="Root directory to scan recursively")
    p.add_argument("--csv", required=True, help="Output CSV file path")
    p.add_argument("--sqlite", default=None, help="Optional SQLite output path")
    p.add_argument("--json", default=None, help="Optional JSON catalog output path")
    p.add_argument("--site", default=None, help="Optional output directory for a static viewer site")
    p.add_argument("--select", choices=["all", "auto", "keyword", "prefix"], default="auto",
                   help="Subset selection strategy for archive membership")
    p.add_argument("--require-keyword", action="append", default=[], help="Keyword token that must be present (repeatable)")
    p.add_argument("--require-prefix", action="append", default=[], help="Keyword prefix; at least one token starting with this must be present (repeatable)")
    p.add_argument("--log", default="INFO", help="Logging level (DEBUG, INFO, WARNING)")
    p.add_argument("--thumbs", default=None, help="Optional directory to write thumbnails")
    p.add_argument("--thumb-size", type=int, default=512, help="Max thumbnail edge in px (default 512)")
    p.add_argument("--jobs", type=int, default=0, help="Worker threads for parsing (0=auto: CPU count; 1=sequential)")
    p.add_argument("--thumbs-scope", choices=["selected", "strict"], default="selected",
                   help="Which items get thumbnails: 'selected' (default) or 'strict' (print_id or negative_sleeve)")
    return p.parse_args()


def make_thumbnail(image_path: Path, thumbs_dir: Path, max_px: int,
                   crop_box: Optional[tuple] = None,
                   crop_angle: Optional[float] = None,
                   exif_normalize: bool = True) -> Optional[Path]:
    """
    Lightroom-like pipeline:
      1) Normalize EXIF orientation (transpose) if exif_normalize=True.
      2) Rotate by -CropAngle (LR is clockwise-positive; PIL is CCW-positive),
         with expand=False (keep canvas size).
      3) Crop using normalized (L,T,R,B) in [0..1] on that rotated canvas.
      4) Resize and save.
    Falls back to sips fast path only if NO crop and NO angle.
    """
    try:
        thumbs_dir.mkdir(parents=True, exist_ok=True)
        out = thumbs_dir / (image_path.stem + ".jpg")

        # Fast path ONLY when there's no crop and no angle
        use_fast_sips = (crop_box is None and (crop_angle is None or abs(crop_angle) < 1e-6))
        if use_fast_sips and shutil.which("sips"):
            cmd = [
                "sips", "-Z", str(max_px),
                image_path.as_posix(),
                "--setProperty", "format", "jpeg",
                "--out", out.as_posix()
            ]
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if res.returncode == 0 and out.exists():
                return out
            # otherwise fall through to PIL

        from PIL import Image, ImageOps
        with Image.open(image_path.as_posix()) as im:
            # 1) Normalize orientation ONLY if requested
            if exif_normalize:
                im = ImageOps.exif_transpose(im)

            # 2) Apply LR crop angle (LR is clockwise-positive → negate for PIL)
            if crop_angle is not None and abs(crop_angle) > 1e-6:
                im = im.rotate(-float(crop_angle), resample=Image.BICUBIC, expand=False)

            # 3) Crop with normalized box in this canvas
            if crop_box is not None:
                nL, nT, nR, nB = crop_box
                nL = max(0.0, min(1.0, float(nL)))
                nT = max(0.0, min(1.0, float(nT)))
                nR = max(0.0, min(1.0, float(nR)))
                nB = max(0.0, min(1.0, float(nB)))
                W, H = im.size
                L = int(round(nL * W))
                T = int(round(nT * H))
                R = int(round(nR * W))
                B = int(round(nB * H))
                # guard
                L = max(0, min(L, W)); R = max(0, min(R, W))
                T = max(0, min(T, H)); B = max(0, min(B, H))
                if R > L and B > T:
                    im = im.crop((L, T, R, B))

            # 4) Resize
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
    KEY_ALIASES = {
        "printid": "print_id", "print_id": "print_id", "print-id": "print_id",
        "negativesleeve": "negative_sleeve", "negative_sleeve": "negative_sleeve",
        "sleeve": "negative_sleeve", "sleeve#": "negative_sleeve", "sleeve_no": "negative_sleeve",
        "frame": "frame_number", "frame_number": "frame_number", "frame-no": "frame_number", "frame#": "frame_number",
        "film": "film", "film_type": "film", "filmtype": "film",
    }
    def _assign(k: str, v: str):
        kk = KEY_ALIASES.get(k.lower())
        if not kk or not v:
            return
        if kk == "print_id" and not rec.print_id:
            rec.print_id = v
        elif kk == "negative_sleeve" and not rec.negative_sleeve:
            rec.negative_sleeve = v
        elif kk == "frame_number" and not rec.frame_number:
            rec.frame_number = v
        elif kk == "film" and not rec.film:
            rec.film = v

    for token in (rec.keywords or []):
        t = token.strip()
        if ":" in t:
            k, v = t.split(":", 1)
            _assign(k.strip(), v.strip())

    for token in (rec.hierarchical_keywords or []):
        parts = [p.strip() for p in token.split("|") if p.strip()]
        for i, part in enumerate(parts):
            low = part.lower()
            if low in {"printid","print_id","print-id","negativesleeve","negative_sleeve","sleeve","sleeve#","sleeve_no","frame","frame_number","frame-no","frame#","film","film_type","filmtype"} and i + 1 < len(parts):
                _assign(low, parts[i + 1]); break

def parse_xmp_from_root(xmp_root: ET.Element, image_path_for_record: Optional[Path], xmp_path_for_record: Path) -> Record:
    desc = xmp_root.find(".//rdf:RDF/rdf:Description", NS)
    if desc is None:
        return Record(file_path=str(image_path_for_record or ""), xmp_path=str(xmp_path_for_record))

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

    keywords = extract_dc_bag_items(desc, "dc:subject")
    h_keywords = extract_dc_bag_items(desc, "lr:hierarchicalSubject")
    lr_crop = extract_lr_crop(desc)

    rec = Record(
        file_path=str(image_path_for_record or ""),
        xmp_path=str(xmp_path_for_record),
        title=title,
        description=description,
        keywords=keywords,
        hierarchical_keywords=h_keywords,
    )
    if lr_crop:
        rec.crop_left   = lr_crop.get("left")
        rec.crop_top    = lr_crop.get("top")
        rec.crop_right  = lr_crop.get("right")
        rec.crop_bottom = lr_crop.get("bottom")
        rec.crop_angle  = lr_crop.get("angle")
    _assign_keywords(rec)
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
        rec.exif_normalize_for_crop = False  # sidecar: do NOT EXIF-normalize for crop
        return rec

    root = extract_embedded_xmp(xmp_or_image_file)
    if root is None:
        return Record(file_path=str(xmp_or_image_file), xmp_path=str(xmp_or_image_file))
    rec = parse_xmp_from_root(root, xmp_or_image_file, xmp_or_image_file)
    rec.exif_normalize_for_crop = True  # embedded: DO EXIF-normalize for crop
    return rec

def walk_targets(root: Path) -> Iterable[Path]:
    """
    Yield .xmp sidecars; then yield images that do NOT have a sidecar
    in the same directory (prevents skipping JPGs in sibling folders).
    """
    sidecar_keys = set()  # (dirpath, stem)
    for p in root.rglob("*.xmp"):
        if p.is_file():
            sidecar_keys.add((str(p.parent.resolve()), p.stem))
            yield p
    exts = tuple(e.lower() for e in IMAGE_EXTS)
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            key = (str(p.parent.resolve()), p.stem)
            if key not in sidecar_keys:
                yield p

def write_csv(rows: List[Record], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].to_row().keys()) if rows else [
        "file_path","xmp_path","title","description","keywords",
        "hierarchical_keywords","print_id","negative_sleeve","frame_number","film","thumb_path"
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
            thumb_path TEXT
        )
        """
    )
    cur.execute("DELETE FROM photos")
    cur.executemany(
        """
        INSERT INTO photos (
            file_path, xmp_path, title, description, keywords, hierarchical_keywords,
            print_id, negative_sleeve, frame_number, film, thumb_path
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                r.file_path, r.xmp_path, r.title, r.description,
                "; ".join(r.keywords), "; ".join(r.hierarchical_keywords),
                r.print_id, r.negative_sleeve, r.frame_number, r.film, r.thumb_path
            )
            for r in rows
        ],
    )
    conn.commit(); conn.close()

def write_json(rows: List[Record], out_json: Path) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump([r.to_row() for r in rows], f, ensure_ascii=False, indent=2)

# -------------------- STATIC SITE --------------------

def write_index_html(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    html = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Photo Archive</title>
<style>
  :root { --gap:16px; --border:#ddd; --muted:#666; --bg:#fff; --bg2:#fafafa; --shadow:0 8px 24px rgba(0,0,0,.12); }
  *{box-sizing:border-box}
  body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Ubuntu;margin:24px;background:var(--bg2)}
  header.toolbar{display:flex;gap:12px;align-items:center;margin-bottom:16px;flex-wrap:wrap}
  input,select,button{padding:6px 8px}
  .cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:var(--gap)}
  .cards.cards--list{display:flex;flex-direction:column}
  .card{border:1px solid var(--border);border-radius:12px;padding:12px;background:var(--bg)}
  .muted{color:var(--muted);font-size:12px}
  .row{margin:6px 0}
  a{word-break:break-all}
  .pill{display:inline-block;border:1px solid #ccc;border-radius:999px;padding:2px 8px;margin-right:6px;font-size:12px}
  .list-row{display:flex;gap:12px;align-items:flex-start}
  .thumb{max-width:100%;border-radius:8px;margin-bottom:8px}
  .list-row .thumb{max-width:160px}
  .group{display:flex;gap:8px;align-items:center}
  .stack{display:flex;gap:8px;align-items:flex-start}
  .stack > *{display:block}
  /* sleeves multi-select */
  .ms { position:relative; }
  .ms__button { display:flex;align-items:center;gap:8px; border:1px solid var(--border); background:var(--bg); border-radius:10px; padding:6px 10px; cursor:pointer; }
  .ms__button-badge { background:#eee;border-radius:999px;padding:2px 8px;font-size:12px }
  .ms__popover { position:absolute; top:calc(100% + 8px); left:0; z-index:20; width:280px; background:var(--bg); border:1px solid var(--border); border-radius:12px; box-shadow:var(--shadow); padding:10px; }
  .ms__popover[hidden]{ display:none; }
  .ms__search { width:100%; border:1px solid var(--border); border-radius:8px; padding:6px 8px; }
  .ms__list { margin-top:8px; max-height:220px; overflow:auto; border:1px solid var(--border); border-radius:10px; padding:6px; }
  .ms__item { display:flex; align-items:center; gap:8px; padding:4px 2px; }
  .ms__actions { display:flex; gap:8px; margin-top:8px; }
  .ms__chips { display:flex; flex-wrap:wrap; gap:6px; margin-top:8px; }
  .chip { display:flex; align-items:center; gap:6px; background:#f0f0f0; border-radius:999px; padding:2px 8px; font-size:12px; }
  .chip button { border:none; background:transparent; cursor:pointer; padding:0 2px; font-size:14px; line-height:1; }
</style>
</head>
<body>
<header class="toolbar">
  <h1 style="margin:0">Photo Archive</h1>
  <div class="group">
    <label for="film">Film</label>
    <select id="film"><option value="">All films</option></select>
  </div>
  <div class="ms" id="sleevesMs">
    <button type="button" class="ms__button" id="msBtn" aria-haspopup="listbox" aria-expanded="false">
      <span>Pick sleeves</span>
      <span class="ms__button-badge" id="msCount">All</span>
    </button>
    <div class="ms__popover" id="msPop" hidden>
      <input id="msSearch" class="ms__search" placeholder="Search sleeves… (e.g. 57 or Sleeve: 57)"/>
      <div class="ms__actions">
        <button id="msSelectAll" type="button">Select all (filtered)</button>
        <button id="msClear" type="button">Clear</button>
      </div>
      <div id="msList" class="ms__list" role="listbox" aria-multiselectable="true"></div>
      <div class="ms__chips" id="msChips"></div>
    </div>
  </div>
  <div class="group">
    <label for="hasprint">Prints</label>
    <select id="hasprint">
      <option value="">All</option>
      <option value="yes">Has PrintID</option>
      <option value="no">No PrintID</option>
    </select>
  </div>
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
  <button id="toggleThumbs">Hide thumbnails</button>
</header>

<div id="cards" class="cards"></div>

<script>
(async function(){
  // Load catalog safely (clear error if missing)
  async function loadCatalog(){
    const url = 'catalog.json?ts=' + Date.now();
    const res = await fetch(url, {cache:'no-store'});
    if (!res.ok) throw new Error('GET '+url+' → '+res.status+' '+res.statusText);
    const txt = await res.text();
    if (!txt.trim()) throw new Error('catalog.json is empty');
    try { return JSON.parse(txt); } catch(e){ throw new Error('Invalid JSON: '+e.message); }
  }

  let items = [];
  try { items = await loadCatalog(); }
  catch(e){
    const warn = document.createElement('div');
    warn.style.cssText='background:#fee;border:1px solid #f99;color:#900;padding:8px 12px;border-radius:8px;margin-bottom:12px';
    warn.textContent='⚠️ '+e.message;
    document.body.insertBefore(warn, document.body.firstChild);
    items = [];
  }

  const $ = s => document.querySelector(s);
  const cards = $('#cards');
  const filmSel = $('#film');
  const hasPrint = $('#hasprint');
  const viewMode = $('#viewMode');
  const sortKey = $('#sortKey');
  const toggleThumbsBtn = $('#toggleThumbs');

  // Multi-select refs
  const ms = $('#sleevesMs');
  const msBtn = $('#msBtn');
  const msPop = $('#msPop');
  const msSearch = $('#msSearch');
  const msList = $('#msList');
  const msCount = $('#msCount');
  const msChips = $('#msChips');
  const msSelectAll = $('#msSelectAll');
  const msClear = $('#msClear');

  const norm = v => (v ?? '').toString().toLowerCase();
  const naturalSort = (a,b)=> String(a).localeCompare(String(b), undefined, {numeric:true, sensitivity:'base'});

  // Populate films
  const films = Array.from(new Set(items.map(i=>i.film).filter(Boolean))).sort();
  for (const f of films) {
    const o = document.createElement('option');
    o.value = f; o.textContent = f;
    filmSel.appendChild(o);
  }

  // Prepare sleeves list
  const allSleeves = Array.from(new Set(items.map(i=>i.negative_sleeve).filter(Boolean))).sort(naturalSort);

  // ----- Multi-select state -----
  const selectedSleeves = new Set();   // empty => All
  let filteredSleeves = allSleeves.slice();

  function updateMsCount(){
    if (selectedSleeves.size === 0) { msCount.textContent = 'All'; return; }
    const first = Array.from(selectedSleeves).sort(naturalSort).slice(0,2).join(', ');
    msCount.textContent = selectedSleeves.size > 2 ? `${first} +${selectedSleeves.size-2}` : first || 'All';
  }

  function renderMsList(){
    msList.innerHTML = '';
    for (const s of filteredSleeves){
      const id = 'sleeve_' + btoa(unescape(encodeURIComponent(s))).replace(/=/g,'');
      const row = document.createElement('label');
      row.className = 'ms__item';
      row.setAttribute('role','option');
      row.setAttribute('aria-selected', selectedSleeves.has(s) ? 'true' : 'false');
      row.htmlFor = id;

      const cb = document.createElement('input');
      cb.type = 'checkbox'; cb.id = id; cb.checked = selectedSleeves.has(s);
      cb.addEventListener('change', ()=>{
        if (cb.checked) selectedSleeves.add(s); else selectedSleeves.delete(s);
        row.setAttribute('aria-selected', cb.checked ? 'true' : 'false');
        renderMsChips(); updateMsCount(); render();
      });

      const txt = document.createElement('span');
      txt.textContent = s;

      row.appendChild(cb); row.appendChild(txt);
      msList.appendChild(row);
    }
  }

  function renderMsChips(){
    msChips.innerHTML = '';
    if (selectedSleeves.size === 0) return;
    for (const s of Array.from(selectedSleeves).sort(naturalSort)){
      const chip = document.createElement('span');
      chip.className = 'chip';
      chip.innerHTML = `${s} <button title="Remove">×</button>`;
      chip.querySelector('button').addEventListener('click', ()=>{
        selectedSleeves.delete(s);
        for (const lab of msList.children){
          const span = lab.querySelector('span'); const cb = lab.querySelector('input[type="checkbox"]');
          if (span && span.textContent === s && cb){ cb.checked = false; lab.setAttribute('aria-selected','false'); }
        }
        updateMsCount(); renderMsChips(); render();
      });
      msChips.appendChild(chip);
    }
  }

  function openMs(){ msPop.hidden = false; msBtn.setAttribute('aria-expanded','true'); msSearch.focus(); }
  function closeMs(){ msPop.hidden = true; msBtn.setAttribute('aria-expanded','false'); }

  msBtn.addEventListener('click', ()=>{ if (msPop.hidden) openMs(); else closeMs(); });
  document.addEventListener('click', (e)=>{ if (!ms.contains(e.target)) closeMs(); });
  msSearch.addEventListener('input', ()=>{
    const q = norm(msSearch.value).replace(/^sleeve[:\\s]*/,'');
    filteredSleeves = allSleeves.filter(s => norm(s).includes(q));
    renderMsList();
  });
  msSelectAll.addEventListener('click', ()=>{
    for (const s of filteredSleeves) selectedSleeves.add(s);
    renderMsList(); renderMsChips(); updateMsCount(); render();
  });
  msClear.addEventListener('click', ()=>{
    selectedSleeves.clear();
    renderMsList(); renderMsChips(); updateMsCount(); render();
  });

  renderMsList(); updateMsCount();

  // ----- Archive rendering -----
  let showThumbs = true;

  function matches(i){
    if (filmSel.value && i.film !== filmSel.value) return false;
    if (hasPrint.value === 'yes' && !i.print_id) return false;
    if (hasPrint.value === 'no'  &&  i.print_id) return false;
    if (selectedSleeves.size > 0){
      const val = i.negative_sleeve || '';
      if (!selectedSleeves.has(val)) return false;
    }
    return true;
  }

  function sortItems(arr){
    const key = sortKey.value;
    return arr.slice().sort((a,b)=>{
      const A = (a?.[key] ?? '').toString().toLowerCase();
      const B = (b?.[key] ?? '').toString().toLowerCase();
      return A < B ? -1 : A > B ? 1 : 0;
    });
  }

  function render(){
    cards.classList.toggle('cards--list', viewMode.value === 'list');

    const filtered = items.filter(matches);
    const sorted = sortItems(filtered);

    cards.innerHTML = '';
    for (const i of sorted){
      const card = document.createElement('div');
      card.className = 'card';
      const title = (i.print_id || i.title || 'Untitled');

      const row = document.createElement('div');
      if (viewMode.value === 'list') row.className = 'list-row';

      if (i.thumb_path){
        const img = document.createElement('img');
        img.className = 'thumb';
        img.src = i.thumb_path + '?ts=' + Date.now();
        img.alt = title;
        img.style.display = showThumbs ? 'block' : 'none';
        row.appendChild(img);
      }

      const content = document.createElement('div');

      const h = document.createElement('div');
      h.className = 'row';
      h.innerHTML = `<strong>${title}</strong>`;
      content.appendChild(h);

      const meta = document.createElement('div');
      meta.innerHTML =
        `<div class="row">`+
        (i.print_id?`<span class="pill">PrintID: ${i.print_id}</span>`:'')+
        (i.negative_sleeve?`<span class="pill">Sleeve: ${i.negative_sleeve}</span>`:'')+
        (i.frame_number?`<span class="pill">Frame: ${i.frame_number}</span>`:'')+
        (i.film?`<span class="pill">Film: ${i.film}</span>`:'')+
        `</div>`;
      content.appendChild(meta);

      if (i.title || i.description){
        const p = document.createElement('div');
        p.className = 'row muted';
        p.textContent = [i.title, i.description].filter(Boolean).join(' — ');
        content.appendChild(p);
      }

      if (i.file_path){
        const link = document.createElement('a');
        link.href = i.file_path;
        link.textContent = 'Local file path';
        link.target = '_blank';
        content.appendChild(link);
      }

      row.appendChild(content);
      card.appendChild(row);
      cards.appendChild(card);
    }
  }

  filmSel.addEventListener('change', render);
  hasPrint.addEventListener('change', render);
  viewMode.addEventListener('change', render);
  sortKey.addEventListener('change', render);
  toggleThumbsBtn.addEventListener('click', ()=>{
    showThumbs = !showThumbs;
    toggleThumbsBtn.textContent = showThumbs ? 'Hide thumbnails' : 'Show thumbnails';
    render();
  });

  render();
})();
</script>
</body>
</html>"""
    (out_dir / "index.html").write_text(html, encoding="utf-8")

def build_static_site(rows: List[Record] | List[dict], out_dir: Path) -> None:
    """
    Full build: write catalog.json from Records and index.html.
    Quick rebuild: if this is already a list of dicts, just rewrite catalog.json as-is.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    # Normalize rows → list[dict]
    if rows and isinstance(rows[0], dict):
        data = rows
    else:
        data = [r.to_row() for r in rows]  # Records → dicts
    # Write catalog and HTML
    with (out_dir / "catalog.json").open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    write_index_html(out_dir)

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

            # Keep crop box normalized (0..1) — let make_thumbnail do rotation + cropping
            crop_box = None
            if (r.crop_left is not None and r.crop_top is not None and
                r.crop_right is not None and r.crop_bottom is not None):
                nl = max(0.0, min(1.0, float(r.crop_left)))
                nt = max(0.0, min(1.0, float(r.crop_top)))
                nr = max(0.0, min(1.0, float(r.crop_right)))
                nb = max(0.0, min(1.0, float(r.crop_bottom)))
                if nr > nl and nb > nt:
                    crop_box = (nl, nt, nr, nb)

            todo.append((r, p, out_thumb, crop_box, r.crop_angle))

        total = len(todo)
        logging.info(
            "Generating thumbnails (scope=%s): %d to build, %d already present",
            args.thumbs_scope, total, already
        )

        done = 0
        last_print = 0.0
        for r, p, out_thumb, crop_box, crop_angle in todo:
            t = make_thumbnail(
                p, thumbs_dir, args.thumb_size,
                crop_box=crop_box,
                crop_angle=crop_angle,
                exif_normalize=r.exif_normalize_for_crop
            )
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
        build_static_site(selected, out_site)
        logging.info("Built static site → %s", out_site)

if __name__ == "__main__":
    main()