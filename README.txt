XMP Archive Harvester — v2 (ready to run)
========================================

Files:
- xmp_archive_harvester.py

Quick start (macOS)
-------------------
1) Open Terminal and cd to this folder, e.g.:
   cd "/Users/alvarotrabanco/documents_mis_proyectos_small/Mis_Proyectos/Fotografia/AnalogProcess/Archive"

2) Run with your photo root:
   python3 xmp_archive_harvester.py      --root "/Users/alvarotrabanco/Photography/les_mios_semeyes"      --csv out/archive.csv      --json out/catalog.json      --site out/site      --select auto

Only include items explicitly marked for the archive (if you use Archive:true):
   python3 xmp_archive_harvester.py      --root "/Users/alvarotrabanco/Photography/les_mios_semeyes"      --csv out/archive.csv      --json out/catalog.json      --site out/site      --select keyword      --require-keyword "Archive:true"

Outputs:
- out/archive.csv
- out/catalog.json
- out/site/index.html   (open in your browser)

Notes:
- Ensure Lightroom Classic: Catalog Settings → Metadata → "Automatically write changes into XMP" is enabled.
- The harvester reads both flat keywords (PrintID:2023-23) and hierarchical (PrintID|2023-23).
- Embedded XMP (JPG/TIFF/DNG) can be added later if needed.
- Logging level: add --log DEBUG to see more details.
Generated on: 2025-10-12T12:03:23.494931
