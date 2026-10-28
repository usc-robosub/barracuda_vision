#!/usr/bin/env python3
"""auto_annotate_gates.py

Download the RoboSub gate dataset from Roboflow and run SAM-based
auto-annotation to generate segmentation masks in YOLO-seg format.

Usage
-----
1. Copy .env.example to .env and fill in your Roboflow API key:
       cp .env.example .env

2. Run (inside the robosub-seg conda env):
       conda activate robosub-seg
       python auto_annotate_gates.py

Output
------
  data/
    images/   — original images
    labels/   — YOLO-seg .txt files (one per image)
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("ROBOFLOW_API_KEY")
if not API_KEY:
    print("ERROR: ROBOFLOW_API_KEY not set.")
    print("  1. Copy .env.example to .env")
    print("  2. Fill in your Roboflow API key")
    sys.exit(1)

WORKSPACE   = "uscauv"
PROJECT     = "combined-dataset-v2-0cngh"
VERSION     = 2
DATA_FORMAT = "yolov8"
OUTPUT_DIR  = Path("data")
SAM_MODEL   = "sam2_b.pt"

print(f"\n[1/2] Downloading {WORKSPACE}/{PROJECT} v{VERSION} ...")

from roboflow import Roboflow
rf = Roboflow(api_key=API_KEY)
project = rf.workspace(WORKSPACE).project(PROJECT)
dataset = project.version(VERSION).download(DATA_FORMAT, location=str(OUTPUT_DIR))

image_dir = OUTPUT_DIR / "train" / "images"
label_dir = OUTPUT_DIR / "train" / "labels"

print(f"      Images : {image_dir}")
print(f"      Labels : {label_dir}")

print(f"\n[2/2] Running SAM auto-annotation (model: {SAM_MODEL}) ...")

from ultralytics.data.annotator import auto_annotate

auto_annotate(
    data=str(image_dir),
    det_model="yolov8x.pt",
    sam_model=SAM_MODEL,
    device="cuda:0",
    output_dir=str(label_dir),
)

print("\nDone! Segmentation masks saved to:", label_dir)
print("\nNext steps:")
print("  1. Review masks in Roboflow (upload the labels/ folder)")
print("  2. Fix any bad masks manually in Roboflow")
print("  3. Export the corrected dataset for YOLOv8-seg training")
