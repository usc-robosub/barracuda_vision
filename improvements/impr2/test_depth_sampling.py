#!/usr/bin/env python3
"""Unit tests for Improvement 2 depth sampling functions — no ROS needed."""

import sys
import numpy as np


# ---------------------------------------------------------------------------
# Copy of the three functions (no ROS dependency)
# ---------------------------------------------------------------------------

def sample_depth_bbox(depth_image, xmin, ymin, xmax, ymax, min_valid=5):
    h, w = depth_image.shape[:2]
    x0 = max(0, int(xmin)); y0 = max(0, int(ymin))
    x1 = min(w, int(xmax)); y1 = min(h, int(ymax))
    if x1 <= x0 or y1 <= y0:
        return None
    roi = depth_image[y0:y1, x0:x1]
    valid = roi[np.isfinite(roi) & (roi > 0.0)]
    if len(valid) < min_valid:
        return None
    return float(np.median(valid))


def sample_depth_border(depth_image, xmin, ymin, xmax, ymax, border=8, min_valid=5):
    h, w = depth_image.shape[:2]
    x0 = max(0, int(xmin)); y0 = max(0, int(ymin))
    x1 = min(w, int(xmax)); y1 = min(h, int(ymax))
    if x1 <= x0 or y1 <= y0:
        return None
    roi = depth_image[y0:y1, x0:x1]
    rh, rw = roi.shape[:2]
    b = min(border, rh // 2, rw // 2)
    if b < 1:
        return sample_depth_bbox(depth_image, xmin, ymin, xmax, ymax, min_valid)
    mask = np.zeros((rh, rw), dtype=bool)
    mask[:b, :] = True; mask[-b:, :] = True
    mask[:, :b] = True; mask[:, -b:] = True
    border_px = roi[mask]
    valid = border_px[np.isfinite(border_px) & (border_px > 0.0)]
    if len(valid) < min_valid:
        return None
    return float(np.median(valid))


def sample_depth_pipe(depth_image, xmin, ymin, xmax, ymax, min_valid=5):
    h, w = depth_image.shape[:2]
    x0 = max(0, int(xmin)); x1 = min(w, int(xmax))
    cy = int((ymin + ymax) / 2)
    strip_h = max(1, int((ymax - ymin) * 0.2))
    y0 = max(0, cy - strip_h // 2)
    y1 = min(h, cy + strip_h // 2 + 1)
    if x1 <= x0 or y1 <= y0:
        return None
    roi = depth_image[y0:y1, x0:x1]
    valid = roi[np.isfinite(roi) & (roi > 0.0)]
    if len(valid) < min_valid:
        return None
    return float(np.median(valid))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

PASS = '\033[92mPASS\033[0m'
FAIL = '\033[91mFAIL\033[0m'
results = []

def check(name, got, expected, tol=0.01):
    ok = got is not None and abs(got - expected) <= tol
    print(f'  [{PASS if ok else FAIL}] {name}: got={got}, expected={expected}')
    results.append(ok)

def check_none(name, got):
    ok = got is None
    print(f'  [{PASS if ok else FAIL}] {name}: got={got}, expected=None')
    results.append(ok)

img = np.full((100, 100), 3.0, dtype=np.float32)
img[18:82, 18:82] = 8.0  # hollow interior

print('\n=== sample_depth_bbox ===')
check('hollow gate full bbox ~ 8.0m', sample_depth_bbox(img, 10, 10, 90, 90), 8.0, tol=0.5)
check('solid region ~ 3.0m', sample_depth_bbox(img, 0, 0, 10, 10), 3.0)
check_none('all NaN → None', sample_depth_bbox(np.full((100,100), np.nan, dtype=np.float32), 0, 0, 100, 100))

print('\n=== sample_depth_border (gate) ===')
check('hollow gate border ~ 3.0m', sample_depth_border(img, 10, 10, 90, 90), 3.0)
check('tiny bbox fallback ~ 3.0m', sample_depth_border(img, 0, 0, 5, 5), 3.0)

print('\n=== sample_depth_pipe ===')
pipe_img = np.full((100, 100), 8.0, dtype=np.float32)
pipe_img[45:55, 10:90] = 3.0
check('pipe centre strip ~ 3.0m', sample_depth_pipe(pipe_img, 10, 30, 90, 70), 3.0)
check('uniform background ~ 5.0m', sample_depth_pipe(np.full((100,100), 5.0, dtype=np.float32), 10, 10, 90, 90), 5.0)

print(f'\nResults: {sum(results)}/{len(results)} passed')
if not all(results):
    sys.exit(1)
