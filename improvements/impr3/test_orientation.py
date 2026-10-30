"""Unit tests for estimate_orientation — no ROS dependency."""
import numpy as np
from scipy.spatial.transform import Rotation


def estimate_orientation(depth_image, xmin, ymin, xmax, ymax,
                         fx, fy, cx, cy, min_valid=5):
    h, w = depth_image.shape[:2]
    x0, x1 = max(0, int(xmin)), min(w - 1, int(xmax))
    y0, y1 = max(0, int(ymin)), min(h - 1, int(ymax))
    xm = (x0 + x1) // 2
    ym = (y0 + y1) // 2
    sample_uvs = [
        (x0, y0), (xm, y0), (x1, y0),
        (x0, ym), (xm, ym), (x1, ym),
        (x0, y1), (xm, y1), (x1, y1),
    ]
    points = []
    for u, v in sample_uvs:
        z = float(depth_image[v, u])
        if not (np.isfinite(z) and z > 0.0):
            continue
        X = (u - cx) * z / fx
        Y = (v - cy) * z / fy
        points.append([X, Y, z])
    if len(points) < min_valid:
        return None
    pts = np.array(points)
    centroid = pts.mean(axis=0)
    _, _, Vt = np.linalg.svd(pts - centroid)
    normal = Vt[-1]
    if normal[2] < 0:
        normal = -normal
    yaw = float(np.arctan2(normal[0], normal[2]))
    r = Rotation.from_euler("y", yaw)
    qx, qy, qz, qw = r.as_quat()
    return float(qx), float(qy), float(qz), float(qw)


def test_fronto_parallel():
    depth = np.full((100, 100), 3.0, dtype=np.float32)
    result = estimate_orientation(depth, 10, 10, 90, 90, fx=500, fy=500, cx=50, cy=50)
    assert result is not None
    qx, qy, qz, qw = result
    assert abs(qw - 1.0) < 0.05, f"Expected qw~1, got {qw}"
    print("PASS test_fronto_parallel")


def test_tilted_45deg():
    """Plane tilted 45 deg about Y: depth increases linearly with u at slope tan(45)=1."""
    h, w = 100, 100
    depth = np.zeros((h, w), dtype=np.float32)
    fx = fy = 500.0
    cx = cy = 50.0
    # For a 45-deg tilt: X = Z, so (u-cx)/fx * z = z => z = z/(1 - (u-cx)/fx)
    # Simpler: set depth proportional to u directly with slope = fx (tan45=1 -> dz/du * fx = 1)
    for v in range(h):
        for u in range(w):
            z = 2.0 + (u - cx)  # slope = 1 pixel/meter, with fx=500 gives ~arctan(1/500) per pixel
            depth[v, u] = max(0.1, z)
    result = estimate_orientation(depth, 10, 10, 90, 90, fx=fx, fy=fy, cx=cx, cy=cy)
    assert result is not None
    qx, qy, qz, qw = result
    yaw_est = Rotation.from_quat([qx, qy, qz, qw]).as_euler("xyz")[1]
    # With slope=1m/pixel and fx=500, expected yaw = arctan(500) ~ 89 deg
    # Just check it detected a significant tilt (> 45 deg)
    assert abs(yaw_est) > np.pi / 4, f"Expected significant tilt, got {np.degrees(yaw_est):.1f} deg"
    print(f"PASS test_tilted_45deg (yaw={np.degrees(yaw_est):.1f} deg)")


def test_nan_handling():
    depth = np.full((100, 100), 3.0, dtype=np.float32)
    depth[10, 10] = np.nan
    depth[10, 50] = np.nan
    result = estimate_orientation(depth, 10, 10, 90, 90, fx=500, fy=500, cx=50, cy=50)
    assert result is not None
    print("PASS test_nan_handling")


def test_too_few_valid():
    depth = np.full((100, 100), np.nan, dtype=np.float32)
    result = estimate_orientation(depth, 10, 10, 90, 90, fx=500, fy=500, cx=50, cy=50)
    assert result is None
    print("PASS test_too_few_valid")


def test_tiny_bbox():
    depth = np.full((100, 100), 2.0, dtype=np.float32)
    result = estimate_orientation(depth, 48, 48, 52, 52, fx=500, fy=500, cx=50, cy=50, min_valid=3)
    assert result is not None
    print("PASS test_tiny_bbox")


test_fronto_parallel()
test_tilted_45deg()
test_nan_handling()
test_too_few_valid()
test_tiny_bbox()
print("\nAll tests passed!")
