import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from models.matching import Matching
from pyproj import Geod
import time


#10.10.2025 USED TO MEASURE IMAGE PROCESSING TIME

# --- GPU CACHE CLEANING (optional) ---
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

# --- DEVICE SELECTION ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- SuperGlue MODEL SETTINGS ---
config = {
    'superpoint': {
        'nms_radius': 3,
        'keypoint_threshold': 0.002,
        'max_keypoints': 4096
    },
    'superglue': {
        'weights': 'outdoor',
        'sinkhorn_iterations': 100,
        'match_threshold': 0.3
    }
}
matching = Matching(config).eval().to(device)

# --- LOAD IMAGES ---
img_sat = cv2.imread(
    '...............Satellite_Image.jpg',
    cv2.IMREAD_GRAYSCALE
)
img_drone = cv2.imread(
    '...............Drone_Image.jpg',
    cv2.IMREAD_GRAYSCALE
)
if img_sat is None or img_drone is None:
    raise RuntimeError('Images could not be loaded')


start_time = time.perf_counter()


# --- TILE PARAMETERS ---
tile_size = 1024
stride    = tile_size // 2
h, w      = img_sat.shape

# --- Resize the drone image and convert to tensor ---
drone_small = cv2.resize(img_drone, (tile_size, tile_size)).astype('float32') / 255.0
t1 = torch.from_numpy(drone_small)[None, None].to(device)

# --- List to collect potential matching points (as p0, p1 pairs) ---
matches_all = []  # each item: (p0_x, p0_y, p1_x, p1_y)

# Best homography information
best_inliers = 0
best_H       = None
best_offset  = None
mask_best    = None

# --- Tile scanning and matching ---
for y in range(0, h - tile_size + 1, stride):
    for x in range(0, w - tile_size + 1, stride):
        patch = img_sat[y:y+tile_size, x:x+tile_size].astype('float32') / 255.0
        t0    = torch.from_numpy(patch)[None, None].to(device)

        with torch.no_grad():
            pred = matching({'image0': t0, 'image1': t1})
        k0      = pred['keypoints0'][0].cpu().numpy()  # satellite keypoints within tile
        k1      = pred['keypoints1'][0].cpu().numpy()  # drone keypoints
        matches = pred['matches0'][0].cpu().numpy()    # each k0 corresponds to k1 index or -1

        valid_idx = np.where(matches > -1)[0]
        if valid_idx.size == 0:
            continue

        # Convert to global satellite coordinates
        pts0 = k0[valid_idx] + np.array([x, y])  # (Nx2)
        pts1 = k1[matches[valid_idx]]            # (Nx2)

        # Add all matches to matches_all list as (u0, v0, u1, v1)
        for (u0, v0), (u1, v1) in zip(pts0, pts1):
            matches_all.append((u0, v0, u1, v1))

        # At least 4 matches are needed for homography
        if valid_idx.size < 4:
            continue

        src = pts1.reshape(-1, 1, 2)   # drone_small points
        dst = pts0.reshape(-1, 1, 2)   # satellite (global) points
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if mask is None:
            continue

        num_inliers = int(mask.sum())
        if num_inliers > best_inliers:
            best_inliers = num_inliers
            best_H       = H
            best_offset  = (x, y)
            mask_best    = mask.ravel().astype(bool)

if best_H is None:
    raise RuntimeError('No tile matched')

end_time = time.perf_counter()
elapsed = end_time - start_time
print(f"Time elapsed from image loading to best homography found: {elapsed:.2f} seconds")


# --- Remove duplicate matches (based on p0) ---
matches_arr = np.array(matches_all)  # shape = (M, 4)
unique_pts, unique_indices = np.unique(
    matches_arr[:, 0:2].round().astype(int),
    axis=0,
    return_index=True
)
unique_pairs = matches_arr[unique_indices]  # shape = (K, 4)
all_pts0 = [tuple(pt) for pt in unique_pairs[:, 0:2]]
all_pts1 = [tuple(pt) for pt in unique_pairs[:, 2:4]]

# --- Visualization ---
img_out = cv2.cvtColor(img_sat, cv2.COLOR_GRAY2BGR)

# 1) Grid (magenta)
grid_color, grid_thick = (255, 0, 255), 2
for gx in range(0, w+1, stride):
    cv2.line(img_out, (gx, 0), (gx, h), grid_color, grid_thick)
for gy in range(0, h+1, stride):
    cv2.line(img_out, (0, gy), (w, gy), grid_color, grid_thick)

# 2) Best tile (blue rectangle)
x0, y0 = best_offset
x1, y1 = x0 + tile_size, y0 + tile_size
cv2.rectangle(img_out, (x0, y0), (x1, y1), (255, 0, 0), 3)

# 3) Green: all candidate matches in the background (later red inliers will be drawn over)
dot_radius    = 6
dot_thickness = -1  # -1 = filled
for (u0, v0) in all_pts0:
    cv2.circle(img_out, (int(u0), int(v0)), dot_radius, (0, 255, 0), dot_thickness)

# 4) Yellow: best homography projection rectangle
corners = np.float32([
    [0, 0],
    [tile_size, 0],
    [tile_size, tile_size],
    [0, tile_size]
]).reshape(-1, 1, 2)
proj = cv2.perspectiveTransform(corners, best_H)
cv2.polylines(img_out, [np.int32(proj)], True, (0, 255, 255), 3)

# 5) Find inliers and total candidates inside projection
inliers_pts = []
inside_candidates = []

tolerance = 5.0
for (u0, v0), (u1, v1) in zip(all_pts0, all_pts1):
    coord = (int(u0), int(u0))  # note: we use coordinate (u0, v0)
    coord = (int(u0), int(v0))

    # If this satellite point is inside the yellow polygon
    if cv2.pointPolygonTest(proj, coord, False) >= 0:
        inside_candidates.append(coord)

        # Project drone_small point using homography
        pred_pt = cv2.perspectiveTransform(
            np.array([[[u1, v1]]], dtype=float),
            best_H
        )[0, 0]

        # If within tolerance, mark as inlier; otherwise discard
        if np.linalg.norm(pred_pt - np.array([u0, v0])) <= tolerance:
            inliers_pts.append(coord)

# 6) Draw red inliers
inlier_radius = 7
for pt in inliers_pts:
    cv2.circle(img_out, pt, inlier_radius, (0, 0, 255), dot_thickness)

# 7) Show results with matplotlib, title includes Inlier/TotalInside info
plt.figure(figsize=(12, 12),dpi=200)
plt.imshow(cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB))
plt.axis('off')

total_inside = len(inside_candidates)
inlier_count = len(inliers_pts)

plt.show()
print(f"Inlier/Total: {inlier_count}/{total_inside} ({inlier_count/total_inside*100:.1f}%)")
# 8) Calculate the center of the yellow rectangle (projection center)
center_drone_resized = np.array([[[tile_size / 2.0, tile_size / 2.0]]], dtype=np.float32)
center_proj  = cv2.perspectiveTransform(center_drone_resized, best_H)[0, 0]
cx, cy = int(center_proj[0]), int(center_proj[1])
print(f"Drone center on satellite (projection center): ({cx}, {cy})")

# 9) Get the real-world coordinate of this pixel using Satellite TIFF
with rasterio.open('.....................Satellite_Image_modified.tif') as src_sat:
    gt_sat = src_sat.transform
    crs_sat = src_sat.crs
    Xproj = gt_sat.c + cx * gt_sat.a + cy * gt_sat.b
    Yproj = gt_sat.f + cx * gt_sat.d + cy * gt_sat.e
print(f"→ Projection center world coordinate (Satellite TIFF): X={Xproj:.6f}, Y={Yproj:.6f}")

# 10) Get the real-world coordinate of the drone TIFF image center
with rasterio.open('..................Drone_Image_modified.tif') as dr:
    gt_dr = dr.transform
    crs_dr = dr.crs
    w_dr, h_dr = dr.width, dr.height
    cx_dr, cy_dr = w_dr / 2.0, h_dr / 2.0
    Xdr = gt_dr.c + cx_dr * gt_dr.a + cy_dr * gt_dr.b
    Ydr = gt_dr.f + cx_dr * gt_dr.d + cy_dr * gt_dr.e
print(f"Drone TIFF center world coordinate: X={Xdr:.6f}, Y={Ydr:.6f}")

# --- 10.1) Optionally use manual DMS coordinates ---
# If the variable below is set to True, the manually entered DMS value will be used.
use_manual_dms = True  # ← set to True only when needed

# Manually entered DMS coordinates (example values)
manual_lat_dms = "39°58'10.34\"N"
manual_lon_dms = "32°44'42.13\"E"

def dms_to_decimal(dms_str):
    import re
    match = re.match(r"(\d+)°(\d+)'([\d.]+)\"([NSEW])", dms_str)
    if not match:
        raise ValueError(f"Invalid DMS format: {dms_str}")
    deg, min_, sec, dir_ = match.groups()
    decimal = int(deg) + int(min_) / 60 + float(sec) / 3600
    if dir_ in ['S', 'W']:
        decimal *= -1
    return decimal

if use_manual_dms:
    Xdr = dms_to_decimal(manual_lon_dms)
    Ydr = dms_to_decimal(manual_lat_dms)
    print("Drone center coordinate entered manually:")
else:
    print("Drone center coordinate obtained from TIFF file:")

print(f"Drone TIFF center world coordinate: X={Xdr:.6f}, Y={Ydr:.6f}")


# 11) Convert to DMS format
def decimal_to_dms(value, is_latitude):
    direction = ''
    if is_latitude:
        direction = 'N' if value >= 0 else 'S'
    else:
        direction = 'E' if value >= 0 else 'W'

    value = abs(value)
    degrees = int(value)
    minutes_float = (value - degrees) * 60
    minutes = int(minutes_float)
    seconds = (minutes_float - minutes) * 60

    return f"{degrees}°{minutes}'{seconds:.2f}\" {direction}"

# Projection center DMS
dms_proj_lat = decimal_to_dms(Yproj, is_latitude=True)
dms_proj_lon = decimal_to_dms(Xproj, is_latitude=False)

# Drone TIFF center DMS
dms_dr_lat = decimal_to_dms(Ydr, is_latitude=True)
dms_dr_lon = decimal_to_dms(Xdr, is_latitude=False)

print("Projection center (DMS):")
print(f"  Latitude:  {dms_proj_lat}")
print(f"  Longitude: {dms_proj_lon}")

print("Drone TIFF center (DMS):")
print(f"  Latitude:  {dms_dr_lat}")
print(f"  Longitude: {dms_dr_lon}")

# 12) Calculate geodetic distance using decimal degrees (not DMS)
geod = Geod(ellps="WGS84")
_, _, dist_m = geod.inv(Xproj, Yproj, Xdr, Ydr)
print(f"Real-world distance between two centers: {dist_m:.2f} m")

# 13) Mark the projection center with a yellow circle on the satellite image
dot_center = (int(cx), int(cy))
dot_radius = 15
dot_color  = (0, 255, 255)
dot_thickness = -1
cv2.circle(img_out, dot_center, dot_radius, dot_color, dot_thickness)
_ = img_out
