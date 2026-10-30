import os
import cv2
import torch
import numpy as np
import rasterio
import kornia as K
import kornia.feature as KF
import matplotlib.pyplot as plt
from pyproj import Geod
import re

# USED FOR TIME MEASUREMENT PURPOSES IN THE PAPER.


# when not using DMS, the deviation is slightly larger; DMS provides closer distance.

# --- GPU CACHE CLEANING (optional) ---
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

# --- DEVICE SELECTION ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- LoFTR MODEL SETUP ---
loftr = KF.LoFTR(pretrained='outdoor').to(device).eval()

import time
t_start = time.time()  # <-- ADD HERE

# --- Load images ---

img_uydu  = cv2.imread("......._Satellite_Image.jpg",   cv2.IMREAD_GRAYSCALE)
img_drone = cv2.imread(".....Drone_Image.jpg",cv2.IMREAD_GRAYSCALE)
if img_uydu is None or img_drone is None:
    raise RuntimeError("Images could not be loaded")

tile_size = 1024
stride    = tile_size // 2

h, w = img_uydu.shape
img_drone_resized = cv2.resize(img_drone, (tile_size, tile_size))

# 1) Collect all tile matches and find the best tile
all_pts0     = []
all_pts1     = []
best_inliers = 0
best_offset  = None
best_out     = None
tile_infos   = []
tile_idx     = 0

for y in range(0, h - tile_size + 1, stride):
    for x in range(0, w - tile_size + 1, stride):
        tile_infos.append((tile_idx, x, y))
        tile = img_uydu[y:y+tile_size, x:x+tile_size]
        t0   = torch.from_numpy(tile/255.).float()[None,None].to(device)
        t1   = torch.from_numpy(img_drone_resized/255.).float()[None,None].to(device)
        with torch.no_grad():
            try:
                out = loftr({"image0": t0, "image1": t1})
            except RuntimeError:
                torch.cuda.empty_cache()
                tile_idx += 1
                continue

        kp0 = out["keypoints0"].cpu().numpy()
        kp1 = out["keypoints1"].cpu().numpy()
        if len(kp0) < 10:
            tile_idx += 1
            continue

        pts0 = kp0 + np.array([x, y])
        pts1 = kp1.copy()
        H, mask = cv2.findHomography(pts1, pts0, cv2.RANSAC, 5.0)
        if mask is None:
            tile_idx += 1
            continue
        mask = mask.ravel().astype(bool)

        all_pts0.extend(pts0)
        all_pts1.extend(pts1)

        cnt = int(mask.sum())
        if cnt > best_inliers:
            best_inliers = cnt
            best_offset  = (x, y)
            best_out     = (tile.copy(), out, mask)

        tile_idx += 1

if best_out is None:
    raise RuntimeError("No tile matched")

t_end = time.time()
print(f"Matching time (until H_best): {(t_end - t_start):.3f} seconds")


# 2) Best tile homography (GLOBAL coordinates)
x0, y0                 = best_offset
tile_best, ob, mask_best = best_out
kp1_best = ob["keypoints1"].cpu().numpy()
kp0_best = ob["keypoints0"].cpu().numpy() + np.array([x0, y0])
H_best, _  = cv2.findHomography(kp1_best, kp0_best, cv2.RANSAC, 5.0)

# 3) Project all pts1 using H_best, select real inliers (int coordinates)
tolerance       = 5.0
outside_inliers = []
for p0, p1 in zip(all_pts0, all_pts1):
    p1f  = p1.reshape(1,1,2).astype(float)
    pred = cv2.perspectiveTransform(p1f, H_best)[0,0]
    if np.linalg.norm(pred - p0) <= tolerance:
        outside_inliers.append((int(p0[0]), int(p0[1])))

# 4) Visualization
img_out = cv2.cvtColor(img_uydu, cv2.COLOR_GRAY2BGR)

# --- Draw full grid ---
grid_color = (255,0,255)
grid_thick = 3
for gx in range(0, w+1, stride):
    cv2.line(img_out, (gx, 0), (gx, h), grid_color, grid_thick)
for gy in range(0, h+1, stride):
    cv2.line(img_out, (0, gy), (w, gy), grid_color, grid_thick)

# a) Homography projection rectangle (yellow)
corners = np.float32([[0,0],[tile_size,0],[tile_size,tile_size],[0,tile_size]]).reshape(-1,1,2)
proj_corners = cv2.perspectiveTransform(corners, H_best)
cv2.polylines(img_out, [np.int32(proj_corners)], True, (0,255,255), 6)

# b) Colorize all keypoint candidates:
#    - True matches: red
#    - Inside homography but unmatched: dark blue
#    - Other candidates: green
for pt in all_pts0:
    pt_t = (int(pt[0]), int(pt[1]))
    if pt_t in outside_inliers:
        color = (0,0,255)
    elif cv2.pointPolygonTest(proj_corners, pt_t, False) >= 0:
        color = (139,0,0)
    else:
        color = (0,255,0)
    cv2.circle(img_out, pt_t, 4 if color==(0,0,255) else 3, color, -1)

# c) Draw each tile boundary in magenta, thickness=3, and index text
for idx, x, y in tile_infos:
    x1, y1 = x + tile_size, y + tile_size
    cv2.rectangle(img_out, (x, y), (x1, y1), (255,0,255), 3)
    cv2.putText(img_out, str(idx), (x1 - 30, y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,255), 2)

# d) Highlight best tile region with bright blue box (thickness=2)
x1, y1 = x0 + tile_size, y0 + tile_size
cv2.polylines(img_out, [np.int32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])],
              True, (255, 0, 0), 6)

# e) Mark outliers of the best tile in blue
mask_b = mask_best.ravel().astype(bool)
for i, ok in enumerate(mask_b):
    if not ok:
        pt = tuple((ob["keypoints0"].cpu().numpy()[i] + np.array([x0, y0])).astype(int))
        cv2.circle(img_out, pt, 5, (255,0,0), -1)

# ——— Project the drone image center onto satellite coordinates & draw ———
# 1) Original drone dimensions
h_d, w_d = img_drone.shape

# 2) Downscaling factor (original→1024×1024)
S = np.array([
    [tile_size / w_d, 0,               0],
    [0,               tile_size / h_d, 0],
    [0,               0,               1]
])

# 3) Full homography: original drone→satellite
H_full = H_best @ S

# 4) Take the exact center of the drone image
center_drone = np.array([[[w_d/2.0, h_d/2.0]]], dtype=float)

# 5) Project to satellite image
center_sat = cv2.perspectiveTransform(center_drone, H_full)[0,0]
cx, cy = int(center_sat[0]), int(center_sat[1])
print(f"Drone center on satellite: ({cx}, {cy})")

# 6) Mark it on the image
cv2.circle(img_out, (cx, cy), 12, (0,255,255), 3)

# ——— Pixel → World coordinate conversion (via GeoTIFF) ———
with rasterio.open('........Satellite_Image_modified.tif') as src:
    gt = src.transform

Hgeo = torch.tensor([
    [gt.a, gt.b, gt.c],
    [gt.d, gt.e, gt.f],
    [0.0,  0.0,  1.0]
], dtype=torch.float32, device=device)

pix = torch.tensor([cx, cy, 1.0], dtype=torch.float32, device=device).unsqueeze(-1)
world = Hgeo @ pix
Xgeo, Ygeo = world[0,0].item(), world[1,0].item()
print(f"→ Drone image center point in real-world coordinates: X={Xgeo:.6f}, Y={Ygeo:.6f}")

# ——— Optional: manually input drone center from DMS ———
use_manual_dms = True  # ← set to True if needed
manual_lat_dms = "39°58'10.34\"N"
manual_lon_dms = "32°44'42.13\"E"

def dms_to_decimal(dms_str):
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
    print("→ Drone coordinate entered manually")
else:
    with rasterio.open('.......Drone_Image_modified.tif') as ds_dr:
        gt_dr = ds_dr.transform
        w_dr, h_dr = ds_dr.width, ds_dr.height
        cx_dr, cy_dr = w_dr/2.0, h_dr/2.0
        Xdr = gt_dr.c + cx_dr * gt_dr.a + cy_dr * gt_dr.b
        Ydr = gt_dr.f + cx_dr * gt_dr.d + cy_dr * gt_dr.e
    print("→ Drone coordinate obtained from GeoTIFF")

print(f"Drone center coordinate: X={Xdr:.6f}, Y={Ydr:.6f}")

# ——— Distance calculation (geodetic distance on WGS84) ———
geod = Geod(ellps="WGS84")
_, _, dist_m = geod.inv(Xgeo, Ygeo, Xdr, Ydr)
print(f"Real-world distance between centers: {dist_m:.2f} m")
print(f"Tile Inlier/TOTAL: {best_inliers}/{len(mask_best)} ({best_inliers/len(mask_best)*100:.1f}%)")
# f) Display results
plt.figure(figsize=(12,12),dpi=200)
plt.imshow(cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.show()
