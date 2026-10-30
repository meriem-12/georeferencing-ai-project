import cv2
import numpy as np
import matplotlib.pyplot as plt
from rasterio import open as rio_open
from geopy.distance import geodesic
import os

# === Path Definitions ===
# It's recommended to use English names for directories and files for better compatibility.
base_dir = '/content/drive/MyDrive/Geotiff_Files/'
drone_geotiff_path = os.path.join(base_dir, '150m_DroneImage_Georeferenced.tif')

# === Helper Function: Normalize to uint8 ===
def normalize_to_uint8(img):
    """Normalizes a float image to a uint8 image for display and feature detection."""
    img_min, img_max = img.min(), img.max()
    if img_max - img_min > 0:
        img_norm = 255 * (img - img_min) / (img_max - img_min)
        return img_norm.astype(np.uint8)
    return img.astype(np.uint8)

# === Load and Preprocess Drone Image ===
with rio_open(drone_geotiff_path) as src1:
    drone_img_raw = src1.read(1).astype(np.float32)
    drone_transform = src1.transform
drone_img_normalized = normalize_to_uint8(drone_img_raw)
# Apply CLAHE to enhance local contrast, which helps feature detectors.
clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
drone_img_enhanced = clahe.apply(drone_img_normalized)

h_drone, w_drone = drone_img_enhanced.shape

# === Initialize AKAZE Detector and Extract Drone Features ===
akaze = cv2.AKAZE_create()
keypoints_drone, descriptors_drone = akaze.detectAndCompute(drone_img_enhanced, None)

# === Ground Truth Coordinates for the Drone's Center ===
drone_lat_gt = 39.969539
drone_lon_gt = 32.745036

# === Main Loop: Iterate through satellite images from different altitudes ===
for altitude in range(500, 5001, 500):
    print(f"\n🚀 Processing satellite image from {altitude}m...")

    satellite_filename = f'{altitude}m_SatelliteImage_Georeferenced.tif'
    satellite_path = os.path.join(base_dir, satellite_filename)

    try:
        with rio_open(satellite_path) as src2:
            satellite_img_raw = src2.read(1).astype(np.float32)
            satellite_transform = src2.transform
    except Exception as e:
        print(f"❌ Could not load {satellite_filename}, skipping. Error: {e}")
        continue

    satellite_img_normalized = normalize_to_uint8(satellite_img_raw)
    satellite_img_enhanced = clahe.apply(satellite_img_normalized)

    # Resize satellite image for faster processing. Matching is often more robust at lower resolutions.
    h_sat, w_sat = satellite_img_enhanced.shape
    satellite_img_resized = cv2.resize(satellite_img_enhanced, (w_sat // 2, h_sat // 2), interpolation=cv2.INTER_AREA)
    resize_ratio = (w_sat / satellite_img_resized.shape[1], h_sat / satellite_img_resized.shape[0])

    # === AKAZE Feature Matching ===
    keypoints_satellite, descriptors_satellite = akaze.detectAndCompute(satellite_img_resized, None)
    if descriptors_satellite is None or len(descriptors_satellite) < 4:
        print("❌ Insufficient features found in the satellite image.")
        continue

    # FLANN-based matcher for fast feature matching
    index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1) # LSH for binary descriptors like AKAZE
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors_drone, descriptors_satellite, k=2)
    
    # Apply Lowe's Ratio Test to find good matches
    good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

    if len(good_matches) < 4:
        print("❌ Not enough good matches found after ratio test.")
        continue

    # === Homography Estimation with Robust USAC_MAGSAC ===
    src_pts = np.float32([keypoints_drone[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints_satellite[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    H, mask = cv2.findHomography(src_pts, dst_pts, method=cv2.USAC_MAGSAC, ransacReprojThreshold=3.0,
                                 maxIters=10000, confidence=0.999)
    if H is None:
        print("❌ Homography estimation failed.")
        continue

    # === Calculate Estimated Position and Distance Error ===
    # Transform drone image corners to find its location on the satellite image
    corners_drone = np.float32([[0, 0], [0, h_drone], [w_drone, h_drone], [w_drone, 0]]).reshape(-1, 1, 2)
    transformed_corners = cv2.perspectiveTransform(corners_drone, H)
    
    # Calculate the center of the projected polygon
    center_pt_resized = np.mean(transformed_corners.reshape(-1, 2), axis=0)
    
    # Scale center point back to original satellite image coordinates
    satellite_center_x = center_pt_resized[0] * resize_ratio[0]
    satellite_center_y = center_pt_resized[1] * resize_ratio[1]
    
    # Convert pixel coordinates to geographic coordinates
    satellite_lon_est, satellite_lat_est = satellite_transform * (satellite_center_x, satellite_center_y)
    
    # Calculate geodesic distance between ground truth and estimate
    distance_m = geodesic((drone_lat_gt, drone_lon_gt), (satellite_lat_est, satellite_lon_est)).meters
    print(f"📍 {altitude}m | Distance Error: {distance_m:.2f} meters")

    # === Visualization and Saving Results ===
    # Visualization 1: Projected bounding box on satellite image
    satellite_rgb_viz = cv2.cvtColor(satellite_img_resized, cv2.COLOR_GRAY2BGR)
    cv2.polylines(satellite_rgb_viz, [np.int32(transformed_corners)], True, (0, 215, 255), 3, cv2.LINE_AA)
    
    fig1 = plt.figure(figsize=(12, 12))
    plt.imshow(satellite_rgb_viz)
    plt.title(f'Estimated Drone Location on Satellite Image ({altitude}m)', fontsize=18)
    plt.axis('off')
    fig1.savefig(os.path.join(base_dir, f'akaze_bbox_{altitude}m.png'), dpi=300, bbox_inches='tight')

    # Visualization 2: Inlier match lines between images
    img_matches = cv2.drawMatches(drone_img_enhanced, keypoints_drone, satellite_img_resized, keypoints_satellite,
                                  [m for i, m in enumerate(good_matches) if mask[i] == 1], None,
                                  matchColor=(0, 255, 0), singlePointColor=None, flags=2)

    fig2 = plt.figure(figsize=(20, 10))
    plt.imshow(img_matches)
    plt.title(f'AKAZE Inlier Matches - {altitude}m', fontsize=18)
    plt.axis('off')
    fig2.savefig(os.path.join(base_dir, f'akaze_matches_{altitude}m.png'), dpi=300, bbox_inches='tight')

    plt.close('all')

    # Log coordinates and distance to a text file
    log_path = os.path.join(base_dir, f'akaze_result_{altitude}m.txt')
    with open(log_path, 'w') as f:
        f.write(f"📌 AKAZE Matching Results - {altitude}m\n")
        f.write(f"Drone Center (GT):   Lat = {drone_lat_gt:.8f}, Lon = {drone_lon_gt:.8f}\n")
        f.write(f"Est. Satellite Match: Lat = {satellite_lat_est:.8f}, Lon = {satellite_lon_est:.8f}\n")
        f.write(f"Distance Error: {distance_m:.2f} meters\n")
