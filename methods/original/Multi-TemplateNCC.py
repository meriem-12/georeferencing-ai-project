import cv2
import numpy as np
import matplotlib.pyplot as plt
from rasterio import open as rio_open
from geopy.distance import geodesic
import os

# === Path Definitions ===
base_dir = '/content/drive/MyDrive/Geotiff_Files/' # Using English folder name is recommended
drone_geotiff_path = os.path.join(base_dir, '150m_DroneImage_Georeferenced.tif')
log_path = os.path.join(base_dir, 'multitemplate_ncc_voting_log.txt')
coord_log_path = os.path.join(base_dir, 'coordinate_comparison_log.txt')

# === Normalization Function ===
def normalize_to_uint8(img):
    """Normalizes a float image to a uint8 image."""
    img_min, img_max = img.min(), img.max()
    if img_max - img_min > 0:
        img_norm = 255 * (img - img_min) / (img_max - img_min)
        return img_norm.astype(np.uint8)
    return img.astype(np.uint8)

# === Template Augmentation Generator ===
def generate_templates(base_img):
    """Generates augmented templates from a base image to handle variations."""
    templates = [base_img]
    # Rotational variations
    for angle in [15, -15]:
        M = cv2.getRotationMatrix2D((base_img.shape[1] // 2, base_img.shape[0] // 2), angle, 1.0)
        rotated = cv2.warpAffine(base_img, M, (base_img.shape[1], base_img.shape[0]), borderMode=cv2.BORDER_REFLECT)
        templates.append(rotated)
    # Scale variations
    for scale in [0.9, 1.1]:
        resized = cv2.resize(base_img, (0, 0), fx=scale, fy=scale)
        h, w = base_img.shape
        resized = cv2.resize(resized, (w, h)) # Resize back to original dimensions
        templates.append(resized)
    # Contrast variation using CLAHE
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    templates.append(clahe.apply(base_img))
    return templates

# === Height Matching Helper Function ===
def resize_to_match_height(img, target_height):
    """Resizes an image to a target height while maintaining aspect ratio."""
    h, w = img.shape[:2]
    if h == target_height:
        return img
    scale = target_height / h
    new_w = int(w * scale)
    return cv2.resize(img, (new_w, target_height), interpolation=cv2.INTER_AREA)

# === Load Drone Image and Prepare Template Variants ===
with rio_open(drone_geotiff_path) as src1:
    drone_img_raw = src1.read(1).astype(np.float32)
drone_img_normalized = normalize_to_uint8(drone_img_raw)
templates = generate_templates(drone_img_normalized)
h_drone, w_drone = drone_img_normalized.shape

# === Ground Truth Drone Coordinates ===
drone_lat, drone_lon = 39.969539, 32.745036

# === Initialize Log Files ===
with open(log_path, 'w') as f:
    f.write("Altitude(m),Distance(m),Inliers,Outliers,Percentage(%),Drone_Lat,Drone_Lon,Satellite_Lat,Satellite_Lon\n")
with open(coord_log_path, 'w') as f:
    f.write("Drone-to-Satellite Matching Results (in Degrees and Meters):\n\n")

# === Main Loop for Processing Satellite Images ===
for altitude in range(500, 5001, 500):
    print(f"\n🛰️ Processing satellite image from {altitude}m...")

    satellite_filename = f'{altitude}m_SatelliteImage_Georeferenced.tif'
    satellite_path = os.path.join(base_dir, satellite_filename)

    try:
        with rio_open(satellite_path) as src2:
            satellite_img_raw = src2.read(1).astype(np.float32)
            satellite_transform = src2.transform
    except Exception as e:
        print(f"❌ Could not load {satellite_filename}: {e}")
        continue

    satellite_img_normalized = normalize_to_uint8(satellite_img_raw)
    vote_map = np.zeros_like(satellite_img_normalized, dtype=np.float32)
    all_match_locations = []

    # Apply multi-template matching and voting
    for template in templates:
        res = cv2.matchTemplate(satellite_img_normalized, template, cv2.TM_CCOEFF_NORMED)
        _, _, _, max_loc = cv2.minMaxLoc(res)
        x, y = max_loc
        # Cast a vote for the entire matched region
        vote_map[y:y + h_drone, x:x + w_drone] += 1
        all_match_locations.append((x, y))

    # === Find the location with the highest number of votes ===
    _, max_vote_count, _, top_left_corner = cv2.minMaxLoc(vote_map)
    x_final, y_final = top_left_corner
    
    # Calculate the center of the final bounding box
    satellite_center_x = x_final + w_drone // 2
    satellite_center_y = y_final + h_drone // 2
    
    # Convert pixel coordinates to geographic coordinates
    satellite_lon, satellite_lat = satellite_transform * (satellite_center_x, satellite_center_y)
    
    # Calculate geodesic distance
    distance_m = geodesic((drone_lat, drone_lon), (satellite_lat, satellite_lon)).meters

    # Calculate inlier/outlier statistics based on votes
    total_votes = len(templates)
    inlier_count = int(max_vote_count) # Max votes at the best pixel
    outlier_count = total_votes - inlier_count
    percentage = 100.0 * inlier_count / total_votes if total_votes > 0 else 0

    print(f"✅ Prediction: ({x_final},{y_final}) | Distance: {distance_m:.2f} m | Inliers: {inlier_count}, Outliers: {outlier_count}, Percentage: {percentage:.2f}%")

    # === Write to Log Files ===
    with open(log_path, 'a') as f:
        f.write(f"{altitude},{distance_m:.2f},{inlier_count},{outlier_count},{percentage:.2f},{drone_lat:.6f},{drone_lon:.6f},{satellite_lat:.6f},{satellite_lon:.6f}\n")
    with open(coord_log_path, 'a') as f:
        f.write(f"🔹 {altitude} m:\n")
        f.write(f"   Distance: {distance_m:.2f} meters | Inliers: {inlier_count}, Outliers: {outlier_count}, Percentage: {percentage:.2f}%\n")
        f.write(f"   Drone GT: {drone_lat:.6f}°, {drone_lon:.6f}°\n")
        f.write(f"   Sat. Est: {satellite_lat:.6f}°, {satellite_lon:.6f}°\n\n")

    # === Visualization 1: Bounding box on satellite image ===
    satellite_rgb = cv2.cvtColor(satellite_img_normalized, cv2.COLOR_GRAY2BGR)
    thickness_main = max(2, int((altitude / 1000) * 2))
    cv2.rectangle(satellite_rgb, (x_final, y_final), (x_final + w_drone, y_final + h_drone), (0, 255, 255), thickness_main)
    
    fig = plt.figure(figsize=(12, 12))
    plt.imshow(satellite_rgb)
    plt.title(f'Drone Location Estimate - Satellite Image at {altitude}m', fontsize=16)
    plt.axis('off')
    fig.savefig(os.path.join(base_dir, f'drone_multitemplate_{altitude}m.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # === Visualization 2: Match contribution lines ===
    satellite_color = cv2.cvtColor(satellite_img_normalized, cv2.COLOR_GRAY2BGR)
    drone_color = cv2.cvtColor(drone_img_normalized, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(drone_color, (0, 0), (w_drone - 1, h_drone - 1), (0, 255, 255), 2) # Yellow border for drone

    target_height = max(drone_color.shape[0], satellite_color.shape[0])
    drone_color = resize_to_match_height(drone_color, target_height)
    satellite_color = resize_to_match_height(satellite_color, target_height)
    white_strip = 255 * np.ones((target_height, 50, 3), dtype=np.uint8)
    combined_img = np.hstack((drone_color, white_strip, satellite_color))

    # Draw lines from drone center to each template's match location
    for x, y in all_match_locations:
        pt1 = (drone_color.shape[1] // 2, target_height // 2)
        pt2_x = x + w_drone // 2 + drone_color.shape[1] + 50
        pt2_y = y + h_drone // 2
        # Ensure the point is within the combined image bounds before drawing
        if 0 <= pt2_x < combined_img.shape[1] and 0 <= pt2_y < combined_img.shape[0]:
            cv2.line(combined_img, pt1, (pt2_x, pt2_y), (0, 255, 255), 2)

    # Draw the final bounding box and center point
    x1_combined = drone_color.shape[1] + 50 + x_final
    y1_combined = y_final
    cv2.rectangle(combined_img, (x1_combined, y1_combined), (x1_combined + w_drone, y1_combined + h_drone), (0, 255, 255), 4)
    center_x_combined = x1_combined + w_drone // 2
    center_y_combined = y1_combined + h_drone // 2
    cv2.circle(combined_img, (center_x_combined, center_y_combined), 12, (0, 0, 255), -1) # Red center dot
    cv2.circle(combined_img, (center_x_combined, center_y_combined), 16, (0, 255, 255), 3) # Yellow circle

    fig2 = plt.figure(figsize=(20, 10))
    plt.imshow(cv2.cvtColor(combined_img, cv2.COLOR_BGR2RGB))
    plt.title(f'Matching Contributions (Multi-Template NCC) - {altitude}m', fontsize=18)
    plt.axis('off')
    fig2.savefig(os.path.join(base_dir, f'multitemplate_matchlines_{altitude}m.png'), dpi=300, bbox_inches='tight')
    plt.close()
