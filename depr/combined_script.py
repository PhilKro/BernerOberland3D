# %%
# CELL 1: SETUP & IMPORTS
import os
import requests
import rasterio
from rasterio.merge import merge
import numpy as np
import pyvista as pv
from tqdm import tqdm
from io import BytesIO

# Configuration
# Jungfraujoch Coordinates (Approximate Center)
CENTER_LAT = 46.5475
CENTER_LON = 7.9851

# 10km box means +/- 5km from center
# 1 deg lat ~= 111km, 1 deg lon at this lat ~= 76km
LAT_OFFSET = 5 / 111
LON_OFFSET = 5 / 76

# Create Bounding Box [min_lon, min_lat, max_lon, max_lat]
BBOX = [
    CENTER_LON - LON_OFFSET,
    CENTER_LAT - LAT_OFFSET,
    CENTER_LON + LON_OFFSET,
    CENTER_LAT + LAT_OFFSET
]

BBOX_STR = f"{BBOX[0]},{BBOX[1]},{BBOX[2]},{BBOX[3]}"
DATA_DIR = "./swiss_topo_data"
os.makedirs(DATA_DIR, exist_ok=True)

print(f"ROI Defined: 10km box around Jungfraujoch.")
print(f"Boundaries: {BBOX_STR}")

# %%
# CELL 2: SEARCH & DOWNLOAD (Fixed based on Debug Info)
stac_url = "https://data.geo.admin.ch/api/stac/v0.9/collections/ch.swisstopo.swissalti3d/items"
params = {
    "bbox": BBOX_STR,
    "limit": 500
}

print("Querying Swisstopo API...")
response = requests.get(stac_url, params=params)
response.raise_for_status()
features = response.json()["features"]

print(f"Found {len(features)} tiles covering this area.")

downloaded_files = []

print("Starting download...")
for item in tqdm(features):
    assets = item["assets"]
    selected_url = None
    
    # STRATEGY: Use the 'eo:gsd' field (Ground Sample Distance)
    # We want 2.0m resolution if available, otherwise 0.5m
    
    # 1. Try to find exactly 2.0m resolution TIFF
    for asset in assets.values():
        # Check if it's a TIFF (loose check to handle extra profile info)
        if "image/tiff" in asset.get("type", ""):
            # Check resolution
            if asset.get("eo:gsd") == 2.0:
                selected_url = asset["href"]
                break
    
    # 2. If no 2m file found, fall back to ANY TIFF (likely 0.5m)
    if not selected_url:
        for asset in assets.values():
            if "image/tiff" in asset.get("type", ""):
                selected_url = asset["href"]
                break
                
    # 3. Download
    if selected_url:
        # Create a filename based on the Item ID (unique per tile)
        fname = os.path.join(DATA_DIR, f"{item['id']}.tif")
        downloaded_files.append(fname)
        
        if not os.path.exists(fname):
            try:
                with requests.get(selected_url, stream=True) as r:
                    r.raise_for_status()
                    with open(fname, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
            except Exception as e:
                print(f"Failed to download {selected_url}: {e}")

# Safety check to ensure we actually got files before moving to Cell 3
if not downloaded_files:
    print("WARNING: No files were added to the download list.")
else:
    print(f"\nAll downloads complete. {len(downloaded_files)} files ready for merging.")
# %%
# CELL 3: MERGE DATA (STITCHING)
# This opens all files and combines them into one massive array.

print("Stitching tiles together (this may take a moment)...")

src_files_to_mosaic = []
for fp in downloaded_files:
    src = rasterio.open(fp)
    src_files_to_mosaic.append(src)

# The Merge Magic
mosaic, out_trans = merge(src_files_to_mosaic)

# Mosaic shape is (Bands, Height, Width). We need (Height, Width) for plotting
elevation = mosaic[0]

# Close handles
for src in src_files_to_mosaic:
    src.close()

# Compute new coordinates for the merged grid
# out_trans is an Affine transform: (resolution_x, 0, min_x, 0, -resolution_y, max_y)
rows, cols = elevation.shape
min_x = out_trans[2]
max_y = out_trans[5]
pixel_width = out_trans[0]
pixel_height = out_trans[4] # Usually negative

# Generate coordinate arrays
x_coords = np.linspace(min_x, min_x + (cols * pixel_width), cols)
y_coords = np.linspace(max_y, max_y + (rows * pixel_height), rows)

# Create 2D grid
x_grid, y_grid = np.meshgrid(x_coords, y_coords) # No flip needed if using valid Affine transform logic usually

# Sanity check on data (remove -9999 or nodata values if any)
elevation[elevation < 0] = np.nan

print(f"Mosaic Complete. Final Grid Size: {elevation.shape}")

# %%
# CELL 4: VISUALIZE
# Plot the massive 10x10km model

print("Preparing 3D Mesh...")

# 1. Create Grid
# Use a flat Z to start
grid = pv.StructuredGrid(x_grid, y_grid, np.zeros_like(elevation))

# 2. Add Data
grid.point_data["Elevation"] = elevation.flatten(order="F")

# 3. Warp
# We use a slight exaggeration (1.2x) to make the mountains look dramatic
terrain = grid.warp_by_scalar(scalars="Elevation", factor=1.0)

print("Rendering...")
plotter = pv.Plotter()
plotter.add_mesh(terrain, cmap="gist_earth", show_scalar_bar=False)

# Add Label for Jungfraujoch
# We locate it by finding the max height near the center of the grid
z_max = np.nanmax(elevation)
center_idx_x = cols // 2
center_idx_y = rows // 2
center_x = x_coords[center_idx_x]
center_y = y_coords[center_idx_y]

plotter.add_point_labels(
    [[center_x, center_y, z_max]], 
    ["Jungfraujoch Area"], 
    point_size=20, 
    font_size=24,
    text_color="white"
)

plotter.show_grid()
plotter.show(title="Eiger, Mönch, Jungfrau (10x10km)")