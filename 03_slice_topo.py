import os
import glob
import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio import features
from shapely.geometry import shape, Polygon
import geopandas as gpd
from tqdm import tqdm

# --- CONFIGURATION ---
DATA_DIR = "./Jungfraujoch10km"
OUTPUT_FILENAME = "Jungfraujoch10km_sliced_100m.gpkg"

SLICE_HEIGHT_METERS = 100   
ELEVATION_SCALE_FACTOR = 1 

def main():
    # 1. LOAD AND STITCH
    search_path = os.path.join(DATA_DIR, "*.tif")
    tif_files = glob.glob(search_path)
    
    if not tif_files:
        print("No TIF files found.")
        return

    print("Loading terrain data...")
    src_files = [rasterio.open(fp) for fp in tif_files]
    try:
        mosaic, out_trans = merge(src_files)
        elevation = mosaic[0]
    finally:
        for src in src_files:
            src.close()

    # Handle nodata
    elevation = elevation.astype(float)
    elevation[elevation < 0] = np.nan

    # Apply Exaggeration
    if ELEVATION_SCALE_FACTOR != 1.0:
        elevation = elevation * ELEVATION_SCALE_FACTOR

    # 2. DEFINE LEVELS
    min_elev = np.nanmin(elevation)
    max_elev = np.nanmax(elevation)
    
    # Start slightly above min to avoid a giant rectangle covering everything
    start_level = np.floor(min_elev / SLICE_HEIGHT_METERS) * SLICE_HEIGHT_METERS
    levels = np.arange(start_level, max_elev, SLICE_HEIGHT_METERS)
    
    print(f"Slicing {len(levels)} layers (Threshold Method)...")

    # 3. GENERATE CLOSED POLYGONS
    all_shapes = []

    for level in tqdm(levels):
        # A. Create a Boolean Mask (1 = Land, 0 = Air)
        # This determines "Inside" vs "Outside"
        mask = (elevation >= level).astype(np.uint8)
        
        # If the layer is empty, skip
        if not np.any(mask):
            continue

        # B. Polygonize the mask
        # rasterio.features.shapes returns a generator of (geojson_geometry, value)
        # We pass the transform so it calculates real-world coordinates immediately
        shapes_gen = features.shapes(mask, transform=out_trans)

        for geojson_geom, value in shapes_gen:
            # We only want the shapes where value == 1 (The mountains)
            if value == 1:
                # Convert GeoJSON to Shapely
                poly = shape(geojson_geom)
                
                # C. Clean up
                # Sometimes pixelation creates tiny artifacts. 
                # We can filter out tiny specs (e.g. < 100m area) if needed, 
                # but usually Swisstopo is clean.
                
                all_shapes.append({
                    'elevation': float(level),
                    'geometry': poly
                })

    # 4. SAVE
    print(f"Created {len(all_shapes)} closed polygons.")
    
    if all_shapes:
        gdf = gpd.GeoDataFrame(all_shapes, crs="EPSG:2056")
        
        # Optional: Simplify slightly to remove "stair-step" pixel edges
        # Tolerance of 1.0 meter preserves detail but smooths the pixel jaggedness
        gdf['geometry'] = gdf.simplify(tolerance=1.0, preserve_topology=True)
        
        print(f"Saving to {OUTPUT_FILENAME}...")
        gdf.to_file(OUTPUT_FILENAME, driver="GPKG")
        print("Done.")
    else:
        print("No shapes generated.")

if __name__ == "__main__":
    main()