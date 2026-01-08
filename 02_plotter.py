import os
import glob
import rasterio
from rasterio.merge import merge
import numpy as np
import pyvista as pv

# --- CONFIGURATION ---
DATA_DIR = "./Jungfraujoch10km"

def main():
    # 1. Find Files
    # We look for all .tif files in the folder
    search_path = os.path.join(DATA_DIR, "*.tif")
    tif_files = glob.glob(search_path)
    
    if not tif_files:
        print(f"No .tif files found in '{DATA_DIR}'. Run download_data.py first!")
        return

    print(f"Found {len(tif_files)} tiles. Stitching them together...")

    # 2. Merge (Stitch)
    src_files = []
    try:
        for fp in tif_files:
            src_files.append(rasterio.open(fp))
        
        mosaic, out_trans = merge(src_files)
        elevation = mosaic[0] # Take the first band
    finally:
        # Always close files even if code crashes
        for src in src_files:
            src.close()

    # 3. Create 3D Mesh
    print(f"Stitched successfully. Grid size: {elevation.shape}")
    print("Generating 3D mesh...")

    # Get dimensions and spatial transform
    rows, cols = elevation.shape
    min_x = out_trans[2]
    max_y = out_trans[5]
    pixel_width = out_trans[0]
    pixel_height = out_trans[4]

    # Generate X and Y coordinates
    x_coords = np.linspace(min_x, min_x + (cols * pixel_width), cols)
    y_coords = np.linspace(max_y, max_y + (rows * pixel_height), rows)
    x_grid, y_grid = np.meshgrid(x_coords, y_coords)

    # Clean data (Swisstopo uses negative values for nodata sometimes)
    elevation[elevation < 0] = np.nan

    # Create PyVista Grid
    # Start with flat Z=0 plane
    grid = pv.StructuredGrid(x_grid, y_grid, np.zeros_like(elevation))
    # Add elevation as data
    grid.point_data["Elevation"] = elevation.flatten(order="F")
    
    # Warp by scalar (The actual 3D step)
    # factor=1.5 makes the mountains look steeper/more dramatic
    terrain = grid.warp_by_scalar(scalars="Elevation", factor=1.5)

    # 4. Visualization
    print("Opening Plotter...")
    plotter = pv.Plotter()
    
    # Add mesh with a nice colormap
    plotter.add_mesh(terrain, cmap="gist_earth", show_scalar_bar=False)

    # Calculate center for label (Jungfraujoch approx location in local grid)
    # We just grab the max height in the center of the map
    center_idx_x = cols // 2
    center_idx_y = rows // 2
    
    # # Safety check for label index
    # try:
    #     label_x = x_coords[center_idx_x]
    #     label_y = y_coords[center_idx_y]
    #     label_z = np.nanmax(elevation)
        
    #     plotter.add_point_labels(
    #         [[label_x, label_y, label_z]], 
    #         ["Jungfraujoch Region"], 
    #         point_size=20, 
    #         font_size=24,
    #         text_color="white",
    #         always_visible=True
    #     )
    # except:
    #     pass # Skip label if indices fail

    plotter.show_grid()
    plotter.show(title="SwissALTI3D Visualization")

if __name__ == "__main__":
    main()