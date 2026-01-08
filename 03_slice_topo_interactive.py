import os
import glob
import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio import features
from shapely.geometry import shape, Polygon, Point, box
from shapely import affinity
import geopandas as gpd
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons

# --- CONFIGURATION ---
DATA_DIR = "./Jungfraujoch10km"
OUTPUT_FILENAME = "Jungfraujoch10km_custom_crop.gpkg"

SLICE_HEIGHT_METERS = 100   
ELEVATION_SCALE_FACTOR = 1.0

# --- GUI CLASS ---
class ROISelector:
    def __init__(self, elevation_data, transform):
        self.data = elevation_data
        self.transform = transform
        self.rows, self.cols = elevation_data.shape
        
        # Calculate bounds in real coordinates
        self.minx = transform[2]
        self.maxy = transform[5]
        self.maxx = self.minx + (self.cols * transform[0])
        self.miny = self.maxy + (self.rows * transform[4]) # transform[4] is usually negative
        
        # Initial ROI settings (Center of map, 50% size)
        self.cx = (self.minx + self.maxx) / 2
        self.cy = (self.miny + self.maxy) / 2
        self.width = (self.maxx - self.minx) * 0.5
        self.height = (self.maxy - self.miny) * 0.5
        self.angle = 0
        self.shape_type = 'Rectangle' # or 'Ellipse'
        self.final_geometry = None
        
        # Setup Figure
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        plt.subplots_adjust(left=0.1, bottom=0.35) # Make room for sliders
        
        # Downsample for display speed (show max 1000x1000 pixels)
        step = max(1, max(self.rows, self.cols) // 1000)
        self.ax.imshow(self.data[::step, ::step], cmap='terrain', 
                       extent=[self.minx, self.maxx, self.miny, self.maxy], origin='upper')
        self.ax.set_title("Adjust Sliders to Select Area. Close window to Confirm.")
        
        # Placeholder for the selection patch
        self.patch = None
        self.update_patch()
        
        # --- WIDGETS ---
        # Helper to create slider
        def add_slider(pos, label, valmin, valmax, valinit):
            ax_s = plt.axes(pos)
            return Slider(ax_s, label, valmin, valmax, valinit=valinit)

        # Sliders
        self.s_x = add_slider([0.15, 0.25, 0.65, 0.03], 'Center X', self.minx, self.maxx, self.cx)
        self.s_y = add_slider([0.15, 0.20, 0.65, 0.03], 'Center Y', self.miny, self.maxy, self.cy)
        self.s_w = add_slider([0.15, 0.15, 0.65, 0.03], 'Width/Axis 1', 100, (self.maxx-self.minx), self.width)
        self.s_h = add_slider([0.15, 0.10, 0.65, 0.03], 'Height/Axis 2', 100, (self.maxy-self.miny), self.height)
        self.s_a = add_slider([0.15, 0.05, 0.65, 0.03], 'Rotation', 0, 360, 0)
        
        # Shape Type Radio Button
        ax_radio = plt.axes([0.85, 0.15, 0.12, 0.12])
        self.radio = RadioButtons(ax_radio, ('Rectangle', 'Ellipse'))
        
        # Callbacks
        self.s_x.on_changed(self.update)
        self.s_y.on_changed(self.update)
        self.s_w.on_changed(self.update)
        self.s_h.on_changed(self.update)
        self.s_a.on_changed(self.update)
        self.radio.on_clicked(self.change_type)
        
        plt.show() # This blocks until window is closed

    def get_geometry(self):
        # Re-calculate geometry one last time to be sure
        return self._compute_geom(self.s_x.val, self.s_y.val, self.s_w.val, self.s_h.val, self.s_a.val)

    def _compute_geom(self, x, y, w, h, angle):
        if self.shape_type == 'Rectangle':
            # Create box at origin then shift/rotate
            # box(minx, miny, maxx, maxy)
            base = box(x - w/2, y - h/2, x + w/2, y + h/2)
            geom = affinity.rotate(base, angle, origin='centroid')
        else:
            # Create circle then scale to ellipse
            base = Point(x, y).buffer(1) # Unit circle
            # Scale x and y (buffer creates radius 1, so we scale by w/2 and h/2)
            scaled = affinity.scale(base, xfact=w/2, yfact=h/2)
            geom = affinity.rotate(scaled, angle, origin='centroid')
        return geom

    def update_patch(self):
        # Remove old patch
        if self.patch:
            self.patch.remove()
        
        # Create Shapely geom based on sliders (or defaults)
        if hasattr(self, 's_x'):
            geom = self._compute_geom(self.s_x.val, self.s_y.val, self.s_w.val, self.s_h.val, self.s_a.val)
        else:
            geom = self._compute_geom(self.cx, self.cy, self.width, self.height, self.angle)
            
        # Convert to Matplotlib plotting format
        x, y = geom.exterior.xy
        self.patch, = self.ax.plot(x, y, 'r-', linewidth=2)
        self.fig.canvas.draw_idle()

    def update(self, val):
        self.update_patch()
        
    def change_type(self, label):
        self.shape_type = label
        self.update_patch()

# --- MAIN SCRIPT ---
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
    
    print(f"Data Loaded. Grid Size: {elevation.shape}")

    # --- 2. INTERACTIVE SELECTION ---
    print("Opening ROI Selector...")
    selector = ROISelector(elevation, out_trans)
    
    # Get the polygon from the user
    roi_shape = selector.get_geometry()
    
    print("Masking data to selected area...")
    # Create a mask for the raster
    # rasterio geometry_mask creates a boolean array (True = Inside, False = Outside)
    # invert=True means we want the Inside to be True (or kept)
    mask_image = features.geometry_mask([roi_shape], transform=out_trans, invert=True, out_shape=elevation.shape)
    
    # Apply Mask: Set everything OUTSIDE the shape to NaN
    elevation[~mask_image] = np.nan

    # --- 3. STANDARD PROCESSING ---

    # Apply Exaggeration
    if ELEVATION_SCALE_FACTOR != 1.0:
        elevation = elevation * ELEVATION_SCALE_FACTOR

    # Define Levels
    min_elev = np.nanmin(elevation)
    max_elev = np.nanmax(elevation)
    
    start_level = np.floor(min_elev / SLICE_HEIGHT_METERS) * SLICE_HEIGHT_METERS
    levels = np.arange(start_level, max_elev, SLICE_HEIGHT_METERS)
    
    print(f"Slicing {len(levels)} layers from {min_elev:.0f}m to {max_elev:.0f}m...")

    # Generate Polygons
    all_shapes = []

    for level in tqdm(levels):
        mask = (elevation >= level).astype(np.uint8)
        
        if not np.any(mask):
            continue

        shapes_gen = features.shapes(mask, transform=out_trans)

        for geojson_geom, value in shapes_gen:
            if value == 1:
                poly = shape(geojson_geom)
                all_shapes.append({
                    'elevation': float(level),
                    'geometry': poly
                })

    # Save
    print(f"Created {len(all_shapes)} closed polygons.")
    
    if all_shapes:
        gdf = gpd.GeoDataFrame(all_shapes, crs="EPSG:2056")
        
        # Save Geometry (the cropping shape) separately?
        # If you want to see the boundary, you can create a second file or just trust the crop.
        
        gdf['geometry'] = gdf.simplify(tolerance=1.0, preserve_topology=True)
        
        print(f"Saving to {OUTPUT_FILENAME}...")
        gdf.to_file(OUTPUT_FILENAME, driver="GPKG")
        print("Done.")
    else:
        print("No shapes generated (maybe selection was empty?).")

if __name__ == "__main__":
    main()