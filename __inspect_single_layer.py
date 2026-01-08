import geopandas as gpd
import matplotlib.pyplot as plt
from shapely import affinity

# --- CONFIGURATION ---
INPUT_FILE = "grindelwald_slices_100m_exaggerated.gpkg"
TARGET_ELEVATION = 2000  # Change this to the height you want to see
MODEL_SCALE = 1 / 50000  # Must match your packing script

# Paper Settings (A4 in mm)
PAPER_WIDTH_MM = 210
PAPER_HEIGHT_MM = 297

def main():
    # 1. Load Data
    print(f"Loading {INPUT_FILE}...")
    gdf = gpd.read_file(INPUT_FILE)
    
    # 2. Check available elevations
    unique_elevs = sorted(gdf['elevation'].unique())
    print(f"Available layers: {unique_elevs}")
    
    if TARGET_ELEVATION not in unique_elevs:
        print(f"Error: Elevation {TARGET_ELEVATION} not found in file.")
        print(f"Closest match: {min(unique_elevs, key=lambda x:abs(x-TARGET_ELEVATION))}")
        return

    # 3. Filter for the specific layer
    layer = gdf[gdf['elevation'] == TARGET_ELEVATION].copy()
    print(f"Found {len(layer)} shapes at {TARGET_ELEVATION}m.")

    # 4. Prepare Plot
    fig, ax = plt.subplots(figsize=(PAPER_WIDTH_MM/25.4, PAPER_HEIGHT_MM/25.4))
    ax.set_xlim(0, PAPER_WIDTH_MM)
    ax.set_ylim(0, PAPER_HEIGHT_MM)
    ax.set_aspect('equal')
    
    # Draw A4 Border
    rect = plt.Rectangle((0, 0), PAPER_WIDTH_MM, PAPER_HEIGHT_MM, 
                         linewidth=1, edgecolor='black', facecolor='white')
    ax.add_patch(rect)
    ax.text(5, PAPER_HEIGHT_MM - 10, f"Layer: {TARGET_ELEVATION}m (Scale 1:{int(1/MODEL_SCALE)})", fontsize=10)

    # 5. Plot Shapes
    # We center the shape on the page for inspection
    for idx, row in layer.iterrows():
        geom = row.geometry
        
        # Scale to mm
        scaled_geom = affinity.scale(geom, xfact=MODEL_SCALE*1000, yfact=MODEL_SCALE*1000, origin=(0,0))
        
        # Center on page
        minx, miny, maxx, maxy = scaled_geom.bounds
        w = maxx - minx
        h = maxy - miny
        
        # Shift to center of A4
        shift_x = (PAPER_WIDTH_MM - w) / 2 - minx
        shift_y = (PAPER_HEIGHT_MM - h) / 2 - miny
        
        final_geom = affinity.translate(scaled_geom, xoff=shift_x, yoff=shift_y)
        
        # Plot
        if final_geom.geom_type == 'LineString':
            x, y = final_geom.xy
            ax.plot(x, y, color='blue', linewidth=1)
        elif final_geom.geom_type == 'Polygon':
            x, y = final_geom.exterior.xy
            ax.plot(x, y, color='blue', linewidth=1)
            
    print("Displaying plot...")
    plt.show()

if __name__ == "__main__":
    main()