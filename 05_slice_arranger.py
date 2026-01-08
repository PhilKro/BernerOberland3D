import geopandas as gpd
from shapely.geometry import Polygon, LineString, Point
from shapely import affinity
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from rectpack import newPacker, PackingMode
from rectpack.maxrects import MaxRectsBssf
import numpy as np
import math
import sys
from tqdm import tqdm

# --- CONFIGURATION ---
INPUT_FILE = "Jungfraujoch10km_custom_crop.gpkg"
OUTPUT_PDF = "Jungfraujoch10km_custom_crop_printable.pdf"
MODEL_SCALE = 1/56000 
TEST_LIMIT = None 

# Paper Settings (A4 in mm)
PAPER_WIDTH_MM = 210
PAPER_HEIGHT_MM = 297
MARGIN_MM = 10
MIN_AREA_MM2 = 25.0 

# Scale Bar Settings
SCALE_BAR_REAL_KM = 1.0  # Length of the bar in real world (e.g. 1km)

def get_optimal_rotation(geom):
    """Finds angle to align Minimum Rotated Rectangle with X-axis."""
    rect = geom.minimum_rotated_rectangle
    coords = list(rect.exterior.coords)
    max_len = -1.0
    best_angle = 0.0
    for i in range(3): 
        p1 = coords[i]; p2 = coords[i+1]
        dx = p2[0] - p1[0]; dy = p2[1] - p1[1]
        length = math.hypot(dx, dy)
        if length > max_len:
            max_len = length
            best_angle = math.degrees(math.atan2(dy, dx))
    return -best_angle

def rotate_point(point, angle_degrees):
    rad = math.radians(angle_degrees)
    x, y = point
    new_x = x * math.cos(rad) - y * math.sin(rad)
    new_y = x * math.sin(rad) + y * math.cos(rad)
    return (new_x, new_y)

def main():
    # 1. SETUP
    print(f"Loading {INPUT_FILE}...")
    gdf = gpd.read_file(INPUT_FILE)
    
    # Calculate Printable Area
    print_w = PAPER_WIDTH_MM - 2 * MARGIN_MM
    print_h = PAPER_HEIGHT_MM - 2 * MARGIN_MM
    print(f"Printable Area: {print_w}mm x {print_h}mm")

    gdf['length'] = gdf.geometry.length
    gdf = gdf.sort_values(by=['length', 'elevation'], ascending=[False, True])
    
    if TEST_LIMIT:
        print(f"Test Mode: top {TEST_LIMIT} shapes.")
        gdf = gdf.head(TEST_LIMIT)
    
    shapes = []
    skipped_count = 0
    
    print(f"Processing shapes (Min Area: {MIN_AREA_MM2} mm²)...")
    for idx, row in gdf.iterrows():
        geom = row.geometry
        elev = row.elevation
        
        # A. CLOSE SHAPE
        if isinstance(geom, LineString):
            coords = list(geom.coords)
            if coords[0] != coords[-1]: coords.append(coords[0])
            geom = Polygon(coords)
            
        # B. SCALE TO MM
        geom = affinity.scale(geom, xfact=MODEL_SCALE*1000, yfact=MODEL_SCALE*1000, origin=(0,0))
        
        if geom.area < MIN_AREA_MM2:
            skipped_count += 1
            continue
        
        # C. OPTIMIZE ROTATION
        rot_angle = get_optimal_rotation(geom)
        centroid = geom.centroid
        optimized_geom = affinity.rotate(geom, rot_angle, origin=centroid)
        north_vector = rotate_point((0, 1), rot_angle)
        
        # D. GET BOUNDS
        minx, miny, maxx, maxy = optimized_geom.bounds
        w = maxx - minx
        h = maxy - miny
        
        # --- SIZE CHECK ---
        fits_portrait = (w <= print_w and h <= print_h)
        fits_landscape = (w <= print_h and h <= print_w)
        
        if not (fits_portrait or fits_landscape):
            print(f"ERROR: Shape at {elev}m is TOO BIG for the paper! Shape Size: {w:.1f}mm x {h:.1f}mm. Paper Size: {print_w}mm x {print_h}mm")
            
            shape_min = min(w, h)
            shape_max = max(w, h)
            paper_min = min(print_w, print_h)
            paper_max = max(print_w, print_h)
            
            ratio_min = paper_min / shape_min
            ratio_max = paper_max / shape_max
            limiting_ratio = min(ratio_min, ratio_max)
            
            suggested_scale_factor = MODEL_SCALE * limiting_ratio * 0.95
            
            print(f"Current Scale: 1/{int(1/MODEL_SCALE)}. Suggested Scale: 1/{int(1/suggested_scale_factor)} (approx {suggested_scale_factor:.7f})")
            sys.exit(1)
            
        shapes.append({
            'id': idx,
            'geom': optimized_geom, 
            'w': w,
            'h': h,
            'elev': elev,
            'original_minx': minx,
            'original_miny': miny,
            'north_vector': north_vector
        })

    print(f"Kept {len(shapes)} shapes. Skipped {skipped_count} shapes (too small).")

    # 2. PACK
    print("Packing...")
    packer = newPacker(mode=PackingMode.Offline, pack_algo=MaxRectsBssf)
    PRECISION = 10 
    
    for i, shape in enumerate(shapes):
        packer.add_rect(width=shape['w']*PRECISION, height=shape['h']*PRECISION, rid=i)

    safe_w = (PAPER_WIDTH_MM - 2*MARGIN_MM) * PRECISION
    safe_h = (PAPER_HEIGHT_MM - 2*MARGIN_MM) * PRECISION
    
    for _ in range(len(shapes)):
        packer.add_bin(width=safe_w, height=safe_h)

    packer.pack()

    # 3. PDF GENERATION
    print(f"Generating PDF: {OUTPUT_PDF}")
    
    with PdfPages(OUTPUT_PDF) as pdf:
        for bin_idx, abin in enumerate(tqdm(packer, desc="Creating Sheets", unit="sheet")):
            fig, ax = plt.subplots(figsize=(PAPER_WIDTH_MM/25.4, PAPER_HEIGHT_MM/25.4))
            ax.set_xlim(0, PAPER_WIDTH_MM); ax.set_ylim(0, PAPER_HEIGHT_MM)
            
            # Guides
            ax.add_patch(plt.Rectangle((0, 0), PAPER_WIDTH_MM, PAPER_HEIGHT_MM, ec='black', fc='white'))
            ax.add_patch(plt.Rectangle((MARGIN_MM, MARGIN_MM), PAPER_WIDTH_MM-2*MARGIN_MM, PAPER_HEIGHT_MM-2*MARGIN_MM, ec='gray', ls='--', fc='none', lw=0.5))

            # --- SCALE BAR ---
            # Calculate length in mm: Scale(1/X) * KM * 1000m * 1000mm
            scale_bar_len_mm = MODEL_SCALE * SCALE_BAR_REAL_KM * 1000 * 1000
            
            # Position: Bottom Left Margin (x=10mm, y=5mm)
            sb_x = MARGIN_MM
            sb_y = MARGIN_MM / 2 
            
            # Draw Line
            ax.plot([sb_x, sb_x + scale_bar_len_mm], [sb_y, sb_y], color='black', linewidth=1.5)
            # Draw Ticks
            ax.plot([sb_x, sb_x], [sb_y-1, sb_y+1], color='black', linewidth=1.0)
            ax.plot([sb_x + scale_bar_len_mm, sb_x + scale_bar_len_mm], [sb_y-1, sb_y+1], color='black', linewidth=1.0)
            # Draw Text
            ax.text(sb_x + scale_bar_len_mm/2, sb_y + 1.5, f"{SCALE_BAR_REAL_KM} km", 
                    fontsize=8, ha='center', va='bottom', color='black')

            # --- PLOT SHAPES ---
            count = 0
            for rect in abin:
                shape_data = shapes[rect.rid]
                place_x = rect.x / PRECISION + MARGIN_MM
                place_y = rect.y / PRECISION + MARGIN_MM
                
                poly = shape_data['geom']
                n_vec = shape_data['north_vector']
                
                # Check 90 deg rotation by packer
                packed_w_int = int(rect.width)
                orig_w_int = int(shape_data['w'] * PRECISION)
                
                if abs(packed_w_int - orig_w_int) > 1:
                    poly = affinity.rotate(poly, 90, origin='centroid')
                    n_vec = rotate_point(n_vec, 90)
                    minx, miny, maxx, maxy = poly.bounds
                    poly = affinity.translate(poly, xoff=-minx, yoff=-miny)
                else:
                    poly = affinity.translate(poly, xoff=-shape_data['original_minx'], yoff=-shape_data['original_miny'])

                poly = affinity.translate(poly, xoff=place_x, yoff=place_y)
                
                x, y = poly.exterior.xy
                ax.plot(x, y, color='black', linewidth=0.8)
                centroid = poly.centroid
                
                # Label & Arrow
                ax.text(centroid.x, centroid.y - 2, f"{int(shape_data['elev'])}", fontsize=6, ha='center', va='top', color='red')
                ax.arrow(centroid.x, centroid.y, n_vec[0]*4, n_vec[1]*4, head_width=1.5, head_length=1.5, fc='red', ec='red', linewidth=0.5)
                
                count += 1

            ax.set_aspect('equal'); plt.axis('off')
            ax.set_title(f"Sheet {bin_idx + 1} ({count} shapes)", fontsize=8)
            pdf.savefig(fig); plt.close(fig)

    print("Done.")

if __name__ == "__main__":
    main()