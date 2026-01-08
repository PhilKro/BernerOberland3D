import os
import shutil
import requests
from tqdm import tqdm

# --- CONFIGURATION ---
# DATA_DIR = "./Finsteraarhorn10km"
DATA_DIR = "./Jungfraujoch10km"

##------ JUNGFRAUJOCH ---
# Jungfraujoch Coordinates (Approximate Center)
CENTER_LAT = 46.5475
CENTER_LON = 7.9851

# 10km box means +/- 5km from center
# 1 deg lat ~= 111km, 1 deg lon at this lat ~= 76km
LAT_OFFSET = 5 / 111
LON_OFFSET = 5 / 76

BBOX = [
    CENTER_LON - LON_OFFSET,
    CENTER_LAT - LAT_OFFSET,
    CENTER_LON + LON_OFFSET,
    CENTER_LAT + LAT_OFFSET
]
BBOX_STR = f"{BBOX[0]},{BBOX[1]},{BBOX[2]},{BBOX[3]}"

#----- Finsteraarhorn ---
# # Center Point (Located roughly at the Lower Grindelwald Glacier)
# CENTER_LAT = 46.56
# CENTER_LON = 8.07

# # Offsets calculated to capture:
# # - West limit: ~8.00 (Just includes Eiger, cuts off Jungfrau at 7.96)
# # - East limit: ~8.14 (Includes Finsteraarhorn at 8.12)
# LON_OFFSET = 0.075  # ~6km to the left and right
# LAT_OFFSET = 0.045  # ~5km up and down

# BBOX = [
#     CENTER_LON - LON_OFFSET,
#     CENTER_LAT - LAT_OFFSET,
#     CENTER_LON + LON_OFFSET,
#     CENTER_LAT + LAT_OFFSET
# ]
# BBOX_STR = f"{BBOX[0]},{BBOX[1]},{BBOX[2]},{BBOX[3]}"

# --- MAIN LOGIC ---
def main():
    # 1. Folder Management
    if os.path.exists(DATA_DIR):
        print(f"Folder '{DATA_DIR}' already exists.")
        response = input("Do you want to clear it and re-download? (y/n): ").lower().strip()
        if response == 'y':
            shutil.rmtree(DATA_DIR)
            os.makedirs(DATA_DIR)
            print("Folder cleared.")
        else:
            print("Keeping existing files. No files will be downloaded.")
            exit(0)
    else:
        os.makedirs(DATA_DIR)
        print(f"Created folder '{DATA_DIR}'")

    # 2. Query API
    stac_url = "https://data.geo.admin.ch/api/stac/v0.9/collections/ch.swisstopo.swissalti3d/items"
    params = {"bbox": BBOX_STR, "limit": 500}

    print(f"Querying Swisstopo for area: {BBOX_STR}")
    try:
        resp = requests.get(stac_url, params=params)
        resp.raise_for_status()
        features = resp.json()["features"]
    except Exception as e:
        print(f"API Error: {e}")
        return

    print(f"Found {len(features)} tiles.")

    # 3. Download Loop
    print("Starting download...")
    count = 0
    for item in tqdm(features):
        assets = item["assets"]
        selected_url = None

        # Priority: 2m resolution TIF > Any TIF
        # We iterate over values() to ignore the messy keys

        # Check for 2m
        for asset in assets.values():
            if "image/tiff" in asset.get("type", "") and asset.get("eo:gsd") == 2.0:
                selected_url = asset["href"]
                break

        # Fallback to any TIF
        if not selected_url:
            for asset in assets.values():
                if "image/tiff" in asset.get("type", ""):
                    selected_url = asset["href"]
                    break

        if selected_url:
            # Save file
            fname = os.path.join(DATA_DIR, f"{item['id']}.tif")
            if not os.path.exists(fname):
                try:
                    with requests.get(selected_url, stream=True) as r:
                        r.raise_for_status()
                        with open(fname, 'wb') as f:
                            for chunk in r.iter_content(chunk_size=8192):
                                f.write(chunk)
                    count += 1
                except Exception as e:
                    print(f"Failed {fname}: {e}")

    print(f"Job done. Downloaded {count} new files into '{DATA_DIR}'.")

if __name__ == "__main__":
    main()