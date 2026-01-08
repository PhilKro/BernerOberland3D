import geopandas as gpd
import matplotlib.pyplot as plt

def plot_all_slices(gdf):
    # Plot colored by elevation
    gdf.plot(column='elevation', cmap='magma', figsize=(10, 10), legend=True)
    plt.show()  

def plot_slice_at_elevation(gdf, elevation_level):

    slice_gdf = gdf[gdf['elevation'] == elevation_level]

    if slice_gdf.empty:
        print(f"No slice found at elevation {elevation_level}m.")
        return

    slice_gdf.plot(figsize=(10, 10), color='blue', edgecolor='black')
    plt.title(f"Slice at Elevation: {elevation_level}m")
    plt.show()

if __name__ == "__main__":
    gdf = gpd.read_file("Jungfraujoch10km_custom_crop.gpkg")
    # plot_slice_at_elevation(gdf, 1000)
    plot_all_slices(gdf)