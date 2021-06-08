import geopandas as gpd
import xarray as xr 
from pathlib import Path 
import matplotlib.pyplot as plt


def initialise_gb_spatial_plot(ax = None, data_dir: Path = Path("/datadrive/data")):
    # read UK outline data
    assert (data_dir / "RUNOFF/natural_earth_hires/ne_10m_admin_0_countries.shp").exists(), "Download the natural earth hires from https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/10m/cultural/ne_10m_admin_0_countries.zip"

    world = gpd.read_file(data_dir / "RUNOFF/natural_earth_hires/ne_10m_admin_0_countries.shp")
    uk = world.query("ADM0_A3 == 'GBR'")
    
    # plot UK outline
    if ax is None:
        f, ax = plt.subplots(figsize=(5, 8))
    uk.plot(facecolor="none", edgecolor="k", ax=ax, linewidth=0.3)

    ax.set_xlim([-8.2, 2.1])
    ax.set_ylim([50, 59.5])
    ax.axis("off")
    return ax 


def load_latlon_points(data_dir: Path) -> gpd.GeoSeries:
    static = xr.open_dataset(data_dir / "camels_static.nc")
    d = static[["gauge_lat", "gauge_lon"]].to_dataframe()
    points = gpd.GeoSeries(gpd.points_from_xy(d["gauge_lon"], d["gauge_lat"]), index=d.index, crs="epsg:4326")
    points.name = "geometry"
    return points