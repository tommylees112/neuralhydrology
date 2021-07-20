import xarray as xr 
import numpy as np
import pandas as pd
from pathlib import Path 
import rioxarray 
from tqdm import tqdm
from scripts.clip_netcdf_to_shapefile import create_camels_basin_timeseries


def initalise_rio_geospatial(ds: xr.Dataset, crs: str = "epsg:3035", lon_dim: str = "x", lat_dim: str = "y"):
    ds = ds.rio.set_spatial_dims(x_dim=lon_dim, y_dim=lat_dim)
    ds = ds.rio.write_crs(crs)
    return ds 


def reproject_ds(ds: xr.Dataset, reproject_crs: str = "EPSG:4326") -> xr.Dataset:
    try:
        ds = ds.rio.reproject(reproject_crs)
    except MemoryError:
        # do reprojection time by time
        all_times = []
        for time in tqdm(ds.time.values, desc="Reproject Time"):
            reproject_ds = ds.sel(time=time).rio.reproject(reproject_crs)
            all_times.append(reproject_ds)

        print("Concatenating all times")
        ds = xr.concat(all_times, dim="time")

    return ds


def convert_xy_to_station_id(ds: xr.Dataset, gauge_latlons: pd.DataFrame) -> xr.Dataset:
    assert all(np.isin(["gauge_lat", "gauge_lon"], gauge_latlons.columns))
    
    all_sids = []
    # extract 
    for sid, row in gauge_latlons.iterrows():
        lat, lon = row["gauge_lat"], row["gauge_lon"]
        # get the data for the nearest point on the stream
        data = ds.sel(x=lon, y=lat, method="nearest")
        # assign station_id as a coordinate
        data = data.assign_coords({"station_id": ([sid])})
        all_sids.append(data)
    
    # join all station_id making station_id a dimension -> (station_id, )
    _ds = xr.concat(all_sids, dim="station_id")
    
    # expand the time dimension -> (time, station_id)
    if ds.time.size == 1:
        _ds = _ds.expand_dims("time")

    return _ds
    

def create_station_id_data_from_cwatm_output(
    cwatm_ds: xr.Dataset, gauge_latlons: pd.DataFrame
) -> xr.Dataset:
    all_times = []
    pbar = tqdm(cwatm_ds["time"].values, desc="Reproject and Select Station ID")
    for time in pbar:
        pbar.set_postfix_str(time)
        # reproject a single time (memory issues)
        single_time_ds = reproject_ds(cwatm_ds.sel(time=time), reproject_crs="EPSG:4326")
        _ds = convert_xy_to_station_id(single_time_ds, gauge_latlons)
        all_times.append(_ds)

    print("Concatenating all timesteps")
    sid_ds = xr.concat(all_times, dim="time")
    return sid_ds 


if __name__ == "__main__":
    data_dir = Path("/DataDrive200/data")
    cwat_dir = data_dir / "CWATM"
    input_files = {
        "Precipitation": cwat_dir / "Precipitation_daily.nc",
        "Tavg": cwat_dir / "Tavg_daily.nc", 
    }
    target_file = cwat_dir / "discharge_daily.nc"
    assert all([f.exists() for f in input_files + [target_file]])
    
    # target data (DISCHARGE AT A POUR POINT)
    target = xr.open_dataset(target_file)
    target = initalise_rio_geospatial(target[["discharge"]], crs="epsg:3035")
    static = xr.open_dataset(data_dir / "static.nc")
    gauge_latlons = static[["gauge_lat", "gauge_lon"]].to_dataframe()
    
    # 1. reproject gridded cwatm to latlon
    # 2. get gauge lat lon
    # 3. select gauge from gridded cwatm
    sid_ds = create_station_id_data_from_cwatm_output(cwatm_ds=target, gauge_latlons=gauge_latlons)
    sid_ds.to_netcdf(data_dir / "cwatm_discharge.nc")

    # input data
    # 1. reproject gridded cwatm to latlon
    # 2. get catchment shapefiles
    # 3. calculate shapefile means over catchment area
    shp_data_dir = data_dir / "CAMELS_GB_DATASET"

    for var, filepath in input_files.items():
        input_ds = xr.open_dataset(filepath).drop("lambert_azimuthal_equal_area")
        input_ds = initalise_rio_geospatial(input_ds[var], crs="epsg:3035")

        time = input_ds["time"].values[0]
        single_time_ds = reproject_ds(input_ds.sel(time=time), reproject_crs="EPSG:4326")
        out_ds = create_camels_basin_timeseries(
            sm_data=input_data, shp_data_dir=shp_data_dir
        )

        break

    