import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
import rioxarray
from tqdm import tqdm
from scripts.clip_netcdf_to_shapefile import (
    prepare_rio_data,
    rasterize_all_geoms,
    create_timeseries_of_masked_datasets,
)
import geopandas as gpd


def initalise_rio_geospatial(
    ds: xr.Dataset, crs: str = "epsg:3035", lon_dim: str = "x", lat_dim: str = "y"
):
    ds = ds.rio.set_spatial_dims(x_dim=lon_dim, y_dim=lat_dim)
    ds = ds.rio.write_crs(crs)
    return ds


def reproject_ds(ds: xr.Dataset, reproject_crs: str = "EPSG:4326") -> xr.Dataset:
    try:
        ds = ds.rio.reproject(reproject_crs)
    except MemoryError:
        #  do reprojection time by time
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
        #  get the data for the nearest point on the stream
        data = ds.sel(x=lon, y=lat, method="nearest")
        #  assign station_id as a coordinate
        data = data.assign_coords({"station_id": ([sid])})
        all_sids.append(data)

    #  join all station_id making station_id a dimension -> (station_id, )
    _ds = xr.concat(all_sids, dim="station_id")

    # expand the time dimension -> (time, station_id)
    if ds.time.size == 1:
        _ds = _ds.expand_dims("time")

    return _ds


def select_point_data_from_gridded_cwatm_output(
    cwatm_ds: xr.Dataset, gauge_latlons: pd.DataFrame
) -> xr.Dataset:
    all_times = []
    pbar = tqdm(cwatm_ds["time"].values, desc="Reproject and Select Station ID")
    for time in pbar:
        pbar.set_postfix_str(time)
        #  reproject a single time (memory issues)
        single_time_ds = reproject_ds(
            cwatm_ds.sel(time=time), reproject_crs="EPSG:4326"
        )
        _ds = convert_xy_to_station_id(single_time_ds, gauge_latlons)
        all_times.append(_ds)

    print("Concatenating all timesteps")
    sid_ds = xr.concat(all_times, dim="time")
    return sid_ds


def _rasterize_shapefile(
    input_ds: xr.Dataset,
    shp_filepath: Path,
    id_column: str = "ID_STRING",
    shape_dimension: str = "station_id",
    lat_dim: str = "y",
    lon_dim: str = "x",
) -> xr.Dataset:
    # 1. Create station_id xarray shape masks
    single_time_ds = reproject_ds(input_ds.isel(time=0), reproject_crs="EPSG:4326")
    gdf = gpd.read_file(shp_filepath)

    #  ensure that data properly initialised (e.g. CRS is the same)
    single_time_ds, gdf = prepare_rio_data(
        single_time_ds, gdf, lat_dim=lat_dim, lon_dim=lon_dim
    )

    #  rasterize the shapefile
    masks = rasterize_all_geoms(
        ds=single_time_ds,
        gdf=gdf,
        id_column=id_column,
        shape_dimension=shape_dimension,
        geometry_column="geometry",
        lat_dim=lat_dim,
        lon_dim=lon_dim,
    )

    return masks


def select_catchment_average_from_gridded_cwatm_output(
    cwatm_data: xr.Dataset,
    shp_filepath: Path,
    id_column: str = "ID_STRING",
    shape_dimension: str = "station_id",
    lat_dim: str = "y",
    lon_dim: str = "x",
) -> xr.Dataset:
    masks = _rasterize_shapefile(
        input_ds=cwatm_data,
        shp_filepath=shp_filepath,
        id_column=id_column,
        shape_dimension=shape_dimension,
        lat_dim=lat_dim,
        lon_dim=lon_dim,
    )

    # for each timestep extract the mean station_id values for every catchment
    pbar = tqdm(cwatm_data["time"].values, desc="Create Mean of Masked Area")
    all_times = []
    for time in pbar:
        pbar.set_postfix_str(time)
        # 2. Create timeseries of mean values (chop roi)
        single_time_ds = reproject_ds(
            cwatm_data.sel(time=time), reproject_crs="EPSG:4326"
        )
        _ds = create_timeseries_of_masked_datasets(
            ds=single_time_ds,
            masks=masks,
            shape_dimension=shape_dimension,
            lat_dim=lat_dim,
            lon_dim=lon_dim,
            use_pbar=False,
        )
        # expand the time dimension -> (time, station_id)
        if single_time_ds.time.size == 1:
            _ds = _ds.expand_dims("time")

        all_times.append(_ds)

    print("Concatenating all times")
    ds = xr.concat(all_times, dim="time")

    return ds


if __name__ == "__main__":
    data_dir = Path("/DataDrive200/data")
    cwat_dir = data_dir / "CWATM"
    input_files = {
        "Precipitation": cwat_dir / "Precipitation_daily.nc",
        "Tavg": cwat_dir / "Tavg_daily.nc",
    }
    target_file = cwat_dir / "discharge_daily.nc"
    assert all([f.exists() for f in list(input_files.values()) + [target_file]])
    TARGET = False
    INPUT = False

    # target data (DISCHARGE AT A POUR POINT)
    if TARGET:
        target = xr.open_dataset(target_file)
        target = initalise_rio_geospatial(target[["discharge"]], crs="epsg:3035")
        static = xr.open_dataset(data_dir / "static.nc")
        gauge_latlons = static[["gauge_lat", "gauge_lon"]].to_dataframe()

        #  1. reproject gridded cwatm to latlon
        #  2. get gauge lat lon
        #  3. select gauge from gridded cwatm
        sid_ds = select_point_data_from_gridded_cwatm_output(
            cwatm_ds=target, gauge_latlons=gauge_latlons
        )
        sid_ds.to_netcdf(data_dir / "cwatm_discharge.nc")

    #  input data (mean over catchment area - "lumped")
    if INPUT:
        # for var, filepath in [("Tavg", cwat_dir / "Tavg_daily.nc")]:
        # for var, filepath in [("Precipitation", cwat_dir / "Precipitation_daily.nc")]:
        for var, filepath in input_files.items():
            # load in the gridded input data
            input_ds = xr.open_dataset(filepath).drop("lambert_azimuthal_equal_area")
            input_ds = initalise_rio_geospatial(input_ds[[var]], crs="epsg:3035")
            shp_data_dir = data_dir / "CAMELS_GB_DATASET"
            shp_filepath = (
                shp_data_dir / "Catchment_Boundaries/CAMELS_GB_catchment_boundaries.shp"
            )

            #  1. reproject gridded cwatm to latlon
            #  2. get catchment shapefiles
            #  3. calculate shapefile means over catchment area
            ds = select_catchment_average_from_gridded_cwatm_output(
                cwatm_data=input_ds, shp_filepath=shp_filepath
            )
            ds.to_netcdf(data_dir / f"{var.lower()}_cwatm.nc")
