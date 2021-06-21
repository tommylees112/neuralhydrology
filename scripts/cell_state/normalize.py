import xarray as xr
import numpy as np

from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


def normalize_xr_by_basin(ds: xr.Dataset) -> xr.Dataset:
    return (ds - ds.mean(dim="time")) / ds.std(dim="time")


def normalize_cell_states(
    cell_state: np.ndarray, desc: str = "Normalize"
) -> np.ndarray:
    """Normalize each cell state by DIMENSION"""
    original_shape = cell_state.shape
    store = []
    s = StandardScaler()
    n_dims = len(cell_state.shape)
    # (target_time, basins, dimensions)
    if n_dims == 3:
        for ix in tqdm(range(cell_state.shape[-1]), desc=desc):
            store.append(s.fit_transform(cell_state[:, :, ix]))

        c_state = np.stack(store)
        c_state = c_state.transpose(1, 2, 0)
        assert c_state.shape == original_shape

    elif n_dims == 2:
        for ix in tqdm(range(cell_state.shape[-1]), desc=desc):
            store.append(s.fit_transform(cell_state[:, ix].reshape(-1, 1)))
        c_state = np.stack(store)[:, :, 0]
        c_state = c_state.T
        assert c_state.shape == original_shape

    else:
        raise NotImplementedError

    return c_state


def normalize_xarray_cstate(
    c_state: xr.Dataset,
    cell_state_var: str = "cell_state",
    time_coord: str = "date",
    basin_coord: str = "station_id",
) -> xr.Dataset:
    #  Normalize all station values in cs_data:
    all_normed = []
    for station in c_state.station_id.values:
        norm_state = normalize_cell_states(
            c_state.sel(station_id=station)[cell_state_var].values
        )
        all_normed.append(norm_state)

    #  stack the normalized numpy arrays
    all_normed_stack = np.stack(all_normed)
    #   [time, station_id, dimension]
    #  work out how to do transpose [NOTE: assumes all sizes are different]
    time_ix = np.where(np.array(all_normed_stack.shape) == len(c_state[time_coord]))[0][
        0
    ]
    basin_ix = np.where(np.array(all_normed_stack.shape) == len(c_state[basin_coord]))[
        0
    ][0]
    dimension_ix = np.where(
        np.array(all_normed_stack.shape) == len(c_state["dimension"])
    )[0][0]
    all_normed_stack = all_normed_stack.transpose(time_ix, basin_ix, dimension_ix)

    norm_c_state = xr.ones_like(c_state[cell_state_var])
    norm_c_state = norm_c_state * all_normed_stack

    return norm_c_state


def normalize_2d_dataset(    
    ds: xr.Dataset,
    variable_str: str = "sm",
    station_dim: str = "station_id",
    time_dim: str = "time",
    per_basin: bool = True,
) -> xr.Dataset:
    out = []
    if per_basin:
        pbar = tqdm(ds[station_dim].values, desc="Normalising each station")

        for sid in pbar:
            s = StandardScaler()
            # [1, time]
            normed = s.fit_transform(ds[variable_str].sel({station_dim: sid}).values.reshape(-1, 1))
            
            out.append(normed)
    else:  # per variable only
        # [1, (n_time * n_basin)]
        normed = (ds[variable_str] - ds[variable_str].mean()) / ds[variable_str].std()
        out.append(normed)

    # [time, station_id]
    normed_data = np.hstack(out)

    # Transpose IF required
    station_dim_n = len(ds[station_dim].values)
    time_dim_n = len(ds[time_dim].values)

    station_ix = int(np.argwhere(np.array(normed_data.shape) == station_dim_n))
    time_ix = int(np.argwhere(np.array(normed_data.shape) == time_dim_n))

    # [station_dim, time_dim] 
    if normed_data.shape != ds[variable_str].shape:
        normed_data = normed_data.transpose(station_ix, time_ix)
    assert normed_data.shape == ds[variable_str].shape

    # convert to xarray
    ds_norm = xr.ones_like(ds[variable_str]) * normed_data
    return ds_norm


def normalize_cstate(
    ds: xr.Dataset,
    variable_str: str = "c_n",
    station_dim: str = "station_id",
    dimension_dim: str = "dimension",
) -> xr.Dataset:
    out = []
    N_dims = len(ds[dimension_dim].values)

    pbar = tqdm(np.arange(len(ds[station_dim].values)), desc="Normalising each station-dimension")
    for sid in pbar:
        station_arr = []
        for did in np.arange(N_dims):
            s = StandardScaler()
            
            normed = s.fit_transform(ds[variable_str][:, sid, did].values.reshape(-1, 1))
            station_arr.append(normed)
            
        # [time, 1, dimension]
        station_data = np.hstack(station_arr).reshape(-1, 1, N_dims)
        out.append(station_data)

    # [time, station_id, dimension]
    normed_data = np.hstack(out)
    assert normed_data.shape == ds[variable_str].shape

    # convert to xarray
    ds_norm = xr.ones_like(ds[variable_str]) * normed_data
    return ds_norm