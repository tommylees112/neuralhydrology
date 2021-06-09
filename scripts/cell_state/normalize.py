import xarray as xr 
import numpy as np

from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


def normalize_xr_by_basin(ds: xr.Dataset) -> xr.Dataset:
    return (ds - ds.mean(dim="time")) / ds.std(dim="time")


def normalize_cell_states(cell_state: np.ndarray, desc: str = "Normalize") -> np.ndarray:
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
    basin_coord: str = "station_id"
) -> xr.Dataset:
    #  Normalize all station values in cs_data:
    all_normed = []
    for station in c_state.station_id.values:
        norm_state = normalize_cell_states(
            c_state.sel(station_id=station)[cell_state_var].values
        )
        all_normed.append(norm_state)
    
    # stack the normalized numpy arrays
    all_normed_stack = np.stack(all_normed)
    #  [time, station_id, dimension]
    #  work out how to do transpose [NOTE: assumes all sizes are different]
    time_ix = np.where(np.array(all_normed_stack.shape) == len(c_state[time_coord]))[0][0]
    basin_ix = np.where(np.array(all_normed_stack.shape) == len(c_state[basin_coord]))[0][0]
    dimension_ix = np.where(np.array(all_normed_stack.shape) == len(c_state['dimension']))[0][0]
    all_normed_stack = all_normed_stack.transpose(time_ix, basin_ix, dimension_ix)

    norm_c_state = xr.ones_like(c_state[cell_state_var])
    norm_c_state = norm_c_state * all_normed_stack

    return norm_c_state