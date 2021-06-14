from typing import List, Tuple, Union, Dict
import numpy as np
import xarray as xr
import pandas as pd
from torch.utils.data import Dataset
from torch import Tensor
from tqdm import tqdm
import torch
from numba import njit, prange
from torch.utils.data import DataLoader


def get_matching_dim(ds1, ds2, dim: str) -> Tuple[np.ndarray]:
    return (
        np.isin(ds1[dim].values, ds2[dim].values),
        np.isin(ds2[dim].values, ds1[dim].values),
    )


@njit
def validate(x_d: List[np.ndarray], y: List[np.ndarray], seq_length: int):
    n_samples = len(y)
    flag = np.ones(n_samples)

    # if any condition met then go to next iteration of loop
    for target_index in prange(n_samples):
        start_input_idx = target_index - seq_length

        # 1. not enough history (seq_length > history)
        if start_input_idx < 0:
            flag[target_index] = 0
            continue

        #  2. NaN in the dynamic inputs
        _x_d = x_d[start_input_idx:target_index]
        if np.any(np.isnan(_x_d)):
            flag[target_index] = 0
            continue

        #  3. NaN in the outputs (TODO: only for training period)
        _y = y[start_input_idx:target_index]
        if np.any(np.isnan(_y)):
            flag[target_index] = 0
            continue

    return flag


class TimeSeriesDataset(Dataset):
    def __init__(
        self,
        input_data: xr.Dataset,
        target_data: xr.Dataset,
        target_variable: str,
        input_variables: List[str],
        seq_length: int = 64,
        basin_dim: str = "station_id",
        time_dim: str = "time",
        desc: str = "Creating Samples",
    ):
        self.target_variable = target_variable
        self.input_variables = input_variables
        self.seq_length = seq_length
        self.basin_dim = basin_dim
        self.time_dim = time_dim

        # matching data
        target_times, input_times = get_matching_dim(
            target_data, input_data, self.time_dim
        )
        target_sids, input_sids = get_matching_dim(
            target_data, input_data, self.basin_dim
        )

        input_data = input_data.sel(time=input_times, station_id=input_sids)
        target_data = target_data.sel(time=target_times, station_id=target_sids)

        self.input_data = input_data
        self.target_data = target_data

        self.create_lookup_table_of_valid_samples(
            input_data=self.input_data, target_data=self.target_data, desc=desc,
        )

    def create_lookup_table_of_valid_samples(
        self,
        input_data: Union[xr.Dataset, xr.DataArray],
        target_data: xr.DataArray,
        desc: str = "Creating Samples",
    ) -> None:
        lookup: List[Tuple[str, int]] = []
        spatial_units_without_samples: List[Union[str, int]] = []
        self.x_d: Dict[str, np.ndarray] = {}
        self.y: Dict[str, np.ndarray] = {}
        self.times: List[float] = []

        # spatial_unit = target_data[self.basin_dim].values[0]
        pbar = tqdm(target_data.station_id.values, desc=desc)

        #  iterate over each basin
        for spatial_unit in pbar:
            #  create pd.Dataframe timeseries from [xr.Dataset, xr.DataArray]
            in_df = input_data.sel({self.basin_dim: spatial_unit}).to_dataframe()
            out_df = target_data.sel({self.basin_dim: spatial_unit}).to_dataframe()

            #  create np.ndarray
            _x_d = in_df[self.input_variables].values
            _y = out_df[self.target_variable].values

            #  keep pointer to the valid samples
            flag = validate(x_d=_x_d, y=_y, seq_length=self.seq_length)
            valid_samples = np.argwhere(flag == 1)
            [lookup.append((spatial_unit, smp)) for smp in valid_samples]

            # STORE DATA if spatial_unit has at least ONE valid sample
            if valid_samples.size > 0:
                self.x_d[spatial_unit] = _x_d.astype(np.float32)
                self.y[spatial_unit] = _y.astype(np.float32)
            else:
                spatial_units_without_samples.append(spatial_unit)

            if self.times == []:
                #  store times as float32 to keep pytorch happy
                # assert False
                self.times = (
                    in_df.index.values.astype(np.float32)
                    # .astype(np.float32)
                )

        #  save lookup from INT: (spatial_unit, index) for valid samples
        self.lookup_table: Dict[int, Tuple[str, int]] = {
            i: elem for i, elem in enumerate(lookup)
        }
        self.num_samples = len(self.lookup_table)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        spatial_unit, target_ix = self.lookup_table[idx]
        # X, y samples
        # [seq_length, input_features]
        x_d = self.x_d[spatial_unit][int(target_ix - self.seq_length) : int(target_ix)]
        # [seq_length, 1]
        y = np.expand_dims(
            self.y[spatial_unit][int(target_ix - self.seq_length) : int(target_ix)], -1
        )

        #  to torch.Tensor
        x_d = Tensor(x_d)
        y = Tensor(y)

        #  metadata
        time = self.times[int(target_ix - self.seq_length) : int(target_ix)]
        meta = dict(spatial_unit=spatial_unit, time=time)

        data = dict(x_d=x_d, y=y, meta=meta)

        return data


def get_train_test_dataloader(
    input_data: xr.Dataset,
    target_data: xr.Dataset,
    target_variable: str,
    input_variables: List[str],
    seq_length: int = 64,
    basin_dim: str = "station_id",
    time_dim: str = "time",
    batch_size: int = 256,
    train_start_date: pd.Timestamp = pd.to_datetime("01-01-1998"),
    train_end_date: pd.Timestamp = pd.to_datetime("12-31-2006"),
    test_start_date: pd.Timestamp = pd.to_datetime("01-01-2007"),
    test_end_date: pd.Timestamp = pd.to_datetime("01-01-2009"),
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader]:
    train_dataset = TimeSeriesDataset(
        input_data=input_data.sel(time=slice(train_start_date, train_end_date)),
        target_data=target_data.sel(time=slice(train_start_date, train_end_date)),
        target_variable=target_variable,
        input_variables=input_variables,
        seq_length=seq_length,
        basin_dim=basin_dim,
        time_dim=time_dim,
        desc="Creating Train Samples",
    )
    test_dataset = TimeSeriesDataset(
        input_data=input_data.sel(time=slice(test_start_date, test_end_date)),
        target_data=target_data.sel(time=slice(test_start_date, test_end_date)),
        target_variable=target_variable,
        input_variables=input_variables,
        seq_length=seq_length,
        basin_dim=basin_dim,
        time_dim=time_dim,
        desc="Creating Test Samples",
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader


if __name__ == "__main__":
    #  load data
    from pathlib import Path

    data_dir = Path("/datadrive/data")
    target_data = xr.open_dataset(data_dir / "SOIL_MOISTURE/interpolated_esa_cci_sm.nc")
    input_data = xr.open_dataset(
        data_dir / "SOIL_MOISTURE/interpolated_normalised_camels_gb.nc"
    )

    # #  initialize dataset
    # td = TimeSeriesDataset(
    #     input_data=input_data,
    #     target_data=target_data,
    #     target_variable="sm",
    #     input_variables=["precipitation"],
    #     seq_length=64,
    #     basin_dim="station_id",
    #     time_dim="time",
    # )

    # initialize dataloaders]
    batch_size = 256
    seq_length = 64
    num_workers = 4
    input_variables = ["precipitation"]

    train_dl, test_dl = get_train_test_dataloader(
        input_data=input_data,
        target_data=target_data,
        target_variable="sm",
        input_variables=input_variables,
        seq_length=seq_length,
        basin_dim="station_id",
        time_dim="time",
        batch_size=batch_size,
        num_workers=num_workers,
    )

    data = train_dl.__iter__().__next__()
    assert data["x_d"].shape == (batch_size, seq_length, len(input_variables))

    times = data["meta"]["time"].numpy().astype("datetime64[ns]")
    assert False
