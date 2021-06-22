from typing import Dict, Optional, List, Tuple
from pathlib import Path
import numpy as np
import xarray as xr
import pandas as pd
from sklearn.linear_model import SGDRegressor
from torch.utils.data import Dataset

import sys 
sys.path.append("/home/tommy/neuralhydrology")
from scripts.read_nh_results import calculate_all_error_metrics
from scripts.cell_state.timeseries_model import _round_time_to_hour
from scripts.cell_state.timeseries_dataset import create_train_test_datasets
from scripts.cell_state.timeseries_dataset import get_time_basin_aligned_dictionary

def init_sklearn_model(kwargs: Optional[Dict] = None):
    if kwargs is None:
        kwargs = dict(
        loss="huber",
        penalty="elasticnet",
        alpha=0.01,
        l1_ratio=0.15,            # default
        fit_intercept=True, 
        n_iter_no_change=10,
        early_stopping=True,
        verbose=1
    )

    model = SGDRegressor(
        **kwargs,
    )

    return model


def make_predictions(model, data: Dict[str, np.ndarray]) -> pd.DataFrame:
    y_hat = model.predict(data["X"])
    preds = pd.DataFrame({
        "station_id": data["station_ids"].ravel(),
        "time": data["times"].astype("datetime64[ns]").ravel(),
        "obs": data["y"].ravel(),
        "sim": y_hat.ravel(),
    }).set_index(["station_id", "time"]).to_xarray()
    return preds


def create_analysis_dataset(data: Dict[str, np.ndarray]) -> xr.Dataset:
    _x_df = pd.DataFrame(
        {f"dim{i}": data["X"][:, i] for i in range(data["X"].shape[-1])}
    )
    _df = pd.DataFrame({
        "station_id": data["station_ids"].ravel(), "time": data["times"].astype("datetime64[ns]").ravel(), "y": data["y"].ravel(), 
    })
    analysis_df = _df.join(_x_df).set_index(["time", "station_id"])
    analysis_ds = analysis_df.to_xarray()
    analysis_ds["time"] = _round_time_to_hour(analysis_ds["time"].values)

    return analysis_ds


def evaluate(
    model, 
    data: Dict[str, np.ndarray], 
    basin_coord: str = "station_id",
    time_coord: str = "time",
    obs_var: str = "obs",
    sim_var: str = "sim",
    metrics: List[str] = ["NSE", "Pearson-r"],
) -> Tuple[pd.DataFrame, xr.Dataset]:
    preds = make_predictions(model, data)

    errors = calculate_all_error_metrics(
        preds,
        basin_coord=basin_coord,
        time_coord=time_coord,
        obs_var=obs_var,
        sim_var=sim_var,
        metrics=metrics,
    )

    return preds, errors


if __name__ == "__main__":
    data_dir = Path("/datadrive/data")
    # initialise the train-test split
    train_start_date = pd.to_datetime("1998-01-01")
    train_end_date = pd.to_datetime("2006-09-30")
    test_start_date = pd.to_datetime("2006-10-01")
    test_end_date =  pd.to_datetime("2009-10-01")

    input_variables = [f"dim{i}" for i in np.arange(64)]
    target_var = "sm"
    subset_pixels = None

    # get target data `era5_sm`
    # era5_sm = xr.open_dataset(data_dir / "SOIL_MOISTURE/FINAL/era5land_normalized.nc")
    esa_ds = xr.open_dataset(data_dir / "SOIL_MOISTURE/FINAL/esa_ds_interpolated_normalised.nc")

    # get input data `cs`
    cs = xr.open_dataset(data_dir / "SOIL_MOISTURE/FINAL/cs_normalised_64variables.nc")

    # create train-test dataloaders and Dict of X, y arrays
    train_dataset, test_dataset = create_train_test_datasets(
        target_var=target_var,
        input_variables=input_variables,
        target_ds=esa_ds,
        input_ds=cs,
        train_start_date=train_start_date,
        train_end_date=train_end_date,
        test_start_date=test_start_date,
        test_end_date=test_end_date,
        subset_pixels=subset_pixels,
        seq_length=1,
        basin_dim="station_id",
        time_dim="time",
    )

    train = get_time_basin_aligned_dictionary(train_dataset)
    test = get_time_basin_aligned_dictionary(test_dataset)

    # initalise the model 
    model = init_sklearn_model()
    # fit the model 
    model.fit(train["X"], train["y"].ravel())
    
    # make predictions from the fitted model 
    preds, errors = evaluate(model, test)
    train_preds, train_errors = evaluate(model, train)

    # create an easy to work with analysis dataset
    test_analysis = create_analysis_dataset(test)
    train_analysis = create_analysis_dataset(train)
    analysis_ds = xr.concat([train_analysis, test_analysis], dim="time")

    assert False