from typing import Optional
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
import xarray as xr
import pandas as pd
import numpy as np
from collections import defaultdict

from scripts.cell_state.timeseries_dataset import get_train_test_dataloader
from scripts.cell_state.cell_state_model import LinearModel
from scripts.cell_state.cell_state_model import to_xarray


def _train_epoch(
    dataloader: DataLoader,
    model: LinearModel,
    loss_fn: torch.nn.MSELoss,
    optimizer: torch.optim.Adam,
    epoch: Optional[int] = None,
    device: str = "cpu",
) -> np.ndarray:
    desc = f"Training Model Epoch {epoch}" if epoch is not None else "Train Epoch"
    pbar = tqdm(dataloader, desc=desc)
    _losses = []
    for data in pbar:
        for key in [k for k in data.keys() if k != "meta"]:
            data[key] = data[key].to(device)

        y_pred = model(data).squeeze()
        y = data["y"][:, -1, :].squeeze()
        loss = loss_fn(y_pred, y)

        # train/update the weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _loss = loss.detach().cpu().numpy()
        _losses.append(_loss)
        pbar.set_postfix_str(_loss)
    
    mean_loss = np.mean(_losses)
    pbar.set_postfix_str(f"{desc} Loss: {mean_loss}")

    return mean_loss


def train(
    model: LinearModel,
    dataloader: DataLoader,
    learning_rate: float = 1e-2,
    l2_penalty: float = 1,
    device: str = "cpu",
    n_epochs: int = 3,
) -> np.ndarray:
    # GET loss function
    loss_fn = torch.nn.MSELoss(reduction="sum").to(device)

    # GET optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=l2_penalty
    )

    #  train the model
    losses = []
    for epoch in range(n_epochs):
        _epoch_loss = _train_epoch(
            model=model,
            dataloader=dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            epoch=epoch,
            device=device,
        )
        losses.append(_epoch_loss)

    return losses


def predict(model: LinearModel, dataloader: DataLoader, device: str = "cpu") -> xr.Dataset:
    predictions = defaultdict(list)
    model.eval()
    with torch.no_grad():
        for data in tqdm(dataloader, desc="Forward Pass"):
            for key in [k for k in data.keys() if k != "meta"]:
                data[key] = data[key].to(device)

            y_hat = model(data).squeeze()
            y = data["y"][:, -1, :].squeeze()

            _times = pd.to_datetime(
                data["meta"]["time"].numpy().astype("datetime64[ns]")[:, -1]
            )
            # conversion of float to time is weird. Reset to start of day
            time = [t.replace(hour=0, minute=0, second=0, microsecond=0, nanosecond=0) for t in _times]
            spatial_unit = data["meta"]["spatial_unit"].numpy()

            #  Coords / Dimensions
            predictions["time"].extend((time))
            predictions["station_id"].extend(spatial_unit)

            # Variables
            predictions["y_hat"].extend(y_hat.detach().cpu().numpy().flatten())
            predictions["y"].extend(y.detach().cpu().numpy().flatten())

    return to_xarray(predictions)


if __name__ == "__main__":
    from scripts.read_nh_results import calculate_all_error_metrics

    data_dir = Path("/datadrive/data")
    device = "cuda:0"
    subset = True

    # GET data
    target_data = xr.open_dataset(data_dir / "SOIL_MOISTURE/interpolated_esa_cci_sm.nc")
    input_data = xr.open_dataset(
        data_dir / "SOIL_MOISTURE/interpolated_normalised_camels_gb.nc"
    )
    
    if subset:
        pixels = np.random.choice(target_data.station_id.values, 5)
        target_data = target_data.sel(station_id=pixels)
        input_data = input_data.sel(station_id=pixels)

    #  GET dataloader
    batch_size = 256
    num_workers = 4
    train_dl, test_dl = get_train_test_dataloader(
        input_data=input_data,
        target_data=target_data,
        target_variable="sm",
        input_variables=["precipitation"],
        seq_length=64,
        basin_dim="station_id",
        time_dim="time",
        batch_size=batch_size,
        train_start_date=pd.to_datetime("01-01-1998"),
        train_end_date=pd.to_datetime("12-31-2006"),
        test_start_date=pd.to_datetime("01-01-2007"),
        test_end_date=pd.to_datetime("01-01-2009"),
        num_workers=num_workers,
    )
    data = train_dl.__iter__().__next__()
    x, y = data["x_d"], data["y"]
    assert not (x == y).all(), "Data Leakage"

    # GET the model
    #  np.product(data["x_d"].shape[1:]) if data["x_d"].shape[0] == batch_size else np.product(data["x_d"].shape[1:])
    D_in = 64
    model = LinearModel(D_in=D_in)
    model.to(device)

    # TRAIN the model
    learning_rate = 1e-2
    l2_penalty = 1
    n_epochs = 1

    losses = train(
        model=model,
        dataloader=train_dl,
        learning_rate=learning_rate,
        l2_penalty=l2_penalty,
        device=device,
        n_epochs=n_epochs,
    )

    #  PREDICT
    preds = predict(model=model, dataloader=test_dl, device=device)

    #  EVALUATE with error metrics
    errors = calculate_all_error_metrics(
        preds,
        basin_coord="station_id",
        time_coord="time",
        obs_var="y",
        sim_var="y_hat",
        metrics=["NSE", "Pearson-r"],
    )

    print(errors)

    assert False
