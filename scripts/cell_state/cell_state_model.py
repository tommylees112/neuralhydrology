from typing import List, Tuple, Any, Dict, Union, Optional
from tqdm import tqdm
import numpy as np
import xarray as xr
import torch
from torch import Tensor
from torch import nn
from torch.utils.data import DataLoader
from collections import defaultdict
import pandas as pd

import sys

sys.path.append("/home/tommy/neuralhydrology")
from neuralhydrology.modelzoo.basemodel import BaseModel
from scripts.cell_state.cell_state_dataset import (
    CellStateDataset,
    get_train_test_dataset,
    train_validation_split,
)
from scripts.cell_state.analysis import (
    get_all_models_weights,
    calculate_raw_correlations,
)
from neuralhydrology.modelzoo.head import get_head
from neuralhydrology.modelzoo.basemodel import BaseModel
from neuralhydrology.utils.config import Config


class LinearModel(nn.Module):
    def __init__(self, D_in: int, dropout: float = 0.0, **kwargs):
        super(LinearModel, self).__init__(**kwargs)

        #  number of weights == number of dimensions in cell state vector (cfg.hidden_size)
        self.D_in = D_in
        self.model = torch.nn.Sequential(torch.nn.Linear(self.D_in, 1))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, data: Dict[str, torch.Tensor]):
        return self.model(self.dropout(data["x_d"].squeeze()))


def train_model(
    model: nn.Module,
    train_dataset: CellStateDataset,
    learning_rate: float = 1e-2,
    n_epochs: int = 5,
    l2_penalty: float = 0,
    val_split: bool = False,
    desc: str = "Training Epoch",
) -> Tuple[Any, List[float], List[float]]:
    #  GET loss function
    loss_fn = torch.nn.MSELoss(reduction="sum")

    # GET optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=l2_penalty
    )

    #  TRAIN
    train_losses_ALL = []
    val_losses_ALL = []
    for epoch in tqdm(range(n_epochs), desc=desc):
        train_losses = []
        val_losses = []

        #  new train-validation split each epoch
        if val_split:
            #  create a unique test, val set (random) for each ...
            train_sampler, val_sampler = train_validation_split(train_dataset)
            train_loader = DataLoader(
                train_dataset, batch_size=256, sampler=train_sampler
            )
            val_loader = DataLoader(train_dataset, batch_size=256, sampler=val_sampler)
        else:
            train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

        for data in train_loader:
            y_pred = model(X)
            loss = loss_fn(y_pred, y)

            # train/update the weight
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.detach().cpu().numpy())

        # VALIDATE
        if val_split:
            model.eval()
            with torch.no_grad():
                for (basin, time), data in val_loader:
                    X, y = data
                    y_pred = model(X)
                    loss = loss_fn(y_pred, y)
                    val_losses.append(loss.detach().cpu().numpy())

            #  save the epoch-mean losses
            val_losses_ALL = np.mean(val_losses)

        train_losses_ALL.append(np.mean(train_losses))

    return model, train_losses_ALL, val_losses_ALL


#  ALL Training Process
def train_model_loop(
    config: Config,
    input_data: xr.Dataset,
    target_data: xr.DataArray,
    train_test: bool = True,
    train_val: bool = False,
    return_loaders: bool = True,
    desc: str = "Training Epoch",
    dropout: float = 0.0,
    device: str = "cpu",
    l2_penalty: float = 0,
) -> Tuple[List[float], BaseModel, Optional[Tuple[DataLoader]]]:
    #  1. create dataset (input, target)
    dataset = CellStateDataset(
        input_data=input_data, target_data=target_data, device=device,
    )

    #  2. create train-test split
    if train_test:
        #  build the train, test, validation
        train_dataset, test_dataset = get_train_test_dataset(dataset)
        test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    else:
        train_dataset = dataset
        test_dataset = dataset
        test_loader = DataLoader(dataset, batch_size=256, shuffle=False)

    #  3. initialise the model
    model = LinearModel(D_in=dataset.dimensions, dropout=dropout)
    model = model.to(device)

    # 4. Run training loop (iterate over batches)
    model, train_losses, _ = train_model(
        model,
        train_dataset,
        learning_rate=1e-3,
        n_epochs=20,
        weight_decay=0,
        val_split=train_val,
        desc=desc,
        l2_penalty=l2_penalty,
    )

    # 5. Save outputs (losses: List[float], model: BaseModel, dataloader: DataLoader)
    if return_loaders:
        return train_losses, model, test_loader
    else:
        return train_losses, model, None


def to_xarray(predictions: Dict[str, List]) -> xr.Dataset:
    return pd.DataFrame(predictions).set_index(["time", "station_id"]).to_xarray()


def calculate_predictions(model: BaseModel, loader: DataLoader) -> xr.Dataset:
    predictions = defaultdict(list)

    model.eval()
    with torch.no_grad():
        for data in loader:
            X, y = data["x_d"], data["y"]
            basin, time = data["meta"]["basin"], data["meta"]["time"]
            y_hat = model(data)

            #  Coords / Dimensions
            predictions["time"].extend(pd.to_datetime(time))
            predictions["station_id"].extend(basin)

            # Variables
            predictions["y_hat"].extend(y_hat.detach().cpu().numpy().flatten())
            predictions["y"].extend(y.detach().cpu().numpy().flatten())

    return to_xarray(predictions)


if __name__ == "__main__":
    from pathlib import Path

    data_dir = Path("/datadrive/data")
    run_dir = data_dir / "runs/complexity_AZURE/hs_064_0306_205514"

    # load in config
    cfg = Config(run_dir / "config.yml")
    cfg.run_dir = run_dir

    #  load in input data
    input_data = xr.open_dataset(data_dir / "SOIL_MOISTURE/norm_cs_data.nc")
    #  load in target data (Soil Moisture)
    target_data = xr.open_dataset(data_dir / "SOIL_MOISTURE/interpolated_esa_cci_sm.nc")

    losses_list = []
    models = []
    test_loaders = []

    train_test = True
    train_val = False

    data_vars = [v for v in target_data.data_vars]
    target_features = data_vars if len(data_vars) > 1 else ["sm"]
    for feature in target_features:
        train_losses, model, test_loader = train_model_loop(
            config=cfg,
            input_data=input_data,
            target_data=target_data[feature],  #  needs to be xr.DataArray
            train_test=train_test,
            train_val=train_val,
            return_loaders=True,
        )

        # store outputs of training process
        losses_list.append(train_losses)
        models.append(model)
        test_loaders.append(test_loader)

    #  run forward pass and convert to xarray object
    preds = calculate_predictions(model, test_loader)

    # extract weights and biases
    print("-- Extracting weights and biases --")
    ws, bs = get_all_models_weights(models)

    #  calculate raw correlations (cell state and values)
    print("-- Running RAW Correlations --")
    all_corrs = calculate_raw_correlations(target_data, input_data, config=cfg)
