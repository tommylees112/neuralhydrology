from typing import List, Tuple, Optional
from collections import defaultdict
import numpy as np
import xarray as xr
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm 

import sys
sys.path.append("/home/tommy/neuralhydrology")
from neuralhydrology.utils.config import Config
from scripts.cell_state.cell_state_dataset import CellStateDataset


def get_model_weights(model: nn.Linear) -> Tuple[np.ndarray]:
    parameters = list(model.parameters())
    w = parameters[0].cpu().detach().numpy()
    b = parameters[1].cpu().detach().numpy()
    return w, b


def get_all_models_weights(models: List[nn.Linear]) -> Tuple[np.ndarray]:
    model_outputs = defaultdict(dict)
    for sw_ix in range(len(models)):
        w, b = get_model_weights(models[sw_ix])
        model_outputs[f"swvl{sw_ix+1}"]["w"] = w
        model_outputs[f"swvl{sw_ix+1}"]["b"] = b

    ws = np.stack([model_outputs[swl]["w"] for swl in model_outputs.keys()]).reshape(
        len(models), 64
    )
    bs = np.stack([model_outputs[swl]["b"] for swl in model_outputs.keys()])
    return ws, bs


def calculate_raw_correlations(
    norm_sm: xr.Dataset, cs_data: xr.Dataset, config: Config, variable_str: Optional[str] = "cell_state"
) -> np.ndarray:
    """Calculate the correlation coefficient for each feature of cs_data
    using: `np.corrcoef`.

    Args:
        norm_sm (xr.Dataset): The target soil moisture data (1D)
        cs_data (xr.Dataset): The input cell state data (64D)
        variable_str (Optional[str]): (defaults to "cell_state")

    Returns:
        np.ndarray: (4, 64) correlation coefficient for each feature (64),
         for each soil water level (4).
    """
    #  Create the datasets for each feature
    datasets = []

    # features can be soil level or just one soil moisture estimate
    features = list(norm_sm.data_vars) if len(list(norm_sm.data_vars)) > 1 else [variable_str]
    for feature in features:
        # target data = SOIL MOISTURE
        target_data = norm_sm[feature]
        target_data["station_id"] = [int(sid) for sid in target_data["station_id"]]

        #  input data
        input_data = cs_data
        input_data["station_id"] = [int(sid) for sid in input_data["station_id"]]

        sm_dataset = CellStateDataset(
            input_data=input_data, target_data=target_data, config=config,
        )
        datasets.append(sm_dataset)

    # Calculate the correlations for each
    all_correlations = np.zeros((len(features), 64))

    pbar = tqdm(np.arange(len(datasets)), desc="Calculating Correlation")
    for feature in pbar:
        pbar.set_postfix_str(feature)
        
        #  get the DATA for that feature
        all_cs_data = np.array(
            [x.detach().cpu().numpy() for (_, (x, _)) in DataLoader(datasets[feature])]
        )
        all_sm_data = np.array(
            [y.detach().cpu().numpy() for (_, (_, y)) in DataLoader(datasets[feature])]
        )
        Y = all_sm_data.reshape(-1, 1)

        correlations = []
        for cs in np.arange(all_cs_data.shape[-1]):
            X = all_cs_data[:, :, cs]
            correlations.append(np.corrcoef(X, Y, rowvar=False)[0, 1])

        #  save correlations to matrix (n_features, 64)
        correlations = np.array(correlations)
        all_correlations[feature, :] += correlations

    return all_correlations
