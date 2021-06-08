import torch
from pathlib import Path
import numpy as np
import xarray as xr
import pandas as pd
import pickle
import random
from tqdm import tqdm
from typing import Optional, List

import sys

sys.path.append("/home/tommy/neuralhydrology")
from neuralhydrology.utils.config import Config
from neuralhydrology.datasetzoo.basedataset import BaseDataset
from neuralhydrology.datasetzoo import CamelsGB, PixelDataset
from neuralhydrology.datautils.utils import load_basin_file
from scripts.read_model import get_model, _load_weights
from scripts.read_nh_results import (
    get_test_filepath,
    get_all_station_ds,
)


def _generate_path_input(baseline, x_d, device, m):
    """
    Generate m inputs (k = 1, ..., m) of seq_length, n_features size
     which allow us to approximate the integral from the first (baseline where alpha=1)
     to the last (m) where alpha=0.
    """
    #  [batch_size, seq_length, n_features] -> [seq_length, batch_size, n_features]
    #  put seq length first
    xi = torch.zeros((m, x_d.shape[0], x_d.shape[1]), requires_grad=True).to(device)

    for k in range(m):
        xi[k, :, :] = baseline + k / (m - 1) * (x_d - baseline)
    return xi


def integrated_gradients(
    model: torch.nn.Module,
    data: BaseDataset,
    device: torch.device,
    baseline: torch.Tensor = None,
    m: int = 300,
):
    """"baseline input: we usually used a vector of zeros, which however is debatable. 
     It should be an input that results in a neural output. 
     It is only really defined for conv nets and image data, 
     where they usually use a black image as baseline input. 
     However, so far the results were more or less pretty well 
     with a sequence of zeros for us as well
    """
    # m is the number of steps in approximating the integral
    model.zero_grad()

    x_d, x_s = data["x_d"], data["x_s"]
    x_d, x_s = x_d.to(device), x_s.to(device)

    if baseline is None:
        baseline = x_d.new_zeros(x_d.shape)
    else:
        assert baseline.size() == x_d.size(), "Tensor sizes don't match"

    # 'lstm_output', 'h_n', 'c_n', 'y_hat'
    #  update the 'x_d' with the path input (generated with m steps)
    xi = _generate_path_input(baseline, x_d, device, m)
    xi = torch.autograd.Variable(xi, requires_grad=True).to(device)
    data["x_d"] = xi

    #  update the 'x_s' with the static data repeated m times
    x_s = x_s.unsqueeze(0).repeat(m, 1)
    data["x_s"] = x_s

    output = model(data)
    yhat = output["y_hat"]
    h_n, c_n = output["h_n"], output["c_n"]

    #  out.shape == [m, seq_length, 1]
    #   e.g. torch.Size([300, 365, 1])
    #  pass in ones to the backward function = set the dL/dy_hat 1 and continue with dy_hat/dW
    yhat[:, -1, :].backward(yhat.new_ones(m, 1))

    # igrad.shape == [seq_length, n_features]
    #   e.g. torch.Size([365, 3])
    igrad = xi.grad.sum(0) * (x_d - baseline) / m

    # difference between baseline and target input
    #   scalar, quality.shape == ()
    quality = (yhat[0, -1, 0] - yhat[-1, -1, 0]).detach().cpu().numpy()

    return igrad.detach().cpu().numpy(), quality, yhat[0, -1, 0].detach().cpu().numpy()


def main(cfg: Config, basins: Optional[List[str]] = None):
    #   GET the scaler
    with open(cfg.run_dir / "train_data/train_data_scaler.p", "rb") as fp:
        scaler = pickle.load(fp)

    #  GET the model
    model = get_model(cfg).to(cfg.device)
    _load_weights(model, cfg)

    #  GET the test period
    test_date_range = pd.date_range(cfg.test_start_date, cfg.test_end_date)
    n_time_steps = len(test_date_range)

    # GET the list of training basins = basin_file
    if cfg.dataset == "pixel":
        assert basins is not None
        # basins = [sid for sid in preds.station_id.values]
    else:
        basins = load_basin_file(cfg.train_basin_file)

    # intialise baseline
    baseline = None
    num_samples = 300

    #  initialise device
    device = cfg.device

    random.shuffle(basins)
    # pbar = tqdm(basins, desc="Calculating integrated gradients:")
    for basin in basins:
        #     pbar.set_postfix_str(basin)
        # output file
        outfile = run_dir / f"integrated_gradients/seq2one_{basin}.p"

        if outfile.is_file():
            continue

        # if not exist, create an empty dummy file to block other instances to process this basin
        outfile.parent.mkdir(exist_ok=True) if not outfile.parent.exists() else None
        outfile.parent.mkdir(exist_ok=True)
        with outfile.open("wb") as fp:
            pickle.dump(["hello world"], fp)

        # init storage for local gradients and simulated streamflow
        ysim = np.full([n_time_steps], np.nan)
        grads = np.full([n_time_steps, 365, len(cfg.dynamic_inputs)], np.nan)

        if cfg.dataset == "pixel":
            dset = PixelDataset(
                cfg=cfg, is_train=False, basin=basin, period="test", scaler=scaler
            )
        elif cfg.dataset == "camelsgb":
            dset = CamelsGB(
                cfg=cfg, is_train=False, basin=basin, period="test", scaler=scaler
            )

        # calcuate gradients
        pbar = tqdm(range(n_time_steps), desc=f"Basin {basin} integrated gradients")
        for i in pbar:
            # extract data for this timestep
            data = dset.__getitem__(i)

            # convert to device
            for key in data:
                data[key] = data[key].to(device)

            x_d, x_s, y = data["x_d"], data["x_s"], data["y"]
            x_d, x_s = x_d.to(device), x_s.to(device)

            # set up baseline for integrated gradients (only needs to be done once, but after we have a dataset
            if baseline is None:
                #  make zeros
                #  baseline = x_d.new_zeros(x_d.shape).to(device)

                #  turn into true zero (due to normalization)
                #  TODO: make this dynamic # for features in [f for f in scaler. if f not in cfg.target_variables]
                pet_zero = (
                    torch.ones(365, 1)
                    * (
                        -scaler["xarray_feature_center"]["pet"]
                        / scaler["xarray_feature_scale"]["pet"]
                    ).values
                )
                precip_zero = (
                    torch.ones(365, 1)
                    * (
                        -scaler["xarray_feature_center"]["precipitation"]
                        / scaler["xarray_feature_scale"]["precipitation"]
                    ).values
                )
                temp_zero = (
                    torch.ones(365, 1)
                    * (
                        -scaler["xarray_feature_center"]["temperature"]
                        / scaler["xarray_feature_scale"]["temperature"]
                    ).values
                )
                baseline = torch.hstack([pet_zero, precip_zero, temp_zero]).to(device)

            # calculate gradients for this timestep
            gradients, quality, ysim_estimate = integrated_gradients(
                model,
                data=data,
                device=device,
                m=300,  #   num_steps_to_approximate_integral
                baseline=baseline,
            )
            grads[i, :, :], _, ysim[i] = gradients, quality, ysim_estimate

        # save results
        with outfile.open("wb") as f:
            pickle.dump([grads, ysim], f)


def load_np_gradients(fp: Path):
    arr = pickle.load(fp.open("rb"))[0]
    return arr


def get_station_from_name(fp: Path) -> str:
    return fp.stem.split("_")[-1]


def create_xarray_gradients_obj(
    cfg: Config,
    grads_stn: np.ndarray,
    basin: str,
    test_date_range: List[pd.Timestamp],
    var_list: List[str] = ["peti", "precip", "temp"],
) -> xr.Dataset:
    # create xarray object
    gradients = xr.Dataset(
        {
            var_list[i]: (
                ["station_id", "target_time", "seq_length"],
                grads_stn[:, :, :, i],
            )
            for i in range(3)
        },
        coords={
            "station_id": [basin],
            "target_time": test_date_range,
            "seq_length": np.arange(cfg.seq_length)[::-1],
        },
    )
    return gradients


def create_gradient_xarray(cfg: Config, run_dir: Path) -> xr.Dataset:
    #  LOAD integrated gradient pickle files
    ig_dir = run_dir / "integrated_gradients"
    np_pkl = [p for p in ig_dir.glob("*.p")]

    test_date_range = pd.date_range(cfg.test_start_date, cfg.test_end_date)

    # READ into xarray format
    var_list = (
        ["pet", "precipitation", "temperature"]
        if cfg.dataset == "pixel"
        else ["peti", "precipitation", "temperature"]
    )
    all_gradients = []
    pbar = tqdm(np_pkl, desc="Creating Gradient Xarray")
    for fp in pbar:
        grads = load_np_gradients(fp)
        #  ensure 3D array
        if not isinstance(grads, np.ndarray):  #  len(grads.shape) != 3:
            continue
        station_id = get_station_from_name(fp)
        pbar.set_postfix_str(station_id)
        grads_stn = np.expand_dims(grads, 0)
        gradients = create_xarray_gradients_obj(
            grads_stn=grads_stn,
            cfg=cfg,
            basin=station_id,
            var_list=var_list,
            test_date_range=test_date_range,
        )
        all_gradients.append(gradients)

    gradients = xr.concat(all_gradients, dim="station_id")
    return gradients


if __name__ == "__main__":
    run_dir = Path(
        "/datadrive/data/runs/azure_nh_runoff_30EPOCH_PIXEL_ALL_STATIC_0406_093025"
    )

    if not (run_dir / "integrated_gradients/gradients.nc").exists():
        # GET the config
        cfg = Config(run_dir / "config.yml")
        cfg.run_dir = run_dir

        if cfg.dataset == "pixel":
            res_fp = get_test_filepath(run_dir, epoch=30)
            preds = get_all_station_ds(res_fp)
            pixels = preds.station_id.values

        # Run and save integrated gradients
        main(cfg, basins=pixels)

        # create gradients file
        gradients = create_gradient_xarray(cfg, run_dir)
        gradients.to_netcdf(run_dir / "integrated_gradients/gradients.nc")
    else:
        gradients = xr.open_dataset(run_dir / "integrated_gradients/gradients.nc")