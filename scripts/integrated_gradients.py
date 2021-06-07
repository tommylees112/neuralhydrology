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
from scripts.read_model import (get_model, _load_weights)



def _generate_path_input(baseline, x_d, device, m):
    """
    Generate m inputs (k = 1, ..., m) of seq_length, n_features size
     which allow us to approximate the integral from the first (baseline where alpha=1)
     to the last (m) where alpha=0.
    """
    # [batch_size, seq_length, n_features] -> [seq_length, batch_size, n_features]
    # put seq length first
    xi = torch.zeros((m, x_d.shape[0], x_d.shape[1]), requires_grad=True).to(device)
    
    for k in range(m):
        xi[k, :, :] = baseline + k / (m - 1) * (x_d - baseline)
    return xi


def integrated_gradients(model: torch.nn.Module,
                         data: BaseDataset,
                         device: torch.device,
                         baseline: torch.Tensor = None,
                         m: int = 300):
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
    # update the 'x_d' with the path input (generated with m steps)
    xi = _generate_path_input(baseline, x_d, device, m)
    xi = torch.autograd.Variable(xi, requires_grad=True).to(device)
    data["x_d"] = xi
    
    # update the 'x_s' with the static data repeated m times
    x_s = x_s.unsqueeze(0).repeat(m, 1)
    data["x_s"] = x_s
        
    output = model(data)
    yhat = output["y_hat"]
    h_n, c_n = output["h_n"], output["c_n"]
    
    # out.shape == [m, seq_length, 1]
    #  e.g. torch.Size([300, 365, 1])
    # pass in ones to the backward function = set the dL/dy_hat 1 and continue with dy_hat/dW
    yhat[:, -1, :].backward(yhat.new_ones(m, 1))
    
    # igrad.shape == [seq_length, n_features]
    #  e.g. torch.Size([365, 3])
    igrad = xi.grad.sum(0) * (x_d - baseline) / m

    # difference between baseline and target input
    #  scalar, quality.shape == ()
    quality = (yhat[0, -1, 0] - yhat[-1, -1, 0]).detach().cpu().numpy()

    return igrad.detach().cpu().numpy(), quality, yhat[0, -1, 0].detach().cpu().numpy()


def main(cfg: Config, basins: Optional[List[str]] = None):
    #  GET the scaler
    with open(cfg.run_dir / "train_data/train_data_scaler.p", "rb") as fp:
        scaler = pickle.load(fp)

    #  GET the model
    model = get_model(cfg).to(cfg.device)
    _load_weights(model, cfg)

    # GET the test period 
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

    # initialise device
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
                cfg=cfg,
                is_train=False,
                basin=basin,
                period="test",
                scaler=scaler
            )
        elif cfg.dataset == "camelsgb":
            dset = CamelsGB(
                cfg=cfg,
                is_train=False,
                basin=basin,
                period="test",
                scaler=scaler
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
                # make zeros
                # baseline = x_d.new_zeros(x_d.shape).to(device) 
                
                # turn into true zero (due to normalization)
                # TODO: make this dynamic # for features in [f for f in scaler. if f not in cfg.target_variables]
                pet_zero = torch.ones(365, 1) * (-scaler["xarray_feature_center"]["pet"] / scaler["xarray_feature_scale"]["pet"]).values
                precip_zero = torch.ones(365, 1) * (-scaler["xarray_feature_center"]["precipitation"] / scaler["xarray_feature_scale"]["precipitation"]).values
                temp_zero = torch.ones(365, 1) * (-scaler["xarray_feature_center"]["temperature"] / scaler["xarray_feature_scale"]["temperature"]).values
                baseline = torch.hstack([pet_zero, precip_zero, temp_zero]).to(device)
                
            # calculate gradients for this timestep
            gradients, quality, ysim_estimate = integrated_gradients(model,
                                                            data=data,
                                                            device=device,
                                                            m=300,  #  num_steps_to_approximate_integral
                                                            baseline=baseline)
            grads[i, :, :], _, ysim[i] = gradients, quality, ysim_estimate

        # save results
        with outfile.open('wb') as f:
            pickle.dump([grads, ysim], f)


if __name__ == "__main__":
    run_dir = Path("/datadrive/data/runs/azure_nh_runoff_30EPOCH_PIXEL_ALL_STATIC_0406_093025")

    # GET the config
    cfg = Config(run_dir / "config.yml")
    cfg.run_dir = run_dir

    if cfg.dataset == "pixel":
        pixels = pd.read_csv("data/camels_gb_pixel_list.txt")

    # GET the model
    main(cfg, basins=pixels)