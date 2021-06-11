import xarray as xr
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
import argparse
from typing import Tuple, Dict, Optional, DefaultDict, Union, List
from collections import defaultdict
from tqdm import tqdm
import re
import pickle
import pprint

from neuralhydrology.evaluation import RegressionTester as Tester
from neuralhydrology.utils.config import Config
from neuralhydrology.evaluation.metrics import (
    calculate_all_metrics,
    AllNaNError,
    calculate_metrics,
)


def run_evaluation(run_dir: Path, epoch: Optional[int] = None, period: str = "test"):
    """Helper Function to run the evaluation
    (same as def start_evaluation: neuralhydrology/evaluation/evaluate.py:L7)

    Args:
        run_dir (Path): Path of the experiment run
        epoch (Optional[int], optional):
            Model epoch to evaluate. None finds the latest (highest) epoch.
            Defaults to None.
        period (str, optional): {"test", "train", "validation"}. Defaults to "test".
    """
    cfg = Config(run_dir / "config.yml")
    tester = Tester(cfg=cfg, run_dir=run_dir, period=period, init_model=True)

    if epoch is None:
        # get the highest epoch trained model
        all_trained_models = [d.name for d in (run_dir).glob("model_epoch*.pt")]
        epoch = int(
            sorted(all_trained_models)[-1].replace(".pt", "").replace("model_epoch", "")
        )
    print(f"** EVALUATING MODEL EPOCH: {epoch} **")
    tester.evaluate(epoch=epoch, save_results=True, metrics=["NSE", "KGE"])


def get_args() -> Dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str)
    parser.add_argument("--eval", type=bool, default=False)
    parser.add_argument("--epoch", type=int, default=30)
    parser.add_argument("--save_csv", type=bool, default=True)
    parser.add_argument("--ensemble", type=bool, default=False)
    parser.add_argument("--ensemble_filename", type=str, default=None)
    parser.add_argument("--metrics", type=bool, default=False)
    args = vars(parser.parse_args())

    return args


def get_test_filepath(run_dir: Path, epoch: Optional[int] = None) -> Path:
    # create filepath for test
    if epoch is None:
        #  get the maximum epoch
        all_evaluated_results = [d.name for d in (run_dir / "test").iterdir()]
        epoch = int(sorted(all_evaluated_results)[-1].replace("model_epoch", ""))

    test_dir = run_dir / f"test/model_epoch{epoch:03}/"
    res_fp = test_dir / "test_results.p"

    assert (
        res_fp.exists()
    ), f"Has validation been run? ipython --pdb neuralhydrology/nh_run.py evaluate -- --run-dir {run_dir.absolute()}"

    return res_fp


def get_ensemble_path(
    run_dir: Path, ensemble_filename: str = "ensemble_results.p"
) -> Path:
    res_fp = run_dir / ensemble_filename
    assert res_fp.exists(), f"Has validation been run? I cannot find {res_fp}"
    return res_fp


def _load_dict_to_xarray(
    res_dict: Dict[str, Dict[str, Dict[str, xr.Dataset]]]
) -> xr.Dataset:
    stations = [k for k in res_dict.keys()]
    # should only contain one frequency
    freq = [k for k in res_dict[stations[0]].keys()]
    assert len(freq) == 1
    freq = freq[0]

    #  extract the raw results
    all_xr_objects: List[xr.Dataset] = []
    for station_id in tqdm(stations):
        try:
            xr_obj = (
                res_dict[station_id][freq]["xr"].isel(time_step=0).drop("time_step")
            )
        except ValueError:
            # ensemble mode does not have "time_step" dimension
            xr_obj = res_dict[station_id][freq]["xr"].rename({"datetime": "date"})

        xr_obj = xr_obj.expand_dims({"station_id": [station_id]})
        all_xr_objects.append(xr_obj)

    #  merge all stations into one xarray object
    ds = xr.concat(all_xr_objects, dim="station_id")
    return ds


def get_all_station_ds(res_fp: Path) -> xr.Dataset:
    res_dict = pickle.load(res_fp.open("rb"))
    ds = _load_dict_to_xarray(res_dict)
    return ds


def calculate_all_error_metrics(
    preds: xr.Dataset,
    basin_coord: str = "basin",
    time_coord: str = "date",
    obs_var: str = "discharge_spec_obs",
    sim_var: str = "discharge_spec_sim",
    metrics: Optional[List[str]] = None,
) -> xr.Dataset:
    all_errors: List[pd.DataFrame] = []
    missing_data: List[str] = []

    pbar = tqdm(preds[basin_coord].values, desc="Calculating Errors")
    for sid in pbar:
        pbar.set_postfix_str(sid)
        sim = (
            preds[sim_var]
            .rename({basin_coord: "station_id", time_coord: "date"})
            .sel(station_id=sid)
        )
        obs = (
            preds[obs_var]
            .rename({basin_coord: "station_id", time_coord: "date"})
            .sel(station_id=sid)
        )

        if metrics is None:
            try:
                errors = calculate_all_metrics(sim=sim, obs=obs,)
                all_errors.append(pd.DataFrame({sid: errors}).T)
            except AllNaNError:
                missing_data.append(sid)
        else:
            try:
                errors = calculate_metrics(sim=sim, obs=obs, metrics=metrics)
                all_errors.append(pd.DataFrame({sid: errors}).T)
            except AllNaNError:
                missing_data.append(sid)

    errors = pd.concat(all_errors).to_xarray().rename({"index": basin_coord})
    return errors


def get_ds_and_metrics(
    res_fp: Path, metrics: bool = False
) -> Union[xr.Dataset, pd.DataFrame]:
    # load the dictionary of results
    res_dict = pickle.load(res_fp.open("rb"))
    stations = [k for k in res_dict.keys()]

    # should only contain one frequency
    freq = [k for k in res_dict[stations[0]].keys()]
    assert len(freq) == 1
    freq = freq[0]

    #  Create List of Datasets (obs, sim) and metric DataFrame
    output_metrics_dict: DefaultDict[str, List] = defaultdict(list)
    all_xr_objects: List[xr.Dataset] = []

    for station_id in tqdm(stations):
        #  extract the raw results
        try:
            xr_obj = (
                res_dict[station_id][freq]["xr"].isel(time_step=0).drop("time_step")
            )
        except ValueError:
            # ensemble mode does not have "time_step" dimension
            xr_obj = res_dict[station_id][freq]["xr"].rename({"datetime": "date"})
        xr_obj = xr_obj.expand_dims({"station_id": [station_id]}).rename(
            {"date": "time"}
        )
        all_xr_objects.append(xr_obj)

        if metrics:
            # extract the output metrics
            output_metrics_dict["station_id"].append(station_id)
            try:
                output_metrics_dict["NSE"].append(res_dict[station_id][freq]["NSE"])
                output_metrics_dict["KGE"].append(res_dict[station_id][freq]["KGE"])
                output_metrics_dict["MSE"].append(res_dict[station_id][freq]["MSE"])
                output_metrics_dict["FHV"].append(res_dict[station_id][freq]["FHV"])
                output_metrics_dict["FMS"].append(res_dict[station_id][freq]["FMS"])
                output_metrics_dict["FLV"].append(res_dict[station_id][freq]["FLV"])
            except KeyError:
                try:
                    output_metrics_dict["NSE"].append(
                        res_dict[station_id][freq][f"NSE_{freq}"]
                    )
                    output_metrics_dict["KGE"].append(
                        res_dict[station_id][freq][f"KGE_{freq}"]
                    )
                    output_metrics_dict["MSE"].append(
                        res_dict[station_id][freq][f"MSE_{freq}"]
                    )
                    output_metrics_dict["FHV"].append(
                        res_dict[station_id][freq][f"FHV_{freq}"]
                    )
                    output_metrics_dict["FMS"].append(
                        res_dict[station_id][freq][f"FMS_{freq}"]
                    )
                    output_metrics_dict["FLV"].append(
                        res_dict[station_id][freq][f"FLV_{freq}"]
                    )
                except KeyError:
                    output_metrics_dict["NSE"].append(np.nan)
                    output_metrics_dict["KGE"].append(np.nan)
                    output_metrics_dict["MSE"].append(np.nan)
                    output_metrics_dict["FHV"].append(np.nan)
                    output_metrics_dict["FMS"].append(np.nan)
                    output_metrics_dict["FLV"].append(np.nan)

    #  merge all stations into one xarray object
    ds = xr.concat(all_xr_objects, dim="station_id")

    if metrics:
        #  create metric dataframe
        metric_df = pd.DataFrame(output_metrics_dict)
    else:
        metric_df = pd.DataFrame([])

    return ds, metric_df


def get_old_format_dataframe(ds: xr.Dataset) -> pd.DataFrame:
    df = (
        ds.to_dataframe()
        .reset_index()
        .rename({"discharge_spec_obs": "obs", "discharge_spec_sim": "sim"}, axis=1)
    )
    return df


def main(
    run_dir: Path,
    bool_evaluation: bool = False,
    epoch: Optional[int] = None,
    save_csv: bool = True,
    ensemble: bool = False,
    ensemble_filename: str = "ensemble_results.p",
    metrics: bool = False,
) -> None:
    # run evaluation (optional)
    if bool_evaluation:
        run_evaluation(run_dir, epoch=epoch, period="test")

    if ensemble:
        res_fp = get_ensemble_path(run_dir, ensemble_filename)
    else:
        res_fp = get_test_filepath(run_dir, epoch)
    test_dir = res_fp.parents[0]

    ds, metric_df = get_ds_and_metrics(res_fp, metrics=metrics)

    # -------- SAVE ------------------------------------------
    print(f"** Writing `results.nc` and `metric_df.csv` to {test_dir} **")
    ds.to_netcdf(test_dir / "results.nc")

    if metrics:
        metric_df.to_csv(test_dir / "metric_df.csv")

    # create csv with informative name
    old_format = False
    if save_csv:
        print("** Writing results as .csv file **")
        try:
            fname = f"{test_dir.parents[1].name}_E{epoch:03}.csv"
        except:
            fname = f"{test_dir.parents[1].name}_ENS.csv"
        if old_format:
            df = get_old_format_dataframe(ds)
        else:
            df = ds.to_dataframe().reset_index()
        df.to_csv(test_dir / fname)


def read_multi_experiment_results(
    ensemble_dir: Path, ensemble_members: bool = True
) -> xr.Dataset:
    """
    Read all of the {expt_title}/test/model_epoch0{epoch}/*_results.p dictionaries
     into one xarray Dataset.

    Either read multiple ensemble members of the same experiment
        - NOTE: the experiment must be named `ensemble{ENS_NUM}`
    OR read multiple experiments using the experiment titles
    """
    paths = [d for d in (ensemble_dir).glob("**/*_results.p")]
    unique = np.unique([p.parent.name for p in paths])
    assert (
        len(unique) == 1
    ), f"Expected one epoch of test model results. Got: {unique}\nFrom {pprint.pformat(paths)}"
    ps = [pickle.load(p.open("rb")) for p in paths]

    output_dict = {}
    for i, res_dict in tqdm(enumerate(ps), desc="Loading Ensemble Members"):
        stations = [k for k in res_dict.keys()]
        freq = "1D"
        all_xr_objects: List[xr.Dataset] = []

        # get the ensemble number
        if ensemble_members:
            m = re.search("ensemble\d+", paths[i].__str__())
            try:
                name = m.group(0)
            except AttributeError as e:
                print("found ensemble mean")
                name = "mean"
        else:
            #  get the experiment title/name
            name = paths[i].parent.parent.parent.name
            # expect to find a datetime stamp in the name
            m = re.search("_\d+_\d+", paths[i].__str__())
            try:
                _ = m.group(0)
            except AttributeError as e:
                print("Found non-standard member dictionary")
                print(f"Skipping: {paths[i]}")
                continue

        for station_id in stations:
            #  extract the raw results
            try:
                xr_obj = (
                    res_dict[station_id][freq]["xr"].isel(time_step=0).drop("time_step")
                )
            except ValueError:
                # ensemble mode does not have "time_step" dimension
                xr_obj = res_dict[station_id][freq]["xr"].rename({"datetime": "date"})
            xr_obj = xr_obj.expand_dims({"station_id": [station_id]}).rename(
                {"date": "time"}
            )
            all_xr_objects.append(xr_obj)

        preds = xr.concat(all_xr_objects, dim="station_id")
        preds["station_id"] = [int(sid) for sid in preds["station_id"]]
        preds = preds.rename({"discharge_spec_obs": "obs", "discharge_spec_sim": "sim"})

        output_dict[name] = preds

    # return as one dataset
    all_ds = []
    for key in output_dict.keys():
        all_ds.append(
            output_dict[key].assign_coords({"member": key}).expand_dims("member")
        )
    ds = xr.concat(all_ds, dim="member")

    return ds


def calculate_member_errors(
    member_ds: xr.Dataset,
    basin_coord: str = "basin",
    time_coord: str = "date",
    obs_var: str = "discharge_spec_obs",
    sim_var: str = "discharge_spec_sim",
) -> xr.Dataset:
    assert "member" in member_ds.coords

    all_errors = []
    print(f"Calculating Errors for {len(member_ds['member'].values)} members")
    for member in member_ds["member"].values:
        preds = member_ds.sel(member=member)
        err = calculate_all_error_metrics(
            preds,
            basin_coord=basin_coord,
            time_coord=time_coord,
            obs_var=obs_var,
            sim_var=sim_var,
        )
        err = err.assign_coords(member=member).expand_dims("member")
        all_errors.append(err)

    all_errors = xr.merge(all_errors)

    return all_errors


if __name__ == "__main__":
    #  read cmd line arguments
    args = get_args()
    run_dir: Path = Path(args["run_dir"])
    bool_evaluation: bool = args["eval"]
    epoch: Optional[int] = args["epoch"]
    save_csv: bool = args["save_csv"]
    ensemble: bool = args["ensemble"]
    ensemble_filename: str = args["ensemble_filename"]
    metrics: bool = args["metrics"]

    main(
        run_dir=run_dir,
        bool_evaluation=bool_evaluation,
        epoch=epoch,
        save_csv=save_csv,
        ensemble=ensemble,
        ensemble_filename=ensemble_filename,
        metrics=metrics,
    )
