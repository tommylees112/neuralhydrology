from typing import List
import pandas as pd 
import xarray as xr
import numpy as np
from pathlib import Path
from collections import defaultdict
from scipy.stats import spearmanr, kendalltau
from sklearn.metrics import mutual_info_score


# contants
IMP_FEATURES = [
    # topography
    "dpsbar",
    "elev_mean",
    "elev_min",
    "elev_max",
    # Hydrologic Attributes
    "q_mean",
    "runoff_ratio",
    "slope_fdc",
    "baseflow_index",
    "Q5",
    "Q95",
    #  climatic indices
    "p_mean",
    "pet_mean",
    "aridity",
    "frac_snow",
    # landcover
    "dwood_perc",
    "ewood_perc",
    "grass_perc",
    "shrub_perc",
    "crop_perc",
    "urban_perc",
]


def calc_MI(x, y, n_bins):
    c_xy = np.histogram2d(x, y, n_bins)[0]
    mi = mutual_info_score(None, None, contingency=c_xy)
    return mi


def calculate_correlation_with_static_features(metric_values: pd.Series, static_df: pd.DataFrame, method: str = "spearman") -> pd.DataFrame:
    func_lookup = {
        "kendall": kendalltau,
        "spearman": spearmanr,
        "mutual_information": calc_MI,
    }
    out = defaultdict(list)
    
    static_df = static_df.loc[metric_values.index]
    assert all(static_df.index == metric_values.index)
    
    for feature in static_df.columns:
        result = func_lookup[method](metric_values, static_df[feature])
        out["correlation"].append(result.correlation)
        out["pvalue"].append(result.pvalue)
        out["variable"].append(feature)

    rank_correlations = pd.DataFrame(out)
    return rank_correlations


def calculate_correlations(
    static_df: pd.DataFrame, 
    catchment_averaged_var: pd.Series,
    model: str = "LSTM",
    method: str = "spearman",
):
    ## Calculate correlations 
    # for variance_metric
    corr_df = calculate_correlation_with_static_features(catchment_averaged_var, static_df, method=method)
    corr_df["model"] = model
    return corr_df


def calculate_multi_model_correlations(
    all_catchment_vars: List[pd.DataFrame],
    models: List[str] = ["LSTM", "EALSTM", "TOPMODEL", "SACRAMENTO", "ARNOVIC", "PRMS"],
    method: str = "spearman",
):
    assert len(models) == len(all_catchment_vars)
    corr_df = calculate_correlations(static_df, all_catchment_vars[0], method=method)
    for ix, model in enumerate(models):
        if ix != 0:
            _corr_df = calculate_correlations(static_df, all_catchment_vars[ix], method=method)
            _corr_df["model"] = model
            corr_df = pd.concat([corr_df, _corr_df])

    return corr_df


def corr_df_ready_for_plotting(
    corr_df: pd.DataFrame, 
    models: List[str] = ["LSTM", "EALSTM", "TOPMODEL", "SACRAMENTO", "ARNOVIC", "PRMS"],
    important_features: List[str] = IMP_FEATURES,
) -> pd.DataFrame:
    ## Assign the extra columns (for the plotter below)
    corr_df["significant"] = corr_df["pvalue"] < 0.001
    corr_df["positive"] = corr_df["correlation"] > 0

    # Sort Model columns for plotting
    model_sorter = models 
    variable_sorter = important_features

    corr_df["model"] = corr_df["model"].astype("category")
    corr_df["model"].cat.set_categories(model_sorter, inplace=True)

    # Sort Variable columns for plotting
    corr_df["variable"] = corr_df["variable"].astype("category")
    corr_df["variable"].cat.set_categories(variable_sorter, inplace=True)
    corr_df = corr_df.sort_values(["model", "variable"])

    return corr_df


if __name__ == "__main__":
    olddata_dir = Path("/datadrive/olddata")
    data_dir = Path("/datadrive/data")
    run_dir = olddata_dir / "runs/ensemble_lstm_TEMP"
    ealstm_run_dir = olddata_dir / "runs/ensemble_ealstm_TEMP"

    # static data
    all_static = xr.open_dataset(olddata_dir / f'RUNOFF/interim/static/data.nc')
    all_static['station_id'] = all_static['station_id'].astype(int)
    static = all_static

    # model errors 
    ensemble_fp = run_dir / "ensemble_all.nc"
    ensemble_ds = xr.open_dataset(ensemble_fp)
    # ea lstm
    ealstm_ensemble_fp = ealstm_run_dir / "ensemble_all.nc"
    ealstm_ensemble_ds = xr.open_dataset(ealstm_ensemble_fp)

    std_norm_q = (ensemble_ds.drop("obs").std(dim="member")/ static.q_mean).mean(dim="time") .to_dataframe()
    ealstm_std_norm_q = (ensemble_ds.drop("obs").std(dim="member") / static.q_mean).mean(dim="time").to_dataframe()


    important_features = IMP_FEATURES
    static_df = static[important_features].to_dataframe()

    # calculate correlations 
    corr_df = calculate_multi_model_correlations(
        all_catchment_vars=[std_norm_q, ealstm_std_norm_q], 
        models=["LSTM", "EALSTM"],
    )

    # set dataframe ready for plotting
    corr_df = corr_df_ready_for_plotting(corr_df)
    assert False