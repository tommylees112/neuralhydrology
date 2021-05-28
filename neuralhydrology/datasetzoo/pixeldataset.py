from typing import List, Dict, Union, Optional, Tuple

import pandas as pd
import xarray as xr

from neuralhydrology.datasetzoo.basedataset import BaseDataset
from neuralhydrology.utils.config import Config


def _stack_xarray(
    ds: xr.Dataset, spatial_coords: List[str]
) -> Tuple[xr.Dataset, xr.Dataset]:
    #  stack values
    stacked = ds.stack(sample=spatial_coords)
    samples = stacked.sample
    pixel_strs = [f"{ll[0]}_{ll[-1]}" for ll in samples.values] if len(samples.values[0]) == 2 else [f"{sample[0]}" for sample in samples.values]
    stacked["sample"] = pixel_strs

    samples = samples.to_dataset(name="pixel")
    samples = xr.DataArray(
        pixel_strs, dims=["sample"], coords={"sample": samples.sample}
    )
    return stacked, samples


class PixelDataset(BaseDataset):
    """Write a dataset class that loads in stacked xarray data (pixel-time) into NeuralHydrology. 
    We need to implement two functions:
        1) _load_basin_data -> which loads the time series data
        2) _load_attributes -> which loads the static attributes
    The "basin" here will be a string like "lat_lon", where lat and lon are the coordinates of the pixel.
    We treat every pixel as a "basin".

    Need to add this dataset to the `get_dataset()` function in 'neuralhydrology.datasetzoo.__init__.py'

    The results of the tester would be the usual dictionary with a key per pixel and the xarray 
     for that particular pixel, which you can then stack into a 3D-xarray again.
    
    Parameters
    ----------
    cfg : Config
        The run configuration.
    is_train : bool 
        Defines if the dataset is used for training or evaluating. If True (training), means/stds for each feature
        are computed and stored to the run directory. If one-hot encoding is used, the mapping for the one-hot encoding 
        is created and also stored to disk. If False, a `scaler` input is expected and similarly the `id_to_int` input
        if one-hot encoding is used. 
    period : {'train', 'validation', 'test'}
        Defines the period for which the data will be loaded
    basin : str, optional
        If passed, the data for only this basin will be loaded. Otherwise the basin(s) are read from the appropriate
        basin file, corresponding to the `period`.
    additional_features : List[Dict[str, pd.DataFrame]], optional
        List of dictionaries, mapping from a basin id to a pandas DataFrame. This DataFrame will be added to the data
        loaded from the dataset, and all columns are available as 'dynamic_inputs', 'evolving_attributes' and
        'target_variables'
    id_to_int : Dict[str, int], optional
        If the config argument 'use_basin_id_encoding' is True in the config and period is either 'validation' or 
        'test', this input is required. It is a dictionary, mapping from basin id to an integer (the one-hot encoding).
    scaler : Dict[str, Union[pd.Series, xarray.DataArray]], optional
        If period is either 'validation' or 'test', this input is required. It contains the means and standard 
        deviations for each feature and is stored to the run directory during training (train_data/train_data_scaler.p)
    """

    def __init__(self,
                 cfg: Config,
                 is_train: bool,
                 period: str,
                 basin: str = None,
                 additional_features: List[Dict[str, pd.DataFrame]] = [],
                 id_to_int: Dict[str, int] = {},
                 scaler: Dict[str, Union[pd.Series, xr.DataArray]] = {}):
        
        assert cfg.dynamic_nc_file is not None, "With Pixel DataLoader you are required to provide a `train_basin_file` in your config.yml file"
        assert cfg.pixel_dims is not None, "With Pixel DataLoader you are required to provide the `pixel_dims` in your config.yml file"
        self.load_netcdf_memory(cfg=cfg)

        # initialize parent class
        super(PixelDataset, self).__init__(cfg=cfg,
                                              is_train=is_train,
                                              period=period,
                                              basin=basin,
                                              additional_features=additional_features,
                                              id_to_int=id_to_int,
                                              scaler=scaler)

    def load_netcdf_memory(self, cfg: Config) -> xr.Dataset:
        pixel_dims = cfg.pixel_dims
        data = xr.open_dataset(cfg.dynamic_nc_file)
        if cfg.static_nc_file is not None:
            static_data = xr.open_dataset(cfg.static_nc_file)
        else:
            static_data = None

        # stack dataset on pixel dimensions
        stacked_dynamic, sample = _stack_xarray(data, pixel_dims)
        
        stacked_static: Optional[xr.Dataset]
        if static_data is not None:
            stacked_static, _ = _stack_xarray(static_data, cfg.pixel_dims)
        else:
            stacked_static = None

        # for interoperability rename "sample" -> "basin"
        stacked_dynamic = stacked_dynamic.rename({"sample": "basin"})
        if stacked_static is not None:
            stacked_static = stacked_static.rename({"sample": "basin"})

        # rename "time" -> "date"
        stacked_dynamic = stacked_dynamic.rename({"time": "date"})

        self.dynamic = stacked_dynamic
        self.static = stacked_static

    def _load_basin_data(self, basin: str) -> pd.DataFrame:
        """Load basin time series data
        
        This function is used to load the time series data (meteorological forcing, streamflow, etc.) and make available
        as time series input for model training later on. Make sure that the returned dataframe is time-indexed.
        
        Parameters
        ----------
        basin : str
            Basin identifier as string.
        Returns
        -------
        pd.DataFrame
            Time-indexed DataFrame, containing the time series data (e.g., forcings + discharge).
        """
        ########################################
        # Add your code for data loading here. #
        ########################################
        ds = self.dynamic.sel(basin=basin)
        df = ds.to_dataframe().reset_index().set_index("date")
        return df

    def _load_attributes(self) -> pd.DataFrame:
        """Load dataset attributes
        
        This function is used to load basin attribute data (e.g. CAMELS catchments attributes) as a basin-indexed 
        dataframe with features in columns.
        
        Returns
        -------
        pd.DataFrame
            Basin-indexed DataFrame, containing the attributes as columns.
        """
        ########################################
        # Add your code for data loading here. #
        ########################################
        df = self.static.to_dataframe().reset_index().set_index("basin")
        return df
