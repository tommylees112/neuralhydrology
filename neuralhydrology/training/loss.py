from typing import Dict, List, Tuple

import numpy as np
import torch

from neuralhydrology.training.regularization import BaseRegularization
from neuralhydrology.utils.config import Config

ONE_OVER_2PI_SQUARED = 1.0 / np.sqrt(2.0 * np.pi)


class BaseLoss(torch.nn.Module):
    """Base loss class.

    All losses extend this class by implementing `_get_loss`.

    Parameters
    ----------
    cfg : Config
        The run configuration.
    prediction_keys : List[str]
        List of keys that will be predicted. During the forward pass, the passed `prediction` dict
        must contain these keys. Note that the keys listed here should be without frequency identifier.
    ground_truth_keys : List[str]
        List of ground truth keys that will be needed to compute the loss. During the forward pass, the
        passed `data` dict must contain these keys. Note that the keys listed here should be without
        frequency identifier.
    additional_data : List[str], optional
        Additional list of keys that will be taken from `data` in the forward pass to compute the loss.
        For instance, this parameter can be used to pass the variances that are needed to compute an NSE.
    """

    def __init__(self,
                 cfg: Config,
                 prediction_keys: List[str],
                 ground_truth_keys: List[str],
                 additional_data: List[str] = None):
        super(BaseLoss, self).__init__()
        self._predict_last_n = _get_predict_last_n(cfg)
        self._frequencies = [f for f in self._predict_last_n.keys() if f not in cfg.no_loss_frequencies]

        self._regularization_terms = []

        # names of ground truth and prediction keys to be unpacked and subset to predict_last_n items.
        self._prediction_keys = prediction_keys
        self._ground_truth_keys = ground_truth_keys

        # subclasses can use this list to register inputs to be unpacked during the forward call
        # and passed as kwargs to _get_loss() without subsetting.
        self._additional_data = []
        if additional_data is not None:
            self._additional_data = additional_data

    def forward(self, prediction: Dict[str, torch.Tensor], data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Calculate the loss.

        Parameters
        ----------
        prediction : Dict[str, torch.Tensor]
            Dictionary of predictions for each frequency. If more than one frequency is predicted,
            the keys must have suffixes ``_{frequency}``. For the required keys, refer to the documentation
            of the concrete loss.
        data : Dict[str, torch.Tensor]
            Dictionary of ground truth data for each frequency. If more than one frequency is predicted,
            the keys must have suffixes ``_{frequency}``. For the required keys, refer to the documentation
            of the concrete loss.

        Returns
        -------
        torch.Tensor
            The calculated loss.
        """
        # unpack loss-specific additional arguments
        kwargs = {key: data[key] for key in self._additional_data}

        losses = []
        prediction_sub, ground_truth_sub = {}, {}
        for freq in self._frequencies:
            if self._predict_last_n[freq] == 0:
                continue  # no predictions for this frequency
            freq_suffix = '' if freq == '' else f'_{freq}'

            # apply predict_last_n and mask
            freq_pred, freq_gt = self._subset({key: prediction[f'{key}{freq_suffix}'] for key in self._prediction_keys},
                                              {key: data[f'{key}{freq_suffix}'] for key in self._ground_truth_keys},
                                              self._predict_last_n[freq])
            # remember subsets for multi-frequency component
            prediction_sub.update({f'{key}{freq_suffix}': freq_pred[key] for key in freq_pred.keys()})
            ground_truth_sub.update({f'{key}{freq_suffix}': freq_gt[key] for key in freq_gt.keys()})

            losses.append(self._get_loss(freq_pred, freq_gt, **kwargs))

        loss = torch.mean(torch.stack(losses))
        for regularization in self._regularization_terms:
            loss = loss + regularization(prediction_sub, ground_truth_sub,
                                         {k: v for k, v in prediction.items() if k not in self._prediction_keys})
        return loss

    def _subset(self, prediction: Dict[str, torch.Tensor], ground_truth: Dict[str, torch.Tensor], predict_last_n: int) \
            -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
            # TODO: talk to freddy and daniel about this subsetting
        ground_truth_sub = {key: gt[:, -predict_last_n:, :] for key, gt in ground_truth.items()}
        prediction_sub = {key: pred[:, -predict_last_n:, :] for key, pred in prediction.items()}

        return prediction_sub, ground_truth_sub

    def _get_loss(self, prediction: Dict[str, torch.Tensor], ground_truth: Dict[str, torch.Tensor], **kwargs):
        raise NotImplementedError

    def set_regularization_terms(self, regularization_modules: List[BaseRegularization]):
        """Register the passed regularization terms to be added to the loss function.

        Parameters
        ----------
        regularization_modules : List[BaseRegularization]
            List of regularization functions to be added to the loss during `forward`.
        """
        self._regularization_terms = regularization_modules


class MaskedMSELoss(BaseLoss):
    """Mean squared error loss.

    To use this loss in a forward pass, the passed `prediction` dict must contain
    the key ``y_hat``, and the `data` dict must contain ``y``.

    Parameters
    ----------
    cfg : Config
        The run configuration.
    """

    def __init__(self, cfg: Config):
        super(MaskedMSELoss, self).__init__(cfg, ['y_hat'], ['y'])

    def _get_loss(self, prediction: Dict[str, torch.Tensor], ground_truth: Dict[str, torch.Tensor], **kwargs):
        mask = ~torch.isnan(ground_truth['y'])
        loss = 0.5 * torch.mean((prediction['y_hat'][mask] - ground_truth['y'][mask])**2)
        return loss


class MaskedRMSELoss(BaseLoss):
    """Root mean squared error loss.

    To use this loss in a forward pass, the passed `prediction` dict must contain
    the key ``y_hat``, and the `data` dict must contain ``y``.

    Parameters
    ----------
    cfg : Config
        The run configuration.
    """

    def __init__(self, cfg: Config):
        super(MaskedRMSELoss, self).__init__(cfg, ['y_hat'], ['y'])

    def _get_loss(self, prediction: Dict[str, torch.Tensor], ground_truth: Dict[str, torch.Tensor], **kwargs):
        mask = ~torch.isnan(ground_truth['y'])
        loss = torch.sqrt(0.5 * torch.mean((prediction['y_hat'][mask] - ground_truth['y'][mask])**2))
        return loss


class MaskedNSELoss(BaseLoss):
    """Basin-averaged Nash--Sutcliffe Model Efficiency Coefficient loss.

    To use this loss in a forward pass, the passed `prediction` dict must contain
    the key ``y_hat``, and the `data` dict must contain ``y`` and ``per_basin_target_stds``.

    A description of the loss function is available in [#]_.

    Parameters
    ----------
    cfg : Config
        The run configuration.
    eps: float, optional
        Small constant for numeric stability.

    References
    ----------
    .. [#] Kratzert, F., Klotz, D., Shalev, G., Klambauer, G., Hochreiter, S., and Nearing, G.: "Towards learning
       universal, regional, and local hydrological behaviors via machine learning applied to large-sample datasets"
       *Hydrology and Earth System Sciences*, 2019, 23, 5089-5110, doi:10.5194/hess-23-5089-2019
    """

    def __init__(self, cfg: Config, eps: float = 0.1):
        super(MaskedNSELoss, self).__init__(cfg, ['y_hat'], ['y'], additional_data=['per_basin_target_stds'])
        self.eps = eps

    def _get_loss(self, prediction: Dict[str, torch.Tensor], ground_truth: Dict[str, torch.Tensor], **kwargs):
        mask = ~torch.isnan(ground_truth['y'])
        y_hat = prediction['y_hat'][mask]
        y = ground_truth['y'][mask]
        per_basin_target_stds = kwargs['per_basin_target_stds']
        # expand dimension 1 to predict_last_n
        # assert False  # TODO: TOMMY
        per_basin_target_stds = per_basin_target_stds.expand_as(prediction['y_hat'])[mask]

        squared_error = (y_hat - y)**2
        weights = 1 / (per_basin_target_stds + self.eps)**2
        scaled_loss = weights * squared_error
        return torch.mean(scaled_loss)


class MaskedWeightedNSELoss(BaseLoss):
    """Basin-averaged Nash--Sutcliffe Model Efficiency Coefficient loss. 

    This loss function weights multiple outputs according to user-specified weights in the 
    config argument 'target_loss_weights'. To use this loss in a forward pass, the passed 
    `prediction` dict must contain the key ``y_hat``, and the `data` dict must contain 
    ``y`` and ``per_basin_target_stds``.

    Parameters
    ----------
    cfg : Config
        The run configuration.
    eps: float, optional
        Small constant for numeric stability.
    """

    def __init__(self, cfg: Config, eps: float = 0.1):
        super(MaskedWeightedNSELoss, self).__init__(cfg, ['y_hat'], ['y'], additional_data=['per_basin_target_stds'])

        self.eps = eps

        if cfg.target_loss_weights is None:
            raise ValueError('target_loss_weights must be specified for WeightedNSELoss')
        elif len(cfg.target_loss_weights) != len(cfg.target_variables):
            raise ValueError("Number of loss weights must be equal to the number of target_variables.")
        else:
            self._loss_weights = torch.tensor(cfg.target_loss_weights).to(cfg.device)

    def _get_loss(self, prediction: Dict[str, torch.Tensor], ground_truth: Dict[str, torch.Tensor], **kwargs):
    
        mask = ~torch.isnan(ground_truth['y'])
        y_hat = prediction['y_hat'] * torch.sqrt(self._loss_weights)
        y = ground_truth['y'] * torch.sqrt(self._loss_weights)

        per_basin_target_stds = kwargs['per_basin_target_stds']
        per_basin_target_stds = per_basin_target_stds.expand_as(prediction['y_hat'])

        squared_error = (y_hat[mask] - y[mask])**2
        norm_factor = 1 / (per_basin_target_stds[mask] + self.eps)**2
        scaled_loss = norm_factor * squared_error

        return torch.mean(scaled_loss)


def _get_predict_last_n(cfg: Config) -> dict:
    predict_last_n = cfg.predict_last_n
    if isinstance(predict_last_n, int):
        predict_last_n = {'': predict_last_n}
    if len(predict_last_n) == 1:
        predict_last_n = {'': list(predict_last_n.values())[0]}  # if there's only one frequency, we omit its identifier
    return predict_last_n
