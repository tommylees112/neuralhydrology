from pathlib import Path 
from typing import Optional
import torch
from torch import nn
from neuralhydrology.modelzoo import get_model
from neuralhydrology.utils.config import Config


def _load_weights(model, cfg: Config, epoch: int = None) -> nn.Module:
    """Load weights of a certain (or the last) epoch into the model."""
    weight_file = _get_weight_file(cfg, epoch)
    print(f"Using the model weights from {weight_file}")
    model = model.load_state_dict(torch.load(weight_file, map_location=cfg.device))
    return model 


def _get_weight_file(cfg: Config, epoch: Optional[int] = None):
    """Get file path to weight file"""
    if epoch is None:
        weight_file = sorted(list(cfg.run_dir.glob('model_epoch*.pt')))[-1]
    else:
        weight_file = cfg.run_dir / f"model_epoch{str(epoch).zfill(3)}.pt"

    return weight_file


if __name__ == "__main__":
    run_dir = Path("/datadrive/data/runs/nh_runoff_2805_130733")
    cfg = Config(run_dir / "config.yml")
    model = get_model(cfg).to(cfg.device)
    model = _load_weights(model, cfg)
