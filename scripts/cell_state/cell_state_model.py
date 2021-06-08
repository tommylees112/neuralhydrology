from neuralhydrology.modelzoo.basemodel import BaseModel
import torch 
from torch import Tensor
from torch import nn
import numpy as np
from tqdm import tqdm
from typing import List, Tuple, Any, Dict, Union

import sys
sys.path.append("/home/tommy/neuralhydrology")
from scripts.cell_state.cell_state_dataset import CellStateDataset, get_train_test_dataset, train_validation_split
from neuralhydrology.modelzoo.head import get_head
from neuralhydrology.modelzoo.basemodel import BaseModel
from neuralhydrology.utils.config import Config


def create_model(dataset: CellStateDataset, device: str = "cpu"):
    # number of weights == number of dimensions in cell state vector (cfg.hidden_size)
    D_in = dataset['dimensions']
    model = torch.nn.Sequential(torch.nn.Linear(D_in, 1))
    model = model.to(device)
    return model


def train_model(
    model,
    train_dataset,
    learning_rate: float = 1e-2,
    n_epochs: int = 5,
    weight_decay: float = 0,
    val_split: bool = False,
    desc: str = "Training",
) -> Tuple[Any, List[float], List[float]]:

    if not val_split:
        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

    loss_fn = torch.nn.MSELoss(reduction="sum")

    # optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    #  TRAIN
    train_losses_ALL = []
    val_losses_ALL = []
    for epoch in tqdm(range(n_epochs), desc=desc):
        train_losses_ALL = []
        val_losses_ALL = []
        #  new train-validation split each epoch
        if val_split:
            #  create a unique test, val set (random) for each ...
            train_sampler, val_sampler = train_validation_split(train_dataset)
            train_loader = DataLoader(
                train_dataset, batch_size=256, sampler=train_sampler
            )
            val_loader = DataLoader(train_dataset, batch_size=256, sampler=val_sampler)
        else:
            train_loader = DataLoader(train_dataset, batch_size=256)

        for (basin, time), data in train_loader:
            X, y = data
            y_pred = model(X)
            loss = loss_fn(y_pred, y)

            # train/update the weights
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
            val_losses_ALL = np.mean(val_losses)

        train_losses_ALL.append(np.mean(train_losses))
        
    return model, train_losses_ALL, val_losses_ALL


if __name__ == "__main__":
    pass