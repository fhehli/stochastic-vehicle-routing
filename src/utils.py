import pickle
from typing import List, Tuple

import numpy as np
import torch.nn as nn
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader, Dataset, random_split

from src.city import SimpleDirectedGraph
from src.models import FenchelYoungGLM
from src.perturbations.fenchel_young import FenchelYoungLoss

MODELS = {
    "FenchelYoungGLM": FenchelYoungGLM,
}
OPTIMIZERS = {
    "AdamW": AdamW,
}


class CitiesDataset(Dataset):
    def __init__(self, X, Y, graphs):
        self.X: np.ndarray = X
        self.Y: np.ndarray = Y
        self.graphs: List[SimpleDirectedGraph] = graphs

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, SimpleDirectedGraph]:
        return self.X[idx], self.Y[idx], self.graphs[idx]


def get_model(config) -> nn.Module:
    name = config["model"]["name"]
    assert name in MODELS, f"Model not found in {MODELS.keys()}"
    model_args = config["model"]["args"]

    return MODELS[name](**model_args)


def get_dataloaders(config):
    with open(config["data"]["path"], "rb") as file:
        data = pickle.load(file)
        X = data["X"]
        Y = data["Y"]
        graphs = data["graphs"]

    dataset = CitiesDataset(X, Y, graphs)
    split = config["data"]["split"]
    train_size, val_size, test_size = split["train"], split["val"], split["test"]
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    batch_size = config["train"]["batch_size"]
    assert batch_size == 1, "Batch size > 1 not implemented."  # TODO: write a collate_fn for batch_size > 1
    batch_size = None

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def get_optimizer(config) -> Tuple[Optimizer, dict]:
    name = config["train"]["optimizer"]["name"]
    assert name in OPTIMIZERS, f"Optimizer not found in {OPTIMIZERS.keys()}"
    optimizer_args = config["train"]["optimizer"]["args"]
    optimizer_args = optimizer_args if optimizer_args is not None else {}

    return OPTIMIZERS[name], optimizer_args


def get_criterion(config):
    return FenchelYoungLoss()
