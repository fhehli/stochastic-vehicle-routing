import argparse

import numpy as np
import torch
from ray import tune
import yaml

from src.trainer import Trainer


def merge_deep(source, destination):
    for key, value in source.items():
        if isinstance(value, dict):
            if key not in destination or not isinstance(destination[key], dict):
                destination[key] = {}
            merge_deep(value, destination[key])
        else:
            destination[key] = value

    return destination


def objective(config):
    trainer = Trainer(config)
    losses = trainer.train()
    return {"score": np.mean(losses)}


search_space = {
    "train": {
        "criterion": {
            "args": {"num_samples": tune.grid_search([5, 10, 25, 50, 100]), "sigma": tune.grid_search([0.01, 0.1, 1.0])}
        },
        "optimizer": {"args": {"lr": tune.grid_search([1e-4, 1e-3, 1e-2, 1e-1])}},
    }
}


def main(args):
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    param_space = merge_deep(search_space, config)

    tuner = tune.Tuner(objective, param_space=param_space)

    results = tuner.fit()

    print(results.get_best_result(metric="score", mode="min").config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter search.")
    parser.add_argument("--config", type=str, help="Path of config file.")
    args = parser.parse_args()

    torch.manual_seed(0)
    main(args)
