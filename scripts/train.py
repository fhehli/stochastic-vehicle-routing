import argparse
import random

import torch
import yaml

from src.trainer import Trainer
from src.utils import get_model, get_criterion, get_optimizer, get_data_loaders


def main(args):
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    model = get_model(config)
    data_loaders = get_data_loaders(config)
    optimizer_class, optimizer_kwargs = get_optimizer(config)
    criterion = get_criterion(config)

    optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)

    trainer = Trainer(model, data_loaders, optimizer, criterion, config)
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model.")
    parser.add_argument("--config", type=str, help="Path of config file.")
    args = parser.parse_args()

    random.seed(0)
    torch.manual_seed(0)
    main(args)
