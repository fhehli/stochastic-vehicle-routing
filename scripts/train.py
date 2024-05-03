import argparse
import random

import torch
import yaml

from src.trainer import Trainer


def main(args):
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    trainer = Trainer(config)
    trainer.train()
    trainer.test()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model.")
    parser.add_argument("--config", type=str, help="Path of config file.")
    args = parser.parse_args()

    random.seed(0)
    torch.manual_seed(0)
    main(args)
