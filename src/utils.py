from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader

from src.models import FenchelYoungGLM
from src.perturbations.fenchel_young import FenchelYoungLoss

MODELS = {
    "FenchelYoungGLM": FenchelYoungGLM,
}
OPTIMIZERS = {
    "AdamW": AdamW,
}


def get_model(config):
    name = config["model"]["name"]
    assert name in MODELS, f"Model not found in {MODELS.keys()}"
    model_args = config["model"]["args"]

    return MODELS[name](**model_args)


def get_data_loaders(config):
    raise NotImplementedError


def get_optimizer(config):
    name = config["optimizer"]["name"]
    assert name in OPTIMIZERS, f"Optimizer not found in {OPTIMIZERS.keys()}"
    optimizer_args = config["optimizer"]["args"]

    return OPTIMIZERS[name](**optimizer_args)


def get_criterion(config):
    return FenchelYoungLoss()
