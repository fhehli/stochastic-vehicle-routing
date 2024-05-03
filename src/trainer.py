from functools import partial
import torch
from src.VSPSolver import solve_vsp
from src.utils import get_model, get_criterion, get_optimizer, get_dataloaders


class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model = get_model(config, device=self.device)
        self.train_loader, self.val_loader, self.test_loader = get_dataloaders(config)

        optimizer_class, optimizer_kwargs = get_optimizer(config)
        self.optimizer = optimizer_class(self.model.parameters(), **optimizer_kwargs)

        criterion_class, criterion_kwargs = get_criterion(config)
        self.criterion = lambda func: criterion_class(func, **criterion_kwargs)

        self.n_epochs = config["train"]["n_epochs"]
        self.eval_every = config["train"]["eval_every_n_epochs"]
        self.save_every = config["train"]["save_every_n_epochs"]
        self.save_dir = Path(config["train"]["save_dir"])
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def compute_metrics(self, i):
        pass

    def save_model(self, i):
        pass

    def train_epoch(self, i):
        self.model.train()
        for inputs, labels, graph in self.train_loader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            theta = self.model(inputs)
            func = partial(solve_vsp, graph=graph)
            criterion = self.criterion(func)
            loss = criterion(theta, labels)
            loss.backward()
            self.optimizer.step()

    def train(self):
        for i in range(self.n_epochs):
            self.train_epoch(i)
            self.compute_metrics(i)
            self.save_model(i)
