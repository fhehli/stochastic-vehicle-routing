from functools import partial
from pathlib import Path
import pickle

import numpy as np
import torch
from tqdm.auto import tqdm

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
        self.with_city = config["data"]["city"]

    def compute_baseline(self):
        baselines = []
        self.model.eval()
        with torch.no_grad():
            for inputs, labels, instance in self.val_loader:
                graph = instance.graph if self.with_city else instance
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                opt_cost = instance.compute_solution_cost(labels)
                baseline = solve_vsp(inputs[:, 0].unsqueeze(0), graph)
                baseline_cost = instance.compute_solution_cost(baseline.squeeze())
                baseline_percentage = baseline_cost / opt_cost - 1.0
                baselines.append(baseline_percentage)
        print(f"Baseline: {np.mean(baselines):.3f}")

    def compute_metrics(self, i):
        if i % self.eval_every != 0:
            return
        losses, percentage_from_opt = [], []
        self.model.eval()
        with torch.no_grad():
            for inputs, labels, instance in tqdm(self.val_loader, desc=f"Validation epoch {i}"):
                graph = instance.graph if self.with_city else instance
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                theta = self.model(inputs)
                func = partial(solve_vsp, graph=graph)
                criterion = self.criterion(func)
                loss = criterion(theta, labels).mean()
                losses.append(loss.item())

                if self.with_city:
                    solution = solve_vsp(theta.unsqueeze(0), graph)
                    cost = instance.compute_solution_cost(solution.squeeze())
                    opt_cost = instance.compute_solution_cost(labels)
                    percentage_from_opt.append(cost / opt_cost - 1.0)

        print(f"Validation loss: {np.mean(losses):.3f}")
        if self.with_city:
            print(f"Percentage from optimal: {np.mean(percentage_from_opt) * 100:.3f}%")
        return np.mean(losses), np.mean(percentage_from_opt)

    def save_model(self, i):
        if i % self.save_every == 0 and i > 0:
            torch.save(self.model.state_dict(), self.save_dir / f"epoch{i}.pt")

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))

    def train_epoch(self, i):
        self.model.train()
        losses, percentage_from_opt = [], []
        for inputs, labels, instance in tqdm(self.train_loader, desc=f"Train epoch {i}"):
            graph = instance.graph if self.with_city else instance
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            theta = self.model(inputs)

            func = partial(solve_vsp, graph=graph)
            criterion = self.criterion(func)
            loss = criterion(theta, labels)
            losses.append(loss.item())

            with torch.no_grad():
                if self.with_city:
                    solution = solve_vsp(theta.unsqueeze(0), graph)
                    cost = instance.compute_solution_cost(solution.squeeze())
                    opt_cost = instance.compute_solution_cost(labels)
                    percentage_from_opt.append(cost / opt_cost - 1.0)

            loss.backward()
            self.optimizer.step()

        print(f"Train loss: {np.mean(losses):.3f}")

        return np.mean(losses), np.mean(percentage_from_opt)

    def train(self):
        train_losses, train_cost_gaps, valid_losses, valid_cost_gaps = [], [], [], []

        for i in range(1, self.n_epochs + 1):
            train_loss, train_cost_gap = self.train_epoch(i)
            valid_loss, valid_cost_gap = self.compute_metrics(i)

            train_losses.append(train_loss)
            train_cost_gaps.append(train_cost_gap)
            valid_losses.append(valid_loss)
            valid_cost_gaps.append(valid_cost_gap)

            self.save_model(i)

        pickle.dump(
            {
                "train_losses": train_losses,
                "train_cost_gaps": train_cost_gaps,
                "valid_losses": valid_losses,
                "valid_cost_gaps": valid_cost_gaps,
            },
            open(self.save_dir / "metrics.pkl", "wb"),
        )

    def test(self):
        self.model.eval()
        losses = []
        with torch.no_grad():
            for inputs, labels, instance in tqdm(self.test_loader):
                graph = instance.graph if self.with_city else instance
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                theta = self.model(inputs)
                func = partial(solve_vsp, graph=graph)
                criterion = self.criterion(func)
                loss = criterion(theta, labels)
                losses.append(loss.item())

        print(f"Test loss: {np.mean(losses):.3f}")

    def predict(self, inputs, graph):
        self.model.eval()
        with torch.no_grad():
            inputs = inputs.to(self.device)
            theta = self.model(inputs)
            solution = solve_vsp(theta.unsqueeze(0), graph)
        return solution.squeeze()
