import torch


class Trainer:
    def __init__(self, model, data_loaders, optimizer, criterion, config):
        self.config = config
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model = model
        self.train_loader, self.val_loader = data_loaders
        self.optimizer = optimizer
        self.criterion = criterion

    def compute_metrics(self, i):
        raise NotImplementedError

    def save_model(self, i):
        raise NotImplementedError

    def train_epoch(self, i):
        self.model.train()
        for inputs, labels in self.data_loader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels).mean()
            loss.backward()
            self.optimizer.step()

    def train(self):
        for i in range(self.n_epochs):
            self.train_epoch(i)
            self.compute_metrics(i)
            self.save_model(i)
