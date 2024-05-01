import torch


class Trainer:
    def __init__(self, model, data_loaders, optimizer, criterion, config):
        self.config = config
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model = model
        self.train_loader, self.val_loader, self.test_loader = data_loaders
        self.optimizer = optimizer
        self.criterion = criterion

        self.n_epochs = config["train"]["n_epochs"]

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
            outputs = self.model(inputs, graph)
            loss = self.criterion(outputs, labels).mean()
            loss.backward()
            self.optimizer.step()

    def train(self):
        for i in range(self.n_epochs):
            self.train_epoch(i)
            self.compute_metrics(i)
            self.save_model(i)
