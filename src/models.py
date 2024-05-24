import torch.nn as nn


class FenchelYoungGLM(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.encoder = nn.Linear(in_features=n_features, out_features=1, bias=False)

    def forward(self, features):
        features = features.to(self.encoder.weight.data.dtype)
        theta = self.encoder(features)
        theta = theta.squeeze(-1)

        return theta


class MLP(nn.Module):
    def __init__(self, n_features, hidden_size):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features=n_features, out_features=hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size, out_features=1),
        )

    def forward(self, features):
        features = features.to(self.encoder[0].weight.data.dtype)
        theta = self.encoder(features)
        theta = theta.squeeze(-1)

        return theta
