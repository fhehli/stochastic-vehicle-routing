import torch.nn as nn
from torch.nn.functional import normalize


class FenchelYoungGLM(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.encoder = nn.Linear(in_features=n_features, out_features=1, bias=False)

    def forward(self, features):
        features = features.to(self.encoder.weight.data.dtype)
        normalized_features = normalize(features)
        theta = self.encoder(normalized_features)
        theta = theta.squeeze(-1)

        return theta
