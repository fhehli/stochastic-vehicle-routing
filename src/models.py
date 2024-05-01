import torch.nn as nn

from src.VSPSolver import VSPSolver


class FenchelYoungGLM(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.encoder = nn.Linear(in_features=n_features, out_features=1, bias=False)
        self.solver = VSPSolver()

    def forward(self, features, graph):
        features = features.to(self.encoder.weight.data.dtype)
        theta = self.encoder(features)
        return self.solver.solve(theta, graph)
