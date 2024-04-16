import torch.nn as nn

from deterministic import VSPSolver


class FenchelYoungGLM(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.n_features = n_features
        self.encoder = nn.Linear(in_features=n_features, out_features=1, bias=False)

        self.solver = VSPSolver()

    def forward(self, x):
        theta = self.encoder(x)
        return self.solver.solve(theta)
