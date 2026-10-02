"""Small MLP world model predicting the one-step state delta."""

import numpy as np
import torch
import torch.nn as nn


class DeltaMLP(nn.Module):
    """f: R^2 -> R^2 approximating x_{t+1} - x_t."""

    def __init__(self, hidden=(64, 64)):
        super().__init__()
        layers, d = [], 2
        for h in hidden:
            layers += [nn.Linear(d, h), nn.Tanh()]
            d = h
        layers.append(nn.Linear(d, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def model_rollout(model, x0, n_steps):
    """Recursive rollout of the learned one-step model (no noise at query time).

    x0: (B, 2) float64 numpy.  Returns (n_steps+1, B, 2) float64 numpy.
    The network runs in its training precision (float32); predictions are cast
    to float64 for metric computation (signed vectors kept throughout).
    """
    model.eval()
    with torch.no_grad():
        x = torch.as_tensor(x0, dtype=torch.float32)
        out = [x.numpy().astype(np.float64)]
        for _ in range(n_steps):
            x = x + model(x)
            out.append(x.numpy().astype(np.float64))
    return np.stack(out)
