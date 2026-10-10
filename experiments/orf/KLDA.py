import math
from collections import defaultdict
import torch

class KLDA:
    """KLDA with fixed Random Fourier Features or Orthogonal Random Features."""
    def __init__(self, num_classes, d, D, sigma, seed, device, mapping="rff", ridge=1e-3):
        if mapping not in ("rff", "orf") or d < 1 or D < 1 or sigma <= 0 or ridge <= 0:
            raise ValueError("Invalid feature mapping or hyperparameters")
        self.num_classes, self.d, self.D = num_classes, d, D
        self.device = torch.device(device)
        self.mapping, self.ridge = mapping, ridge
        rng = torch.Generator(device="cpu").manual_seed(seed)
        if mapping == "rff":
            omega = torch.randn(d, D, generator=rng)
        else:
            blocks = []
            for _ in range(math.ceil(D / d)):
                A = torch.randn(d, d, generator=rng)
                Q, R = torch.linalg.qr(A)
                Q = Q * torch.where(torch.diagonal(R) >= 0, 1.0, -1.0).unsqueeze(0)
                radii = torch.linalg.vector_norm(torch.randn(d, d, generator=rng), dim=0)
                blocks.append(Q * radii.unsqueeze(0))
            omega = torch.cat(blocks, dim=1)[:, :D]
        self.omega = (omega * math.sqrt(2 * sigma)).to(self.device)
        self.phase = (torch.rand(D, generator=rng) * (2 * math.pi)).to(self.device)
        self.means = defaultdict(lambda: torch.zeros(D, device=self.device))
        self.counts = defaultdict(int)
        self.scatter = torch.zeros(D, D, device=self.device)
        self.weights = self.bias = None

    def transform(self, X):
        return math.sqrt(2.0 / self.D) * torch.cos(X @ self.omega + self.phase)

    @torch.no_grad()
    def batch_update(self, X, y):
        X = X.to(self.device)
        if X.ndim != 2 or X.shape[1] != self.d or X.shape[0] == 0:
            raise ValueError("Expected nonempty (batch, d) input")
        if not 0 <= y < self.num_classes:
            raise ValueError("Invalid class index")
        F = self.transform(X)
        n = F.shape[0]
        mu = F.mean(dim=0)
        delta_F = F - mu
        batch_scatter = delta_F.T @ delta_F
        old = self.counts[y]
        if old:
            delta = mu - self.means[y]
            self.scatter += batch_scatter + (old * n / (old + n)) * torch.outer(delta, delta)
            self.means[y] += (n / (old + n)) * delta
        else:
            self.scatter += batch_scatter
            self.means[y] = mu
        self.counts[y] += n
        self.weights = self.bias = None

    @torch.no_grad()
    def fit(self):
        if any(self.counts[i] == 0 for i in range(self.num_classes)):
            raise ValueError("All classes must have training samples")
        means = torch.stack([self.means[i] for i in range(self.num_classes)])
        matrix = self.scatter.clone()
        matrix.diagonal().add_(self.ridge)
        self.weights = torch.linalg.solve(matrix, means.T)
        self.bias = -0.5 * (means.T * self.weights).sum(dim=0)

    @torch.no_grad()
    def get_logits(self, x):
        if self.weights is None:
            raise RuntimeError("Call fit() before prediction")
        if x.ndim == 1:
            x = x.unsqueeze(0)
        return self.transform(x.to(self.device)) @ self.weights + self.bias

class KLDA_E:
    def __init__(self, num_classes, d, D, sigma, num_ensembles, seed, device=None, mapping="rff", ridge=1e-3):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if num_ensembles < 1:
            raise ValueError("num_ensembles must be positive")
        self.models = [KLDA(num_classes, d, D, sigma, seed + i, device, mapping, ridge) for i in range(num_ensembles)]

    def batch_update(self, X, y):
        for model in self.models:
            model.batch_update(X, y)

    def fit(self):
        for model in self.models:
            model.fit()

    @torch.no_grad()
    def predict(self, x):
        probabilities = sum((torch.softmax(m.get_logits(x), dim=-1) for m in self.models))
        return int(probabilities.argmax(dim=-1).item())
