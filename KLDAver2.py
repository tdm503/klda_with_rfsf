import torch
from collections import defaultdict
import ksig

class KLDA:
    def __init__(self, num_classes, d, D_rff,n_levels, sigma, seed, device):
        self.num_classes = num_classes
        self.d = d
        self.D_rff = D_rff
        self.n_levels = n_levels
        self.D = n_levels * D_rff + 1
        self.sigma = sigma
        self.seed = seed
        self.device = device
        torch.manual_seed(self.seed)
        self.class_means = defaultdict(lambda: torch.zeros(self.D, device=self.device))
        self.class_counts = defaultdict(int)

        self.sigma = torch.zeros((self.D, self.D), device=self.device)
        self.sigma_inv = None
        self.class_mean_matrix = None
        self.rfsf_kernel = ksig.kernels.SignatureFeatures(
            n_levels=self.n_levels,
            static_features=ksig.static.features.RandomFourierFeatures(n_components=D_rff),
            projection=ksig.projections.TensorizedRandomProjection(n_components=D_rff)
        )

    def _compute_rfsf(self, X):
        if isinstance(X, torch.Tensor):
            X_np = X.cpu().numpy()
        else:
            X_np = X
        
        if not hasattr(self.rfsf_kernel, 'is_fitted'):
            self.rfsf_kernel.fit(X_np)
        return torch.from_numpy(self.rfsf_kernel.transform(X_np)).to(self.device)

    def batch_update(self, X, y):
        X = X.to(self.device)
        n = X.size(0)
        phi_X = self._compute_rfsf(X)  # Shape: (n, D)
        phi_X_mean = torch.mean(phi_X, dim=0)

        # Update class mean
        previous_count = self.class_counts[y]
        self.class_counts[y] += n
        self.class_means[y] = (self.class_means[y] * previous_count + phi_X_mean * n) / self.class_counts[y]

        # Update covariance matrix sigma
        centered_phi_X = phi_X - self.class_means[y]
        self.sigma += centered_phi_X.t() @ centered_phi_X

    def fit(self):
        self.sigma_inv = torch.pinverse(self.sigma)
        self.class_mean_matrix = torch.stack([self.class_means[i] for i in range(self.num_classes)]).to(self.device)

    def get_logits(self, x):
        x = x.to(self.device)
        phi_x = self._compute_rfsf(x.unsqueeze(0))  
        diff = self.class_mean_matrix - phi_x      
        # Note:
        # Mahalanobis distance is used here instead of the original LDA because it provides a more intuitive
        # measure of distance. Under reasonable assumptions, Mahalanobis distance can be proven to be equivalent to LDA.
        
        logits = -torch.sum((diff @ self.sigma_inv) * diff, dim=1)  
        return logits

class KLDA_E:
    def __init__(self, num_classes, d, D, sigma, num_ensembles, seed, device=None):

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_ensembles = num_ensembles
        self.device = device
        self.models = [
            KLDA(num_classes, d, D, sigma, seed=seed+i, device=self.device) for i in range(self.num_ensembles)
        ]

    def batch_update(self, X, y):

        for model in self.models:
            model.batch_update(X, y)

    def fit(self):
        for model in self.models:
            model.fit()

    def predict(self, x):
        total_probabilities = torch.zeros(self.models[0].num_classes, device=self.device)

        for model in self.models:
            logits = model.get_logits(x)
            probs = torch.softmax(logits, dim=0)
            total_probabilities += probs

        predicted_class = torch.argmax(total_probabilities).item()
        return predicted_class
