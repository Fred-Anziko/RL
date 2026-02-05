import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class BayesianLinear(nn.Module):
    """
    Bayesian Linear Layer with Reparameterization Trick.
    Learns weight distribution (Gaussian) instead of point estimates.
    input: in_features
    output: out_features
    """
    def __init__(self, in_features, out_features, prior_sigma=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_sigma = prior_sigma

        # Weight parameters
        self.weight_mu = nn.Parameter(torch.zeros(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.zeros(out_features, in_features))

        # Bias parameters
        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_rho = nn.Parameter(torch.zeros(out_features))

        # Initialize parameters
        # He initialization for mu
        nn.init.kaiming_normal_(self.weight_mu, mode='fan_in', nonlinearity='relu')
        # Initialize rho such that sigma starts small (e.g., -3 gives sigma approx 0.05)
        nn.init.constant_(self.weight_rho, -3.0)
        nn.init.constant_(self.bias_mu, 0.0)
        nn.init.constant_(self.bias_rho, -3.0)

    def forward(self, x):
        # Calculate sigma from rho: sigma = softplus(rho)
        weight_sigma = F.softplus(self.weight_rho)
        bias_sigma = F.softplus(self.bias_rho)

        if self.training:
            # Reparameterization trick
            # w = mu + sigma * epsilon
            epsilon_weight = torch.randn_like(self.weight_mu)
            epsilon_bias = torch.randn_like(self.bias_mu)

            weight = self.weight_mu + weight_sigma * epsilon_weight
            bias = self.bias_mu + bias_sigma * epsilon_bias
        else:
            # During inference, use just the mean
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(x, weight, bias)

    def kl_divergence(self):
        """
        Compute KL Divergence between Posterior and Prior.
        Prior is assumed to be Gaussian N(0, prior_sigma).
        """
        weight_sigma = F.softplus(self.weight_rho)
        bias_sigma = F.softplus(self.bias_rho)

        # KL for weights
        # KL(N(mu, sigma) || N(0, prior_sigma))
        # = log(prior_sigma/sigma) + (sigma^2 + mu^2)/(2*prior_sigma^2) - 0.5
        kl_weight = (
            torch.log(self.prior_sigma / weight_sigma)
            + (weight_sigma**2 + self.weight_mu**2) / (2 * self.prior_sigma**2)
            - 0.5
        )

        # KL for bias
        kl_bias = (
            torch.log(self.prior_sigma / bias_sigma)
            + (bias_sigma**2 + self.bias_mu**2) / (2 * self.prior_sigma**2)
            - 0.5
        )

        return kl_weight.sum() + kl_bias.sum()
