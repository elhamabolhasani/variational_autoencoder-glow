import numpy as np
import torch.nn as nn
import torch.nn.utils as utils
from torch.distributions import LowRankMultivariateNormal, Independent, Normal
import torch


def bits_per_dim(x, nll):
    """Get the bits per dimension implied by using model with `loss`
    for compressing `x`, assuming each entry can take on `k` discrete values.

    Args:
        x (torch.Tensor): Input to the model. Just used for dimensions.
        nll (torch.Tensor): Scalar negative log-likelihood loss tensor.

    Returns:
        bpd (torch.Tensor): Bits per dimension implied if compressing `x`.
    """
    dim = np.prod(x.size()[1:])
    bpd = nll / (np.log(2) * dim)

    return bpd


def clip_grad_norm(optimizer, max_norm, norm_type=2):
    """Clip the norm of the gradients for all parameters under `optimizer`.

    Args:
        optimizer (torch.optim.Optimizer):
        max_norm (float): The maximum allowable norm of gradients.
        norm_type (int): The type of norm to use in computing gradient norms.
    """
    for group in optimizer.param_groups:
        utils.clip_grad_norm_(group['params'], max_norm, norm_type)


class NLLLoss(nn.Module):
    """Negative log-likelihood loss assuming isotropic gaussian with unit norm.

    Args:
        k (int or float): Number of discrete values in each input dimension.
            E.g., `k` is 256 for natural images.

    See Also:
        Equation (3) in the RealNVP paper: https://arxiv.org/abs/1605.08803
    """

    def __init__(self, k=256, color_channels=3):
        super(NLLLoss, self).__init__()
        self.k = k
        self.color_channels = color_channels

    def forward(self, z, sldj, mu_d, logvar_d):
        # vae both lr
        # sigma = torch.exp(logvar_d) + 0.001
        # dist = LowRankMultivariateNormal(mu_d, u_d.view(-1, u_d.shape[1], 1), sigma)
        # d = dist.log_prob(z.view(-1, z.shape[2] * z.shape[2] * self.color_channels))

        # vae both n
        dist = Independent(Normal(loc=mu_d, scale=torch.exp(logvar_d)), 1)
        d = dist.log_prob(z.view(-1, z.shape[2] * z.shape[2] * self.color_channels))

        # prior_ll = -0.5 * (z ** 2 + np.log(2 * np.pi))  # TODO = d
        # prior_ll = prior_ll.flatten(1).sum(-1) \
        #     - np.log(self.k) * np.prod(z.size()[1:])
        # ll = prior_ll + sldj
                                                                    # prior_ll -> size (batch_size, 1)
        ll = sldj + d  # TODO = remove
        nll = -ll.mean()

        return nll
