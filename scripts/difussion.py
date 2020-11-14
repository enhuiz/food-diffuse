import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import repeat

from .network import Network


class Diffusion(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.network = Network(*args, **kwargs)

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def n_iters(self):
        if not hasattr(self, "betas"):
            raise RuntimeError(
                "No noise schedule is found. Specify your noise schedule "
                "by pushing arguments into `set_new_noise_schedule(...)` method. "
            )

        return len(self.betas)

    def set_new_noise_schedule(self, betas):
        alphas = 1 - betas
        alpha_bars = alphas.cumprod(dim=0)
        sqrt_alpha_bars = alpha_bars.sqrt()

        # put things to device
        self.register_variable_on_device("betas", betas)
        self.register_variable_on_device("alphas", alphas)
        self.register_variable_on_device("alpha_bars", alpha_bars)
        self.register_variable_on_device("sqrt_alpha_bars", sqrt_alpha_bars)

    def register_variable_on_device(self, name, value):
        setattr(self, name, value.to(self.device))

    def sample_continuous_noise_level(self, batch_size):
        s = np.random.randint(0, self.n_iters, batch_size)
        noise_level = torch.tensor(
            [
                np.random.uniform(
                    self.sqrt_alpha_bars[si].item(),
                    self.sqrt_alpha_bars[si - 1].item() if si > 0 else 1,
                )
                for si in s
            ]
        )
        return noise_level

    def compute_loss(self, labels, y0):
        """
        Computes loss between GT Gaussian noise and predicted noise by model from diffusion process.
        :param labels (torch.Tensor): (b)
        :param y0 (torch.Tensor): GT images
        :return loss (torch.Tensor): loss of diffusion model
        """
        batch_size = len(y0)

        noise_level = self.sample_continuous_noise_level(batch_size)
        noise_level = noise_level.to(y0)

        eps = torch.randn_like(y0)

        unsqueezed_noise_level = noise_level[:, None, None, None]
        yt = (
            unsqueezed_noise_level * y0
            + (1.0 - unsqueezed_noise_level ** 2).sqrt() * eps
        )

        # Reconstruct the added noise
        eps_recon = self.network(labels, yt, noise_level)

        loss = F.l1_loss(eps_recon, eps)

        return loss

    @torch.no_grad()
    def sample(self, labels, store_intermediate_states=False):
        batch_size = len(labels)

        ys = [self.network.generate_noise(batch_size).to(self.device)]

        for t in reversed(range(self.n_iters)):
            coef1 = 1 / self.alphas[t].sqrt()
            coef2 = (1 - self.alphas[t]) / (1 - self.alpha_bars[t]).sqrt()

            noise_level = repeat(self.sqrt_alpha_bars[t], "-> b ", b=batch_size)
            noise_level = noise_level.to(self.device)

            eps_recon = self.network(labels, ys[-1], noise_level)

            yt = coef1 * (ys[-1] - coef2 * eps_recon)

            if t > 0:
                eps = torch.randn_like(yt)
                sigma = (
                    (1 - self.alpha_bars[t - 1])
                    / (1 - self.alpha_bars[t])
                    * self.betas[t]
                ).sqrt()
                yt += sigma * eps
            else:
                yt = yt.clamp(-1, 1)

            ys.append(yt)

        return ys if store_intermediate_states else ys[-1]


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    model = Diffusion(
        num_classes=1,
        class_dim=128,
        factors=[2, 2, 2, 2, 1],
        up_pre_dim=384,
        up_post_dim=3,
        up_dims=[256, 256, 128, 64, 64],
        up_dilations=[
            [1, 2, 1, 2],
            [1, 2, 1, 2],
            [1, 2, 4, 8],
            [1, 2, 4, 8],
            [1, 2, 4, 8],
        ],
        down_pre_dim=32,
        down_dims=[64, 64, 128, 256],
        down_dilations=[
            [1, 2, 4],
            [1, 2, 4],
            [1, 2, 4],
            [1, 2, 4],
        ],
    ).cuda()
    model.set_new_noise_schedule(torch.linspace(1e-6, 1e-2, 10))
    print(
        model.compute_loss(
            torch.zeros(2).long().cuda(),
            torch.randn(2, 3, 64, 64).cuda(),
        )
    )
    output = model.sample(
        torch.zeros(2).long().cuda(),
    )

    print(output.shape)
