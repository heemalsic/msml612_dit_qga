# diffusion.py
import torch
import torch.nn.functional as F


class Diffusion:
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02, device="cuda"):
        self.T = timesteps
        self.device = device

        betas = torch.linspace(beta_start, beta_end, self.T, device=device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=device), alphas_cumprod[:-1]])

        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        self.posterior_variance = betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - alphas_cumprod)

    def q_sample(self, x0, t, noise=None):
        """
        x_t = sqrt(a_bar_t)*x0 + sqrt(1-a_bar_t)*eps
        """
        if noise is None:
            noise = torch.randn_like(x0)
        s1 = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        s2 = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return s1 * x0 + s2 * noise, noise

    def p_losses(self, model, x0, t, y):
        x_t, noise = self.q_sample(x0, t)
        pred = model(x_t, t, y)
        return F.mse_loss(pred, noise)

    @torch.no_grad()
    @torch.no_grad()
    def ddim_sample(self, model, shape, y, steps=50, eta=0.0):
        device = self.device
        B = shape[0]
        x = torch.randn(shape, device=device)

        # Proper timestep schedule
        step_ratio = self.T // steps
        timesteps = list(range(0, self.T, step_ratio))
        timesteps = timesteps[::-1]  # descending

        for i in range(len(timesteps)):
            t = torch.full((B,), timesteps[i], device=device, dtype=torch.long)

            eps = model(x, t, y)

            alpha_bar = self.alphas_cumprod[t].view(-1, 1, 1, 1)

            if i == len(timesteps) - 1:
                alpha_bar_prev = torch.ones_like(alpha_bar)
            else:
                t_prev = torch.full((B,), timesteps[i + 1], device=device, dtype=torch.long)
                alpha_bar_prev = self.alphas_cumprod[t_prev].view(-1, 1, 1, 1)

            x0 = (x - torch.sqrt(1 - alpha_bar) * eps) / torch.sqrt(alpha_bar)

            sigma = eta * torch.sqrt(
                (1 - alpha_bar_prev) / (1 - alpha_bar) *
                (1 - alpha_bar / alpha_bar_prev)
            )

            noise = torch.randn_like(x) if eta > 0 else 0.0
            dir_xt = torch.sqrt(1 - alpha_bar_prev - sigma**2) * eps

            x = torch.sqrt(alpha_bar_prev) * x0 + dir_xt + sigma * noise

        return x.clamp(-1, 1)

