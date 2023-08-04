import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

from models.forward_diffusion import ForwardDiffusion
from models.reverse_diffusion import Unet

class Generator:
    def __init__(self, model: Unet, forward_diffusion: ForwardDiffusion):
        self.model = model.eval()
        self.forward_diffusion = forward_diffusion.eval()
        self.timesteps = forward_diffusion.timesteps

    @torch.no_grad()
    def sample(self, x: torch.Tensor, intermediate_imgs: bool = False):

        imgs = []

        # reversing diffusion process
        for i in tqdm(reversed(0, self.num_steps)):
            # append intermediate results
            if intermediate_imgs:
                imgs.append(x)

            # generating timestep tensor of size (batch_size, )
            t = (torch.ones(x.shape[0], device=x.device, dtype=torch.long) * i)

            # predict noise
            predicted_noise = self.model(x, t)

            # get params diffusion params for timestep
            betas_t = self.forward_diffusion.betas[t][:, None, None, None]
            sqrt_one_minus_alphas_cumprod_t = self.forward_diffusion.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
            sqrt_recip_alphas_t = self.forward_diffusion.sqrt_recip_alphas[t][:, None, None, None]
            posterior_variance = self.forward_diffusion.posterior_variance[t][:, None, None, None]

            if i == 0:
                noise = torch.randn_like(x, device=x.device)
            else:
                noise = torch.zeros_like(x, device=x.device)
            
            x = sqrt_recip_alphas_t * (x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t) + torch.sqrt(posterior_variance) * noise

        imgs.append(x)

        return imgs