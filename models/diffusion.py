import torch
import torch.nn.functional as F


class DiffusionModel:
    def __init__(self, unet, vae, text_encoder, config):
        self.unet = unet
        self.vae = vae
        self.text_encoder = text_encoder
        self.config = config
        self.device = config.device

        # 预计算beta schedule
        self.num_train_timesteps = config.num_train_timesteps
        self.betas = torch.linspace(0.0001, 0.02, self.num_train_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # 移动到设备
        self.betas = self.betas.to(self.device)
        self.alphas_cumprod = self.alphas_cumprod.to(self.device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(self.device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(self.device)

    def add_noise(self, x_start, t):
        """前向加噪"""
        noise = torch.randn_like(x_start)
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        x_noisy = sqrt_alpha * x_start + sqrt_one_minus_alpha * noise
        return x_noisy, noise

    def get_loss(self, x_start, text_tokens, t):
        """计算扩散损失"""
        # 编码文本
        with torch.no_grad():
            text_embeddings = self.text_encoder(text_tokens)[0]  # [B, 77, 768]

        # 编码图像到latent空间
        with torch.no_grad():
            latents = self.vae.encode(x_start).latent_dist.sample() * 0.18215  # VAE缩放因子
        # 加噪
        x_noisy, noise = self.add_noise(latents, t)
        # 预测噪声
        noise_pred = self.unet(x_noisy, t, text_embeddings)

        # MSE损失
        loss = F.mse_loss(noise_pred, noise)
        return loss