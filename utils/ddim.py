import torch


class DDIMSampler:
    def __init__(self, diffusion_model):
        self.diffusion = diffusion_model
        self.device = diffusion_model.device

    @torch.no_grad()
    def sample(self, text_tokens, batch_size=1, guidance_scale=7.5, num_steps=50):
        """DDIM采样生成图像"""
        unet = self.diffusion.unet
        vae = self.diffusion.vae
        text_encoder = self.diffusion.text_encoder
        config = self.diffusion.config

        # 编码文本
        text_embeddings = text_encoder(text_tokens)[0]  # [B, 77, 768]
        uncond_tokens = torch.zeros_like(text_tokens)
        uncond_embeddings = text_encoder(uncond_tokens)[0]

        # 合并条件与无条件嵌入（CFG）
        context = torch.cat([uncond_embeddings, text_embeddings])

        # 初始化随机噪声
        latents = torch.randn(
            (batch_size, config.latent_channels, config.latent_size, config.latent_size),
            dtype=torch.float32, 
            device=self.device
        )

        # DDIM时间步
        timesteps = torch.linspace(
            self.diffusion.num_train_timesteps - 1, 0, num_steps, dtype=torch.long
        ).to(self.device)

        for i, t in enumerate(timesteps):
            t_batch = t.repeat(batch_size * 2).to(latents.dtype) 

            # 预测噪声
            latent_model_input = torch.cat([latents] * 2)
            noise_pred = unet(latent_model_input, t_batch, context)

            # 分离无条件和条件预测
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # DDIM更新公式 
            alpha_t = self.diffusion.alphas_cumprod[t].to(latents.dtype)
            alpha_t_prev = self.diffusion.alphas_cumprod[timesteps[i+1]].to(latents.dtype) if i < len(timesteps)-1 else torch.tensor(1.0, dtype=latents.dtype, device=self.device)

            # 标准 DDIM 公式（sigma=0）
            pred_x0 = (latents - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
            pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)  

            # 重构 latents
            latents = torch.sqrt(alpha_t_prev) * pred_x0 + torch.sqrt(1 - alpha_t_prev) * noise_pred

            # 清理缓存
            latents = latents.detach()
            del noise_pred, noise_pred_uncond, noise_pred_text, pred_x0
            torch.cuda.empty_cache()

        # 解码latent到图像 
        latents = latents / 0.18215
        images = vae.decode(latents.float()).sample 
        images = (images / 2 + 0.5).clamp(0, 1)  # [-1,1] -> [0,1]

        return images