import torch
import torch.optim as optim
from tqdm import tqdm
from transformers import CLIPTextModel
from diffusers import AutoencoderKL
import torch.nn.functional as F

from config import Config
from models.unet import UNet
from models.diffusion import DiffusionModel
from utils.dataset import get_dataloader


def train():
    config = Config()
    device = config.device

    # 加载预训练模型
    print("加载预训练VAE和CLIP...")
    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae", use_safetensors=True).to(device)
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # 初始化U-Net
    unet = UNet(
        in_channels=config.unet_in_channels,
        out_channels=config.unet_out_channels,
        text_embed_dim=config.text_embed_dim
    ).to(device)

    # 初始化扩散模型
    diffusion = DiffusionModel(unet, vae, text_encoder, config)

    # 数据加载器
    dataloader, tokenizer = get_dataloader(config)

    # 优化器
    optimizer = optim.AdamW(unet.parameters(), lr=config.learning_rate)

    # 训练循环
    print("开始训练...")
    unet.train()
    for epoch in range(config.epochs):
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.epochs}")

        for batch_idx, (images, input_ids) in enumerate(progress_bar):
            images = images.to(device) 
            input_ids = input_ids.to(device)

            # 随机时间步
            t = torch.randint(0, config.num_train_timesteps, (images.shape[0],), device=device)

            # 只计算 unet 梯度，其他冻结
            with torch.no_grad():
                text_embeddings = text_encoder(input_ids)[0]
                latents = vae.encode(images).latent_dist.mode()  

            latents = latents * 0.18215  # VAE缩放因子
            latents.requires_grad_(True)  
            x_noisy, noise = diffusion.add_noise(latents, t)
            noise_pred = unet(x_noisy, t, text_embeddings)

            loss = F.mse_loss(noise_pred, noise)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 清理中间变量 + 缓存
            del x_noisy, noise, noise_pred, latents, text_embeddings
            torch.cuda.empty_cache()

            total_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} 平均损失: {avg_loss:.6f}")

    # 保存模型
    torch.save(unet.state_dict(), config.save_path)
    print(f"模型已保存至 {config.save_path}")


if __name__ == "__main__":
    train()
