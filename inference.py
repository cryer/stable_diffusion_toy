import torch
import matplotlib.pyplot as plt
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL

from config import Config
from models.unet import UNet
from models.diffusion import DiffusionModel
from utils.ddim import DDIMSampler


def inference(prompt):
    torch.cuda.empty_cache()  # 清理缓存
    torch.set_grad_enabled(False)  # 确保不计算梯度
    config = Config()
    device = config.device

    # 加载预训练模型
    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae").to(device)
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    # 初始化并加载训练好的U-Net
    unet = UNet(
        in_channels=config.unet_in_channels,
        out_channels=config.unet_out_channels,
        text_embed_dim=config.text_embed_dim
    ).to(device)

    unet.load_state_dict(torch.load(config.save_path, map_location=device))
    unet.eval()    
    # 初始化扩散模型和采样器
    diffusion = DiffusionModel(unet, vae, text_encoder, config)
    sampler = DDIMSampler(diffusion)

    # 编码文本
    tokens = tokenizer(
        prompt,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt"
    ).input_ids.to(device)

    # 采样生成
    print(f"正在生成提示词: {prompt}")
    images = sampler.sample(
        tokens,
        batch_size=1,
        guidance_scale=config.guidance_scale,
        num_steps=config.num_inference_steps）
    )

    # 显示结果
    image = images[0].permute(1, 2, 0).cpu().float().numpy()
    plt.imshow(image)
    plt.title(prompt)
    plt.axis('off')
    plt.show()

    return image

if __name__ == "__main__":
    prompt = "a red car"
    inference(prompt)
