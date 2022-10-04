import torch


class Config:
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据路径
    image_dir = "./dataset/images"
    caption_file = "./dataset/captions.txt"

    # 模型参数
    latent_channels = 4          # VAE压缩后通道数

    image_size = 256     # VAE压缩后图像尺寸
    latent_size = image_size // 8 

    text_embed_dim = 512         # CLIP文本嵌入维度
    unet_in_channels = 4         # 输入通道（latent）
    unet_out_channels = 4        # 输出通道（预测噪声）

    # 训练参数
    batch_size = 8
    epochs = 50
    learning_rate = 5e-5
    num_train_timesteps = 1000   # 扩散步数
    save_path = "./models/sd_toy.pth"

    # 采样参数
    num_inference_steps = 50     # DDIM步数
    guidance_scale = 7.5         # 文本引导强度
