import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import CLIPTokenizer


class TextImageDataset(Dataset):
    def __init__(self, image_dir, caption_file, tokenizer, image_size=64):
        self.image_dir = image_dir
        self.captions = []
        self.image_names = []

        # 读取caption文件
        with open(caption_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                caption = line.strip()
                image_name = f"{i+1}.jpg"  # 假设图片命名为 1.jpg, 2.jpg...
                image_path = os.path.join(image_dir, image_name)
                if os.path.exists(image_path):
                    self.captions.append(caption)
                    self.image_names.append(image_name)

        self.tokenizer = tokenizer
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # [-1, 1]
        ])

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        # 加载图像
        img_path = os.path.join(self.image_dir, self.image_names[idx])
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)  # [C, H, W]

        # 编码文本
        text = self.captions[idx]
        tokens = self.tokenizer(
            text,
            padding="max_length",
            max_length=77,  # CLIP最大长度
            truncation=True,
            return_tensors="pt"
        )
        input_ids = tokens.input_ids.squeeze(0)  # [77]

        return image, input_ids


def get_dataloader(config):
    # 初始化CLIP tokenizer
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")  
    dataset = TextImageDataset(
        config.image_dir,
        config.caption_file,
        tokenizer,
        image_size=config.image_size  # 原始图像尺寸
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0
    )

    return dataloader, tokenizer