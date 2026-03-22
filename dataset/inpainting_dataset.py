import json
import torch
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import CLIPTokenizer

class DefectFillDataset(Dataset):
    def __init__(self, data_dir: str, tokenizer: CLIPTokenizer, size: int = 512):
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.size = size
        
        self.metadata = []
        metadata_path = self.data_dir / "metadata.jsonl"
        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                self.metadata.append(json.loads(line.strip()))
                
        # RGB 图像预处理 (缩放 + 归一化到 [-1, 1])
        self.image_transforms = transforms.Compose([
            transforms.Resize((self.size, self.size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
        # 掩码预处理 (必须使用最近邻插值，保证二值纯粹性)
        self.mask_transforms = transforms.Compose([
            transforms.Resize((self.size, self.size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        item = self.metadata[index]
        
        # 1. 读取原图
        img_path = self.data_dir / item["image_path"]
        image = Image.open(img_path).convert("RGB")
        pixel_values = self.image_transforms(image)
        
        # 2. 读取缺陷掩码
        mask_path = self.data_dir / item["defect_mask_path"]
        mask = Image.open(mask_path).convert("L")
        mask_values = self.mask_transforms(mask)
        mask_values = (mask_values > 0.5).float() # 严格二值化

        component_mask_path = self.data_dir / item["component_mask_path"]
        component_mask = Image.open(component_mask_path).convert("L")
        component_mask_values = self.mask_transforms(component_mask)
        component_mask_values = (component_mask_values > 0.5).float()
        
        # 3. 读取显式的对象 token / 缺陷 token
        prompt = item["prompt"]
        object_token = item["object_token"]
        defect_token = item["defect_token"]

        # 4. 保留原 prompt 的 tokenization，供兼容或调试使用
        input_ids = self.tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids.squeeze(0)
        
        return {
            "pixel_values": pixel_values,
            "mask_values": mask_values,
            # 为后续组件监督/组件可视化预留，当前训练主链路暂未消费。
            "component_mask_values": component_mask_values,
            "input_ids": input_ids,
            "object_token": object_token,
            "defect_token": defect_token,
            "defect_prompt": f"a photo of {defect_token}",
            "object_prompt": f"a photo of {object_token} with {defect_token}",
        }
