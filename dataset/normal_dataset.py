import json
import torch
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class NormalComponentDataset(Dataset):
    def __init__(self, data_dir: str, size: int = 512, target_comp: str = None):
        self.data_dir = Path(data_dir)
        self.size = size
        
        self.metadata = []
        metadata_path = self.data_dir / "metadata.jsonl"
        
        if not metadata_path.exists():
            raise FileNotFoundError(f"❌ 找不到 metadata 文件: {metadata_path}")
            
        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line.strip())
                # 【新增】过滤功能：如果指定了组件，只加载对应的正常图像
                if target_comp is None or item.get("object_token") == target_comp:
                    self.metadata.append(item)
                
        self.image_transforms = transforms.Compose([
            transforms.Resize((self.size, self.size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) 
        ])
        
        self.mask_transforms = transforms.Compose([
            transforms.Resize((self.size, self.size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        item = self.metadata[index]
        
        img_path = self.data_dir / item["image_path"]
        image = Image.open(img_path).convert("RGB")
        pixel_values = self.image_transforms(image)
        
        mask_path = self.data_dir / item["component_mask_path"]
        mask = Image.open(mask_path).convert("L")
        mask_values = self.mask_transforms(mask)
        mask_values = (mask_values > 0.5).float()
        
        return {
            "pixel_values": pixel_values,
            "mask_values": mask_values,
            "object_token": item.get("object_token", ""),
            "prompt": item["prompt"],
            "image_path": item["image_path"],
            "component_mask_path": item["component_mask_path"],
        }
