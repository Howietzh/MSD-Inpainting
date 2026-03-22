import json
import yaml
import torch
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import lpips
from torchvision import transforms

from accelerate import PartialState 
from diffusers import StableDiffusionInpaintPipeline, DDIMScheduler
from transformers import CLIPTokenizer
from peft import PeftModel
from torch.utils.data import RandomSampler, DataLoader

from utils.runtime import resolve_model_source, resolve_weight_dtype
from utils.mask_ops import DefectMaskEngine
from dataset.normal_dataset import NormalComponentDataset

class DefectFillPipeline:
    def __init__(self, model_config_path: str, lora_dir: Path, normal_dir: Path, output_dir: Path, mask_engine: DefectMaskEngine, infer_config: dict):
        self.distributed_state = PartialState()
        self.device = self.distributed_state.device
        
        self.normal_dir = normal_dir
        self.output_dir = output_dir
        self.mask_engine = mask_engine
        self.infer_config = infer_config
        
        if self.distributed_state.is_main_process:
            (self.output_dir / "images").mkdir(parents=True, exist_ok=True)
            (self.output_dir / "defect_masks").mkdir(parents=True, exist_ok=True)
            (self.output_dir / "visualizations").mkdir(parents=True, exist_ok=True)
            
        self.metadata_path = self.output_dir / f"metadata_gpu{self.distributed_state.process_index}.jsonl"
        if self.metadata_path.exists():
            self.metadata_path.unlink()
            
        self.distributed_state.wait_for_everyone()
        
        self.defect_class_map = {
            "<flexible_printed_circuit_crack>": 1,
            "<end_face_scratch>": 2,
            "<lens_scratch>": 3,
            "<foreign_particle>": 4
        }
        
        self._build_models(model_config_path, lora_dir)

    def _build_models(self, config_path, lora_dir):
        with open(config_path, "r", encoding="utf-8") as f:
            train_config = yaml.safe_load(f)
            
        if self.distributed_state.is_main_process:
            print("🚀 初始化 Stable Diffusion 与 LoRA 权重...")
            
        model_source = resolve_model_source(train_config["paths"])
        self.weight_dtype = resolve_weight_dtype(train_config.get("training", {}).get("mixed_precision", "no"))
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            model_source,
            torch_dtype=self.weight_dtype,
            local_files_only=True,
        )
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        
        # 1. 先加载包含新概念的 Tokenizer
        self.pipe.tokenizer = CLIPTokenizer.from_pretrained(str(lora_dir / "tokenizer"))
        
        # ==========================================
        # 【关键修正】必须先扩容 Embedding 层，再挂载 LoRA
        # ==========================================
        new_vocab_size = len(self.pipe.tokenizer)
        self.pipe.text_encoder.resize_token_embeddings(new_vocab_size)
        
        # 2. 现在再挂载 LoRA 权重 (它们是基于扩容后的维度训练的)
        self.pipe.text_encoder = PeftModel.from_pretrained(
            self.pipe.text_encoder, 
            str(lora_dir / "text_encoder_lora")
        )
        self.pipe.unet = PeftModel.from_pretrained(
            self.pipe.unet, 
            str(lora_dir / "unet_lora")
        )
        
        self.pipe.to(self.device)
        self.pipe.set_progress_bar_config(disable=True) 
        self.pipe.enable_attention_slicing()
        
        if self.distributed_state.is_main_process:
            print(f"🧠 已扩容词表至 {new_vocab_size} 并初始化 LPIPS 模型...")
        self.lpips_vgg = lpips.LPIPS(net='vgg').to(self.device)

    def _build_visualization(self, init_img_np, defect_mask_np, generated_img_pil):
        mask_bool = defect_mask_np > 127
        overlay = init_img_np.copy()
        overlay[mask_bool] = (
            0.6 * overlay[mask_bool] + 0.4 * np.array([255, 80, 80], dtype=np.float32)
        ).astype(np.uint8)

        generated_img_np = np.array(generated_img_pil)
        canvas = np.concatenate([init_img_np, overlay, generated_img_np], axis=1)
        return Image.fromarray(canvas)
        
    def _calculate_lpips(self, img_orig, img_gen, mask_np):
        coords = cv2.findNonZero(mask_np)
        if coords is None: return 0.0
        x, y, w, h = cv2.boundingRect(coords)
        
        # 1. 裁剪出缺陷区域
        orig_crop = img_orig[y:y+h, x:x+w]
        gen_crop = img_gen[y:y+h, x:x+w]
        
        # =================================================================
        # 【关键修正】防止由于裁剪区域太小（如宽度 < 32）导致 VGG 下采样归零崩溃
        # 统一缩放到 224x224 像素 (VGG 推荐的标准尺寸)
        # =================================================================
        if h > 0 and w > 0:
            # 使用双三次插值进行缩放以保持平滑
            orig_crop = cv2.resize(orig_crop, (224, 224), interpolation=cv2.INTER_CUBIC)
            gen_crop = cv2.resize(gen_crop, (224, 224), interpolation=cv2.INTER_CUBIC)
        else:
            return 0.0
        
        # 2. 转换为 Tensor 并归一化
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        orig_tensor = transform(orig_crop).unsqueeze(0).to(self.device)
        gen_tensor = transform(gen_crop).unsqueeze(0).to(self.device)
        
        # 3. 计算得分
        with torch.no_grad():
            score = self.lpips_vgg(orig_tensor, gen_tensor)
            
        return score.item()

    def execute_tasks(self, tasks):
        if self.distributed_state.is_main_process:
            self.mask_engine.load_or_compute_stats(tasks)
        self.distributed_state.wait_for_everyone()
        
        with open(self.mask_engine.cache_file, "r", encoding="utf-8") as f:
            self.mask_engine.stats_cache = json.load(f)
            
        batch_size = self.infer_config.get("batch_size", 4)
        num_lfs_samples = self.infer_config.get("num_lfs_samples", 8)
        num_inference_steps = self.infer_config.get("num_inference_steps", 50)
        guidance_scale = self.infer_config.get("guidance_scale", 7.5)
        negative_prompt = self.infer_config.get("negative_prompt", "blurry, smooth, unrealistic, artifacts")
        base_seed = self.infer_config.get("base_seed", 42)
        
        for task_idx, task in enumerate(tasks):
            defect_token, comp_token, target_count = task["defect"], task["comp"], task["count"]
            semantic_class_id = self.defect_class_map.get(defect_token, 255)
            
            full_task_indices = list(range(target_count))
            with self.distributed_state.split_between_processes(full_task_indices) as local_indices:
                local_target_count = len(local_indices)
                if local_target_count == 0: continue
                
                # =========================================================
                # 【核心重构】使用 Dataset 和 DataLoader 代理数据加载
                # =========================================================
                dataset = NormalComponentDataset(
                    data_dir=str(self.normal_dir), 
                    size=512, 
                    target_comp=comp_token
                )
                
                if len(dataset) == 0:
                    if self.distributed_state.is_main_process:
                        print(f"❌ 找不到对应 {comp_token} 的正常样本，跳过该任务！")
                    continue
                
                if self.distributed_state.is_main_process:
                    print(f"\n🎯 任务执行: {defect_token} on {comp_token} (当前 GPU 分配 {local_target_count} 张)")
                    
                # 随机采样器：开启放回抽样(replacement)，并严格指定抽样总数为该卡分配的指标数
                sampler = RandomSampler(dataset, replacement=True, num_samples=local_target_count)
                dataloader = DataLoader(
                    dataset, 
                    batch_size=batch_size, 
                    sampler=sampler, 
                    num_workers=4,       # 开启多线程预读取图
                    pin_memory=True      # 加速向 GPU 转移
                )
                
                progress_bar = tqdm(total=local_target_count, disable=not self.distributed_state.is_local_main_process, desc=f"生成进度")
                global_local_idx = local_indices[0] # 维持文件名的绝对唯一性
                
                # 直接遍历 Dataloader 输出的 Batch
                for batch in dataloader:
                    current_b_size = batch["pixel_values"].shape[0]
                    
                    # 1. 直接复用 DataLoader 产出的 Tensor (已归一化到 [-1, 1])
                    pixel_values = batch["pixel_values"]
                    # SD Inpainting pipeline 要求 image tensor 在 [0, 1] 区间
                    init_imgs_tensor = (pixel_values * 0.5 + 0.5).to(self.device, dtype=self.weight_dtype)
                    
                    # 为了给后续的 LPIPS 计算和保存，恢复出 RGB Numpy
                    init_imgs_np = (init_imgs_tensor.cpu().permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
                    
                    # 取出组件掩码并缩放为 OpenCV 适用的 [0, 255] Numpy Array
                    comp_masks_np = (batch["mask_values"].squeeze(1).numpy() * 255).astype(np.uint8)
                    
                    defect_masks_np = []
                    defect_masks_tensor_list = []
                    
                    # 2. 为当前 Batch 生成动态缺陷掩码
                    for b in range(current_b_size):
                        valid_found = False
                        while not valid_found:
                            defect_mask_np = self.mask_engine.generate_dynamic_mask(comp_masks_np[b], defect_token)
                            if defect_mask_np is not None and cv2.countNonZero(defect_mask_np) > 0:
                                valid_found = True
                        
                        defect_masks_np.append(defect_mask_np)
                        # 将 OpenCV 生成的掩码转为 Pipeline 需要的 [0, 1] Tensor
                        defect_tensor = torch.from_numpy(defect_mask_np).float() / 255.0
                        defect_masks_tensor_list.append(defect_tensor.unsqueeze(0)) # Shape: [1, 512, 512]
                    
                    # 合并成掩码批次: [B, 1, 512, 512]
                    mask_images_tensor = torch.stack(defect_masks_tensor_list).to(self.device, dtype=self.weight_dtype)
                    
                    prompts = [f"a photo of {comp_token} with {defect_token}"] * current_b_size
                    neg_prompts = [negative_prompt] * current_b_size
                    best_scores = [-1.0] * current_b_size
                    best_images = [None] * current_b_size
                    
                    # 3. 开始 LFS 的 8 样本优选循环
                    for s_idx in range(num_lfs_samples):
                        generators = [
                            torch.Generator(device=self.device).manual_seed(
                                base_seed + (global_local_idx + b) * 1000 + s_idx + self.distributed_state.process_index * 100000
                            ) for b in range(current_b_size)
                        ]
                        
                        # Pipeline 直接接收 Torch Tensor，彻底消除与 PIL 的反复转换耗时
                        out_images = self.pipe(
                            prompt=prompts, 
                            negative_prompt=neg_prompts,
                            image=init_imgs_tensor,      # 【原生 Tensor 输入】
                            mask_image=mask_images_tensor, # 【原生 Tensor 输入】
                            num_inference_steps=num_inference_steps, 
                            guidance_scale=guidance_scale, 
                            generator=generators
                        ).images
                        
                        for b in range(current_b_size):
                            score = self._calculate_lpips(init_imgs_np[b], np.array(out_images[b]), defect_masks_np[b])
                            if score > best_scores[b]:
                                best_scores[b] = score
                                best_images[b] = out_images[b]
                                
                    # 4. 保存结果与 Semantic Mask
                    for b in range(current_b_size):
                        semantic_mask_np = np.zeros_like(defect_masks_np[b], dtype=np.uint8)
                        semantic_mask_np[defect_masks_np[b] > 127] = semantic_class_id
                        semantic_mask_pil = Image.fromarray(semantic_mask_np, mode='L')
                        visualization_pil = self._build_visualization(
                            init_imgs_np[b],
                            defect_masks_np[b],
                            best_images[b],
                        )
                                
                        base_name = f"{defect_token.strip('<>')}_{comp_token.strip('<>')}_Task{task_idx}_GPU{self.distributed_state.process_index}_{global_local_idx:04d}"
                        img_save_path = self.output_dir / "images" / f"{base_name}.png"
                        mask_save_path = self.output_dir / "defect_masks" / f"{base_name}_defect_mask.png"
                        vis_save_path = self.output_dir / "visualizations" / f"{base_name}_viz.png"
                        
                        best_images[b].save(img_save_path)
                        semantic_mask_pil.save(mask_save_path)
                        visualization_pil.save(vis_save_path)
                        
                        record = {
                            "image_path": f"images/{img_save_path.name}",
                            "defect_mask_path": f"defect_masks/{mask_save_path.name}",
                            "visualization_path": f"visualizations/{vis_save_path.name}",
                            "prompt": prompts[b],
                            "object_token": comp_token,
                            "defect_token": defect_token
                        }
                        with open(self.metadata_path, "a", encoding="utf-8") as mf:
                            mf.write(json.dumps(record) + "\n")
                        
                        global_local_idx += 1
                        
                    progress_bar.update(current_b_size)
                
                progress_bar.close()
