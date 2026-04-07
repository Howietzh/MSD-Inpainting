import os
import cv2
import json
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from torchvision import transforms

# 导入核心模块
from utils.mask_ops import DefectMaskEngine
from utils.gradio_inference import InteractiveDefectFillEngine
from models.attention_hook import AttentionStore  # 真实的 Attention 拦截器
# =========================================================================
# 1. 全局配置区 (请在运行前修改这里的路径)
# =========================================================================
CONFIG = {
    # 基础目录配置
    "output_dir": "paper_all_assets",
    "workspace_dir": "/home/doctor/tzh/MSD-Inpainting/data/CCM-Defect",
    "train_config": "configs/train_config.yaml",
    "infer_config": "configs/inference_config.yaml",
    
    # 填入您训练好的 LoRA 权重路径
    "lora_weights": "defectfill_lora_weights/increase_text_encoder_learning_rates", 
    
    # 训练流程展示用图 (真实的异常图和掩码)
    "abnormal_img": "defect_train_concept/images/lens_scratch_train_425.png",
    "abnormal_comp_mask": "defect_train_concept/component_masks/lens_scratch_train_425_component_mask.png",
    "abnormal_defect_mask": "defect_train_concept/defect_masks/lens_scratch_train_425_defect_mask.png",
    
    # 推理流程展示用图 (无瑕疵的正常图)
    "normal_img_rel": "images/normal_2_lens.png", # 相对 normal_components 目录的路径
    "normal_comp_mask_rel": "component_masks/normal_2_lens_component_mask.png",
    
    # 目标生成标签
    "comp_token": "<lens>",
    "defect_token": "<lens_scratch>",
    "seed": 100
}

# =========================================================================
# 2. 工具函数
# =========================================================================
def apply_jet_colormap(tensor_or_array):
    """将单通道特征图转换为 Jet 伪彩色热力图 (带百分位数截断、Float32修复和轻微平滑)"""
    if isinstance(tensor_or_array, torch.Tensor):
        arr = tensor_or_array.detach().cpu().numpy()
    else:
        arr = tensor_or_array

    arr = np.asarray(arr).squeeze()
    if arr.ndim != 2:
        raise ValueError(f"apply_jet_colormap expects a 2D heatmap, got shape {arr.shape}.")

    # 强制转为 float32 避免 OpenCV 对 float16 报错
    arr = arr.astype(np.float32)

    # 百分位数截断过滤极端噪点，防止有效特征被压制为黑色
    p_min, p_max = np.percentile(arr, 2), np.percentile(arr, 98)
    arr = np.clip(arr, p_min, p_max)

    if p_max > p_min:
        arr = (arr - p_min) / (p_max - p_min)
    else:
        arr = np.zeros_like(arr)

    # 轻微高斯平滑消除 U-Net 上采样的特征网格伪影
    arr = cv2.GaussianBlur(arr, (15, 15), 0)

    cmap = plt.get_cmap('jet')
    colored = cmap(arr)[:, :, :3] * 255
    return colored.astype(np.uint8)

def save_image(path, img_np, is_rgb=True):
    if is_rgb:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), img_np)

def get_pipeline_from_engine(engine):
    """安全提取底层的 diffusers pipeline"""
    if hasattr(engine, "pipe"):
        return engine.pipe.pipe if hasattr(engine.pipe, "pipe") else engine.pipe
    elif hasattr(engine, "pipeline"):
        return engine.pipeline.pipe if hasattr(engine.pipeline, "pipe") else engine.pipeline
    return engine


def _normalize_attention_map(attention_map: torch.Tensor) -> torch.Tensor:
    attention_map = attention_map.detach().float()
    attn_min = attention_map.amin(dim=(-2, -1), keepdim=True)
    attn_max = attention_map.amax(dim=(-2, -1), keepdim=True)
    return (attention_map - attn_min) / (attn_max - attn_min + 1e-6)


def aggregate_attention_for_visualization(attention_maps, target_size: int = 512):
    if not attention_maps:
        raise ValueError("No attention maps were collected for visualization.")

    max_area = max(attn.shape[-2] * attn.shape[-1] for attn in attention_maps)
    highest_res_maps = [
        attn for attn in attention_maps
        if attn.shape[-2] * attn.shape[-1] == max_area
    ]
    if not highest_res_maps:
        raise ValueError("Failed to find highest-resolution attention maps for visualization.")

    aggregated_map = None
    for target_attn in highest_res_maps:
        normalized_attn = _normalize_attention_map(target_attn)
        resized_attn = F.interpolate(
            normalized_attn,
            size=(target_size, target_size),
            mode="bilinear",
            align_corners=False,
        )
        if aggregated_map is None:
            aggregated_map = resized_attn
        else:
            aggregated_map += resized_attn

    return aggregated_map / len(highest_res_maps)

# =========================================================================
# 模块一：生成拓扑图与【极其严谨的手动前向传播 Attention】
# =========================================================================
def generate_module_1_training_assets(config, engine):
    print("\n[1/3] 🚀 正在生成【训练期架构图 & 纯净的真实 Attention Maps】...")
    out_dir = Path(config["output_dir"]) / "1_training_and_attention"
    out_dir.mkdir(parents=True, exist_ok=True)
    workspace = Path(config["workspace_dir"])

    torch.manual_seed(int(config["seed"]))
    np.random.seed(int(config["seed"]))
    
    img_path = workspace / config["abnormal_img"]
    comp_mask_path = workspace / config["abnormal_comp_mask"]
    defect_mask_path = workspace / config["abnormal_defect_mask"]
    
    if not img_path.exists():
        print(f"⚠️ 找不到异常图片 {img_path}，请检查配置。")
        return

    # --- 1. 生成架构图需要的基础组件 (M_dilated, W_rec 等) ---
    abnormal_img = cv2.resize(cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB), (512, 512))
    comp_mask = cv2.resize(cv2.imread(str(comp_mask_path), cv2.IMREAD_GRAYSCALE), (512, 512), interpolation=cv2.INTER_NEAREST)
    _, comp_mask_bin = cv2.threshold(comp_mask, 127, 255, cv2.THRESH_BINARY)
    defect_mask = cv2.resize(cv2.imread(str(defect_mask_path), cv2.IMREAD_GRAYSCALE), (512, 512), interpolation=cv2.INTER_NEAREST)
    _, defect_mask_bin = cv2.threshold(defect_mask, 127, 255, cv2.THRESH_BINARY)
    
    save_image(out_dir / "01_abnormal_image_x.png", abnormal_img)
    save_image(out_dir / "02_gt_component_mask.png", comp_mask_bin, is_rgb=False)
    save_image(out_dir / "03_gt_defect_mask_m.png", defect_mask_bin, is_rgb=False)
    
    defect_mask_3c = np.stack([defect_mask_bin/255]*3, axis=-1)
    masked_img = (abnormal_img * (1 - defect_mask_3c)).astype(np.uint8)
    masked_img[defect_mask_bin > 127] = [128, 128, 128]
    save_image(out_dir / "04_masked_image_x_masked.png", masked_img)
    
    m_latent = F.interpolate(torch.from_numpy(defect_mask_bin).float().unsqueeze(0).unsqueeze(0)/255.0, size=(64, 64), mode="nearest")
    M_comp_latent = F.interpolate(torch.from_numpy(comp_mask_bin).float().unsqueeze(0).unsqueeze(0)/255.0, size=(64, 64), mode="nearest")
    M_dilated_latent = F.max_pool2d(m_latent, kernel_size=5, stride=1, padding=2) * M_comp_latent
    W_rec = 1.0 + 4.0 * M_dilated_latent
    
    save_image(out_dir / "05_latent_dilated_mask_M_dilated.png", (F.interpolate(M_dilated_latent, size=(512, 512), mode="nearest").squeeze().numpy() * 255).astype(np.uint8), is_rgb=False)
    save_image(out_dir / "06_weight_map_Wrec.png", apply_jet_colormap(F.interpolate(W_rec, size=(512, 512), mode="nearest").squeeze().numpy()))

    # --- 2. 核心：彻底绕开黑盒 Pipeline，手动单步前向提取真实 Attention ---
    print("🎯 正在执行底层 U-Net 前向传播提取特征 (绝对排除 CFG 污染)...")
    pipe = get_pipeline_from_engine(engine)
    device = pipe.device
    weight_dtype = pipe.unet.dtype

    img_transform = transforms.Compose([
        transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    mask_transform = transforms.Compose([
        transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor()
    ])

    image_pil = Image.fromarray(abnormal_img)
    d_mask_pil = Image.fromarray(defect_mask_bin).convert("L")
    
    pixel_values = img_transform(image_pil).unsqueeze(0).to(device, dtype=weight_dtype)
    d_mask_tensor = mask_transform(d_mask_pil).unsqueeze(0).to(device, dtype=weight_dtype)
    d_mask_tensor = (d_mask_tensor > 0.5).float()

    with torch.no_grad():
        latents = pipe.vae.encode(pixel_values).latent_dist.mode() * pipe.vae.config.scaling_factor
        noise = torch.randn_like(latents)
        timesteps = torch.tensor([500], device=device, dtype=torch.long)

        # scheduler 计算中可能会把结果强制拉到 float32
        noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)

        latent_mask = F.interpolate(d_mask_tensor, size=(latents.shape[2], latents.shape[3]), mode="nearest")
        masked_image = pixel_values * (d_mask_tensor < 0.5).to(weight_dtype)
        masked_image_latents = pipe.vae.encode(masked_image).latent_dist.mode() * pipe.vae.config.scaling_factor

        # 【修复关键点】：手动强制将输入压回 weight_dtype (如 float16)，防止 U-Net 类型冲突
        model_input = torch.cat([noisy_latents, latent_mask, masked_image_latents], dim=1).to(dtype=weight_dtype)

        prompt = f"a photo of {config['comp_token']} with {config['defect_token']}"
        input_ids = pipe.tokenizer(
            prompt, truncation=True, padding="max_length",
            max_length=pipe.tokenizer.model_max_length, return_tensors="pt"
        ).input_ids.to(device)

        comp_id = pipe.tokenizer.convert_tokens_to_ids(config['comp_token'])
        def_id = pipe.tokenizer.convert_tokens_to_ids(config['defect_token'])

        comp_idx, def_idx = -1, -1
        for i, tid in enumerate(input_ids[0]):
            if tid.item() == comp_id: comp_idx = i
            if tid.item() == def_id: def_idx = i

        if comp_idx < 0 or def_idx < 0:
            raise ValueError(
                f"Failed to locate target tokens in prompt. "
                f"component={config['comp_token']} idx={comp_idx}, defect={config['defect_token']} idx={def_idx}"
            )

        # 【修复关键点】：强制确保文本编码结果也是正确的 dtype
        encoder_hidden_states = pipe.text_encoder(input_ids)[0].to(dtype=weight_dtype)

    attn_store = AttentionStore()
    unet_to_hook = pipe.unet.base_model.model if hasattr(pipe.unet, "base_model") else pipe.unet
    attn_store.register_to_unet(unet_to_hook)
    attn_store.set_named_target_token_indices({
        "component": [comp_idx],
        "defect": [def_idx]
    })

    with torch.no_grad():
        pipe.unet(model_input, timesteps, encoder_hidden_states)

    component_maps = attn_store.processor.attention_maps.get("component", [])
    defect_maps = attn_store.processor.attention_maps.get("defect", [])
    if not component_maps:
        raise ValueError("No component attention maps were captured during the U-Net forward pass.")
    if not defect_maps:
        raise ValueError("No defect attention maps were captured during the U-Net forward pass.")

    aggregated = {
        "component": aggregate_attention_for_visualization(component_maps, target_size=512),
        "defect": aggregate_attention_for_visualization(defect_maps, target_size=512),
    }
    save_image(out_dir / "07_REAL_attention_map_component.png", apply_jet_colormap(aggregated["component"][0, 0]))
    save_image(out_dir / "08_REAL_attention_map_defect.png", apply_jet_colormap(aggregated["defect"][0, 0]))

    attn_store.clear()

# =========================================================================
# 模块二：掩码引擎推理逻辑 (拓扑几何与动态掩码)
# =========================================================================
def generate_module_2_mask_engine_assets(config):
    print("\n[2/3] 🚀 正在生成【Mask Engine 推理生成引擎】拓扑图素材...")
    out_dir = Path(config["output_dir"]) / "2_mask_engine_inference"
    out_dir.mkdir(parents=True, exist_ok=True)
    workspace = Path(config["workspace_dir"])
    
    normal_img_path = workspace / "normal_components" / config["normal_img_rel"]
    comp_mask_path = workspace / "normal_components" / config["normal_comp_mask_rel"]
    
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    
    normal_img = cv2.resize(cv2.cvtColor(cv2.imread(str(normal_img_path)), cv2.COLOR_BGR2RGB), (512, 512))
    comp_mask_cv = cv2.resize(cv2.imread(str(comp_mask_path), cv2.IMREAD_GRAYSCALE), (512, 512), interpolation=cv2.INTER_NEAREST)
    _, comp_mask_bin = cv2.threshold(comp_mask_cv, 127, 255, cv2.THRESH_BINARY)
    
    save_image(out_dir / "01_input_normal_x.png", normal_img)
    
    dist_map = cv2.distanceTransform(comp_mask_bin, cv2.DIST_L2, 5)
    save_image(out_dir / "02_distance_transform_Dxy.png", apply_jet_colormap(dist_map))
    
    mask_engine = DefectMaskEngine(train_dir=workspace / "defect_train_concept", cache_file=workspace / "defect_stats_cache.json")
    mask_engine.load_or_compute_stats([{"defect": config["defect_token"], "comp": config["comp_token"]}])
    dynamic_defect_mask = mask_engine.generate_dynamic_mask(comp_mask_bin, config["defect_token"])
    
    save_image(out_dir / "03_dynamic_generated_mask_m.png", dynamic_defect_mask, is_rgb=False)

# =========================================================================
# 模块三：基于正常图像的完整扩散推理与候选池
# =========================================================================
def generate_module_3_final_inference(config, engine):
    print("\n[3/3] 🚀 正在执行批量大模型推理，生成【最终缺陷与候选池】...")
    out_dir = Path(config["output_dir"]) / "3_final_generation_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    mask_result = engine.generate_mask_preview(
        component_token=config["comp_token"], 
        defect_token=config["defect_token"],
        selected_image_path=config["normal_img_rel"], 
        use_random_image=False, 
        base_seed=config["seed"], 
        refresh_index=0,
        length=50, 
        thickness=5, 
        width=5, 
        radius=5, 
        count=2, 
        curvature_min=0.05, 
        curvature_max=0.65, 
        random_use_cache_range=True 
    )
    mask_result["overlay"].save(out_dir / "01_mask_overlay_guide.png")

    print("✨ 正在运行 8-样本 LFS 优选推理...")
    gen_result = engine.generate_from_mask(
        mask_payload=mask_result["mask_payload"],
        num_inference_steps=30,
        guidance_scale=7.5,
        negative_prompt="blurry, smooth, unrealistic, artifacts",
        num_lfs_samples=8
    )

    gen_result["best"].save(out_dir / "02_final_best_defect.png")
    gen_result["triptych"].save(out_dir / "03_triptych_comparison.png")

    candidates_dir = out_dir / "candidates_pool"
    candidates_dir.mkdir(exist_ok=True)
    for idx, (img, desc) in enumerate(gen_result["candidates"]):
        img.save(candidates_dir / f"candidate_{idx+1}.png")

    print(f"✅ 最佳结果 LPIPS 得分: {gen_result['info']['best_score']:.4f}")

# =========================================================================
# 主控入口
# =========================================================================
if __name__ == "__main__":
    print("="*65)
    print("📄 MSD-Inpainting 论文全套视觉素材生成工具 (终极防错版本)")
    print("="*65)
    
    # 1. 全局初始化大模型引擎 (只需加载一次)
    print("⏳ 正在加载 Stable Diffusion 大模型与 LoRA 权重，请稍候...")
    global_engine = InteractiveDefectFillEngine(
        train_config_path=CONFIG["train_config"],
        infer_config_path=CONFIG["infer_config"],
        lora_weights=CONFIG["lora_weights"],
        normal_dir=str(Path(CONFIG["workspace_dir"]) / "normal_components"),
        stats_cache=str(Path(CONFIG["workspace_dir"]) / "defect_stats_cache.json"),
        device="cuda"
    )
    
    # 2. 依次执行三个生成模块
    generate_module_1_training_assets(CONFIG, global_engine)
    generate_module_2_mask_engine_assets(CONFIG)
    generate_module_3_final_inference(CONFIG, global_engine)
    
    print("\n🎉 全部执行完毕！所有高度严谨的学术图表素材已存放至:", CONFIG["output_dir"])
