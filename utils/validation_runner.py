import json
import random
from pathlib import Path

import numpy as np
import torch
from diffusers import DDIMScheduler, StableDiffusionInpaintPipeline
from torchvision import transforms
from transformers import CLIPTokenizer
from peft import PeftModel

from dataset.normal_dataset import NormalComponentDataset
from utils.mask_ops import DefectMaskEngine
from utils.ablation import build_conditioning_prompt


def tokenize_prompts(tokenizer, prompts):
    return tokenizer(
        prompts,
        truncation=True,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    ).input_ids


def _seeded_mask_generation(mask_engine, comp_mask_np, defect_token, seed: int):
    py_state = random.getstate()
    np_state = np.random.get_state()
    try:
        random.seed(seed)
        np.random.seed(seed % (2 ** 32))
        return mask_engine.generate_dynamic_mask(comp_mask_np, defect_token)
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)


def build_validation_suite(config):
    val_cfg = config.get("validation_inference", {})
    if not val_cfg.get("enabled", False):
        return []

    normal_dir = val_cfg.get("normal_dir")
    stats_cache = val_cfg.get("stats_cache")
    if not normal_dir or not stats_cache:
        print("⚠️ validation_inference 已启用，但缺少 normal_dir 或 stats_cache，跳过固定推理验证。")
        return []

    train_metadata_path = Path(config["paths"]["data_dir"]) / "metadata.jsonl"
    if not train_metadata_path.exists():
        print(f"⚠️ 找不到训练 metadata: {train_metadata_path}，跳过固定推理验证。")
        return []

    pair_set = set()
    with open(train_metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())
            defect_token = item.get("defect_token")
            object_token = item.get("object_token")
            if defect_token and object_token:
                pair_set.add((defect_token, object_token))

    if not pair_set:
        print("⚠️ 训练 metadata 中缺少 defect/object token 配对信息，跳过固定推理验证。")
        return []

    mask_engine = DefectMaskEngine(
        train_dir=Path(config["paths"]["data_dir"]),
        cache_file=Path(stats_cache),
        target_size=config["training"]["resolution"],
    )
    tasks = [{"defect": defect, "comp": comp} for defect, comp in sorted(pair_set)]
    mask_engine.load_or_compute_stats(tasks)

    base_seed = int(val_cfg.get("base_seed", config["training"].get("seed", 42)))
    validation_suite = []

    for pair_idx, (defect_token, object_token) in enumerate(sorted(pair_set)):
        dataset = NormalComponentDataset(
            data_dir=normal_dir,
            size=config["training"]["resolution"],
            target_comp=object_token,
        )
        if len(dataset) == 0:
            print(f"⚠️ 组件 {object_token} 在 normal_dir 中没有正常图像，跳过 {defect_token}.")
            continue

        local_rng = random.Random(base_seed + pair_idx * 9973)
        candidate_indices = list(range(len(dataset)))
        local_rng.shuffle(candidate_indices)
        dataset_idx = candidate_indices[0]
        sample = dataset[dataset_idx]
        component_mask = sample["mask_values"].clone()
        comp_mask_np = (component_mask.squeeze(0).numpy() * 255).astype(np.uint8)
        mask_seed = base_seed + pair_idx * 1000
        defect_mask_np = _seeded_mask_generation(mask_engine, comp_mask_np, defect_token, mask_seed)
        defect_mask = torch.from_numpy(defect_mask_np).float().unsqueeze(0) / 255.0

        validation_suite.append(
            {
                "defect_token": defect_token,
                "object_token": object_token,
                "image_path": sample["image_path"],
                "pixel_values": sample["pixel_values"].clone(),
                "component_mask": component_mask.clone(),
                "defect_mask": defect_mask.clone(),
                "seed": mask_seed,
            }
        )

    print(f"🧪 固定推理验证样本已准备: {len(validation_suite)}")
    return validation_suite

def _find_first_matching_token_index(prompt_ids, candidate_token_ids):
    for idx, token_id in enumerate(prompt_ids[0]):
        if token_id.item() in candidate_token_ids:
            return idx
    return -1


def run_periodic_inference_validation(
    *,
    config,
    epoch,
    accelerator,
    model_source,
    lora_dir,
    defect_token_ids,
    component_token_ids,
    attn_store,
    visualizer,
    validation_suite,
    weight_dtype,
):
    if not validation_suite or not accelerator.is_main_process:
        return

    val_cfg = config.get("validation_inference", {})
    pipeline = StableDiffusionInpaintPipeline.from_pretrained(model_source, torch_dtype=weight_dtype, local_files_only=True)
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    pipeline.tokenizer = CLIPTokenizer.from_pretrained(str(lora_dir / "tokenizer"))
    pipeline.text_encoder.resize_token_embeddings(len(pipeline.tokenizer))
    pipeline.text_encoder = PeftModel.from_pretrained(
        pipeline.text_encoder,
        str(lora_dir / "text_encoder_lora"),
    )
    pipeline.unet = PeftModel.from_pretrained(
        pipeline.unet,
        str(lora_dir / "unet_lora"),
    )
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)
    pipeline.enable_attention_slicing()
    attn_store.register_to_unet(pipeline.unet.base_model.model)

    prev_unet_mode = pipeline.unet.training
    prev_text_mode = pipeline.text_encoder.training
    pipeline.unet.eval()
    pipeline.text_encoder.eval()

    to_tensor = transforms.ToTensor()
    num_steps = int(val_cfg.get("num_inference_steps", 30))
    guidance_scale = float(val_cfg.get("guidance_scale", 7.5))

    try:
        for sample in validation_suite:
            prompt = build_conditioning_prompt(
                config, sample["object_token"], sample["defect_token"]
            )
            prompt_ids = tokenize_prompts(pipeline.tokenizer, [prompt]).to(accelerator.device)
            defect_token_index = _find_first_matching_token_index(prompt_ids, defect_token_ids)
            component_token_index = _find_first_matching_token_index(prompt_ids, component_token_ids)

            # 与训练保持一致：验证阶段的图像与掩码也统一使用 weight_dtype
            input_image = (sample["pixel_values"].unsqueeze(0) * 0.5 + 0.5).to(accelerator.device, dtype=weight_dtype)
            defect_mask = sample["defect_mask"].unsqueeze(0).to(accelerator.device, dtype=weight_dtype)
            component_mask = sample["component_mask"].unsqueeze(0).to(accelerator.device, dtype=weight_dtype)

            generator = torch.Generator(device=accelerator.device).manual_seed(int(sample["seed"]))
            attn_store.clear()
            with torch.no_grad():
                output = pipeline(
                    prompt=[prompt],
                    image=input_image,
                    mask_image=defect_mask,
                    num_inference_steps=num_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                ).images[0]

            generated_tensor = to_tensor(output).unsqueeze(0).to(accelerator.device, dtype=weight_dtype)
            safe_defect = sample["defect_token"].strip("<>")
            safe_component = sample["object_token"].strip("<>")
            defect_attention = None
            component_attention = None

            if defect_token_index >= 0:
                attn_store.clear()
                attn_store.set_target_token_indices([defect_token_index])
                with torch.no_grad():
                    pipeline(
                        prompt=[prompt],
                        image=input_image,
                        mask_image=defect_mask,
                        num_inference_steps=num_steps,
                        guidance_scale=guidance_scale,
                        generator=torch.Generator(device=accelerator.device).manual_seed(int(sample["seed"])),
                    )
                defect_attention = attn_store.get_aggregated_attention(target_size=input_image.shape[-1])

            if component_token_index >= 0:
                attn_store.clear()
                attn_store.set_target_token_indices([component_token_index])
                with torch.no_grad():
                    pipeline(
                        prompt=[prompt],
                        image=input_image,
                        mask_image=defect_mask,
                        num_inference_steps=num_steps,
                        guidance_scale=guidance_scale,
                        generator=torch.Generator(device=accelerator.device).manual_seed(int(sample["seed"])),
                    )
                component_attention = attn_store.get_aggregated_attention(target_size=input_image.shape[-1])

            visualizer.log_combined_inference_panel(
                tag=f"ValidationInference/{safe_defect}_on_{safe_component}",
                input_image=input_image,
                defect_mask=defect_mask,
                component_mask=component_mask,
                generated_image=generated_tensor,
                defect_attention_map=defect_attention,
                component_attention_map=component_attention,
                step=epoch,
            )

            attn_store.clear()
    finally:
        if prev_unet_mode:
            pipeline.unet.train()
        if prev_text_mode:
            pipeline.text_encoder.train()
        del pipeline
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
