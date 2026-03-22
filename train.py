import os
import yaml
import argparse
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.utils import set_seed
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel

from dataset.inpainting_dataset import DefectFillDataset
from models.lora_handler import setup_lora_and_tokens
from models.attention_hook import AttentionStore
from losses.defectfill_loss import DefectFillLoss
from utils.runtime import resolve_model_source, resolve_pretrained_variant, resolve_weight_dtype
from utils.mask_ops import build_random_box_mask
from utils.monitoring import TokenDriftMonitor, TensorBoardVisualizer
from utils.validation_runner import build_validation_suite, run_periodic_inference_validation, tokenize_prompts


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train_config.yaml")
    return parser.parse_args()

def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    seed = int(config["training"].get("seed", 42))
    set_seed(seed)

    # ==========================================
    # 【改动 1】初始化加速器时开启 tensorboard，并指定日志路径
    # ==========================================
    log_dir = config["paths"].get("logging_dir", "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    accelerator = Accelerator(
        mixed_precision=config["training"]["mixed_precision"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        log_with="tensorboard",
        project_dir=log_dir
    )
    device = accelerator.device
    mixed_precision = config["training"].get("mixed_precision", "no")
    pretrained_variant = resolve_pretrained_variant(mixed_precision)
    weight_dtype = resolve_weight_dtype(mixed_precision)

    # 【改动 2】初始化 Tracker
    if accelerator.is_main_process:
        accelerator.init_trackers("defectfill_training")

    # 加载预训练模型
    model_source = resolve_model_source(config["paths"])
    tokenizer = CLIPTokenizer.from_pretrained(model_source, subfolder="tokenizer", local_files_only=True)
    text_encoder_load_kwargs = {"local_files_only": True}
    vae_load_kwargs = {"local_files_only": True}
    unet_load_kwargs = {"local_files_only": True}
    if pretrained_variant is not None:
        text_encoder_load_kwargs["variant"] = pretrained_variant
        vae_load_kwargs["variant"] = pretrained_variant
        unet_load_kwargs["variant"] = pretrained_variant
    text_encoder = CLIPTextModel.from_pretrained(model_source, subfolder="text_encoder", **text_encoder_load_kwargs)
    vae = AutoencoderKL.from_pretrained(model_source, subfolder="vae", **vae_load_kwargs)
    unet = UNet2DConditionModel.from_pretrained(model_source, subfolder="unet", **unet_load_kwargs)
    noise_scheduler = DDPMScheduler.from_pretrained(model_source, subfolder="scheduler", local_files_only=True)
    vae.requires_grad_(False)

    # ==========================================
    # 👇 把显存优化放在这里！(刚加载完基础模型，还没加 LoRA 之前)
    # ==========================================
    unet.enable_gradient_checkpointing()
    if hasattr(text_encoder, "gradient_checkpointing_enable"):
        text_encoder.gradient_checkpointing_enable()
    
    # 修改了这里的方法名：从 enable_attention_slicing 改为 set_attention_slice
    unet.set_attention_slice("auto") 
    # ==========================================

    # 注入 LoRA 和 Token
    # 【修改为】：
    tokenizer, text_encoder, unet, defect_token_ids, component_token_ids = setup_lora_and_tokens(tokenizer, text_encoder, unet, config)
    
    defect_tokens = config.get("defect_tokens", [])
    component_tokens = config.get("component_tokens", [])

    # 挂载 Attention Hook
    attn_store = AttentionStore()
    attn_store.register_to_unet(unet.base_model.model)

    # 实例化 Loss
    lw = config["loss_weights"]
    criterion = DefectFillLoss(
        lambda_def=lw["lambda_def"],
        lambda_obj=lw["lambda_obj"],
        lambda_attn=lw["lambda_attn"],
        object_loss_alpha=config["loss_weights"].get("object_loss_alpha", 0.25),
    )

    validation_suite = build_validation_suite(config)
    token_monitor = TokenDriftMonitor(text_encoder, tokenizer, defect_tokens, component_tokens)

    # 配置优化器
    opt_cfg = config["optimizer"]
    embedding_params = []
    embedding_param_ids = set()
    for param in text_encoder.get_input_embeddings().parameters():
        if param.requires_grad:
            embedding_params.append(param)
            embedding_param_ids.add(id(param))

    text_lora_params = []
    for param in text_encoder.parameters():
        if param.requires_grad and id(param) not in embedding_param_ids:
            text_lora_params.append(param)

    optimizer = torch.optim.AdamW([
        {"params": [p for p in unet.parameters() if p.requires_grad], "lr": float(opt_cfg["learning_rate_unet"])},
        {"params": text_lora_params, "lr": float(opt_cfg["learning_rate_text"])},
        {"params": embedding_params, "lr": float(opt_cfg["learning_rate_text"])}
    ], weight_decay=float(opt_cfg["weight_decay"]))

    # 准备数据与加速器
    dataset = DefectFillDataset(data_dir=config["paths"]["data_dir"], tokenizer=tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=config["training"]["train_batch_size"],
        shuffle=True,
        num_workers=config["training"].get("dataloader_num_workers", 0),
    )
    unet, text_encoder, optimizer, dataloader = accelerator.prepare(unet, text_encoder, optimizer, dataloader)

    vae.to(device, dtype=weight_dtype)

    # 核心训练循环
    epochs = config["training"]["max_train_epochs"]
    global_step = 0  # 【改动 3】增加全局步数计数器
    # 【新增】初始化可视化器
    visualizer = TensorBoardVisualizer(accelerator)
    checkpoint_every_n_epochs = int(config["training"].get("checkpoint_every_n_epochs", 10))
    if checkpoint_every_n_epochs <= 0:
        raise ValueError("training.checkpoint_every_n_epochs 必须为正整数。")

    for epoch in range(epochs):
        unet.train()
        text_encoder.train()
        is_checkpoint_epoch = (epoch % checkpoint_every_n_epochs == 0)

        for step, batch in enumerate(dataloader):
            with accelerator.accumulate(unet, text_encoder):
                pixel_values = batch["pixel_values"].to(device, dtype=weight_dtype)
                defect_mask = batch["mask_values"].to(device, dtype=weight_dtype)
                defect_prompt_ids = tokenize_prompts(tokenizer, batch["defect_prompt"]).to(device)
                object_prompt_ids = tokenize_prompts(tokenizer, batch["object_prompt"]).to(device)

                # VAE 编码与 9 通道准备
                latents = vae.encode(pixel_values).latent_dist.sample() * vae.config.scaling_factor
                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                latent_mask = F.interpolate(defect_mask, size=(latents.shape[2], latents.shape[3]), mode="nearest")
                masked_image = pixel_values * (defect_mask < 0.5).to(weight_dtype)
                masked_image_latents = vae.encode(masked_image).latent_dist.sample() * vae.config.scaling_factor
                defect_model_input = torch.cat([noisy_latents, latent_mask, masked_image_latents], dim=1)

                random_mask = build_random_box_mask(defect_mask)
                latent_random_mask = F.interpolate(random_mask, size=(latents.shape[2], latents.shape[3]), mode="nearest")
                random_masked_image = pixel_values * (random_mask < 0.5).to(weight_dtype)
                random_masked_image_latents = vae.encode(random_masked_image).latent_dist.sample() * vae.config.scaling_factor
                object_model_input = torch.cat([noisy_latents, latent_random_mask, random_masked_image_latents], dim=1)

                defect_hidden_states = text_encoder(defect_prompt_ids)[0]
                defect_model_pred = unet(defect_model_input, timesteps, defect_hidden_states).sample

                # 原代码会在 prompt 里寻找 new_token_ids
                token_indices = []
                for sample_input_ids in object_prompt_ids:
                    target_token_index = -1
                    for idx, token_id in enumerate(sample_input_ids):
                        # 【关键修改】：这里必须换成 defect_token_ids
                        if token_id.item() in defect_token_ids:
                            target_token_index = idx
                            break
                    token_indices.append(target_token_index)

                attn_store.clear()
                attention_map = None
                if all(token_index >= 0 for token_index in token_indices):
                    attn_store.set_target_token_indices(token_indices)

                object_hidden_states = text_encoder(object_prompt_ids)[0]
                object_model_pred = unet(object_model_input, timesteps, object_hidden_states).sample

                if all(token_index >= 0 for token_index in token_indices):
                    attention_map = attn_store.get_aggregated_attention(target_size=latents.shape[-1])
                loss, loss_dict = criterion(defect_model_pred, object_model_pred, noise, defect_mask, attention_map)

                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
                attn_store.clear()
            
            # 【改动 4】记录日志到 TensorBoard
            if accelerator.is_main_process:
                # 记录详细的拆解 Loss
                accelerator.log({
                    "Loss/Total": loss.item(),
                    "Loss/Defect (L_def)": loss_dict["loss_def"],
                    "Loss/Object (L_obj)": loss_dict["loss_obj"],
                    "Loss/Attention (L_attn)": loss_dict["loss_attn"]
                }, step=global_step)
                
                if step % 10 == 0:
                    print(f"Epoch {epoch} | Step {global_step} | Total Loss: {loss.item():.4f} | {loss_dict}")
                if is_checkpoint_epoch and step == len(dataloader) - 1:
                    visualizer.log_random_box_diagnostics(
                        input_image=(pixel_values * 0.5 + 0.5),
                        defect_mask=defect_mask,
                        random_mask=random_mask,
                        step=epoch,
                    )
            
            global_step += 1
        # ==========================================
        # 【新增】每隔 10 个 Epoch 保存一次检查点
        # ==========================================
        accelerator.wait_for_everyone() # 确保所有进程都完成了当前 epoch
        if accelerator.is_main_process:
            unwrapped_text_encoder = accelerator.unwrap_model(text_encoder)
            token_monitor.log_token_drift(unwrapped_text_encoder, visualizer, epoch)
            visualizer.log_token_confusion_matrix(
                unwrapped_text_encoder,
                tokenizer,
                defect_tokens,
                epoch,
                tag="Embeddings/Defect_Token_Similarity_Matrix",
                title="Defect Tokens Cosine Similarity",
            )
            visualizer.log_token_confusion_matrix(
                unwrapped_text_encoder,
                tokenizer,
                component_tokens,
                epoch,
                tag="Embeddings/Component_Token_Similarity_Matrix",
                title="Component Tokens Cosine Similarity",
            )

        if is_checkpoint_epoch:
            if accelerator.is_main_process:
                base_save_dir = config["paths"]["output_dir"]
                # 为每个 checkpoint 创建独立的子文件夹
                ckpt_dir = os.path.join(base_save_dir, f"checkpoint-epoch-{epoch}")
                os.makedirs(ckpt_dir, exist_ok=True)
                
                # 解包模型后保存，防止 DDP 报错
                unwrapped_unet = accelerator.unwrap_model(unet)
                unwrapped_unet.save_pretrained(os.path.join(ckpt_dir, "unet_lora"))
                unwrapped_text_encoder.save_pretrained(
                    os.path.join(ckpt_dir, "text_encoder_lora"),
                    save_embedding_layers=True,
                )
                tokenizer.save_pretrained(os.path.join(ckpt_dir, "tokenizer"))
                if config.get("validation_inference", {}).get("enabled", False):
                    run_periodic_inference_validation(
                        config=config,
                        epoch=epoch,
                        accelerator=accelerator,
                        model_source=model_source,
                        lora_dir=Path(ckpt_dir),
                        defect_token_ids=defect_token_ids,
                        component_token_ids=component_token_ids,
                        attn_store=attn_store,
                        visualizer=visualizer,
                        validation_suite=validation_suite,
                        weight_dtype=weight_dtype,
                    )
                print(f"💾 Epoch {epoch} 完成，阶段性权重已保存至 {ckpt_dir}")
            accelerator.wait_for_everyone()

    # ==========================================
    # 训练完全结束后的最终保存逻辑
    # ==========================================
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_dir = config["paths"]["output_dir"]
        os.makedirs(save_dir, exist_ok=True)
        
        # 解包模型后保存最终权重
        unwrapped_unet = accelerator.unwrap_model(unet)
        unwrapped_unet.save_pretrained(os.path.join(save_dir, "unet_lora"))
        
        unwrapped_text_encoder = accelerator.unwrap_model(text_encoder)
        unwrapped_text_encoder.save_pretrained(
            os.path.join(save_dir, "text_encoder_lora"),
            save_embedding_layers=True,
        )
        
        tokenizer.save_pretrained(os.path.join(save_dir, "tokenizer"))
        print(f"✅ 训练结束！最终权重已保存至 {save_dir}")
        
        # 结束记录
        accelerator.end_training()

if __name__ == "__main__":
    main()
