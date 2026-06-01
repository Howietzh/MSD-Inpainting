import argparse
import gc
import json
import random
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch
import yaml
from diffusers import DDIMScheduler, StableDiffusionInpaintPipeline
from peft import PeftModel
from transformers import CLIPTokenizer

from dataset.normal_dataset import NormalComponentDataset
from models.attention_hook import AttentionStore
from utils.mask_ops import DefectMaskEngine
from utils.runtime import resolve_model_source, resolve_weight_dtype

matplotlib.use("Agg")
import matplotlib.pyplot as plt


TARGET_SIZE = 512
COMPACT_EPOCHS = (0, 10, 30, 60)


def parse_args():
    parser = argparse.ArgumentParser(description="Build a compact lens_scratch attention evolution figure.")
    parser.add_argument("--train-config", type=str, default="configs/train_config.yaml")
    parser.add_argument("--infer-config", type=str, default="configs/inference_config.yaml")
    parser.add_argument("--weights-root", type=str, default=None)
    parser.add_argument("--normal-dir", type=str, default=None)
    parser.add_argument("--train-dir", type=str, default=None)
    parser.add_argument("--stats-cache", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="paper_all_assets/1_training_and_attention")
    parser.add_argument("--experiment-a", type=str, default="defectfill_origin")
    parser.add_argument("--experiment-b", type=str, default="increase_text_encoder_learning_rates")
    parser.add_argument("--defect-token", type=str, default="<lens_scratch>")
    parser.add_argument("--component-token", type=str, default="<lens>")
    parser.add_argument("--candidate-count", type=int, default=8)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--negative-prompt", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def choose_existing_path(explicit_value, candidate_values, label: str, create_parent: bool = False) -> Path:
    if explicit_value:
        path = Path(explicit_value).expanduser().resolve()
        if create_parent:
            path.parent.mkdir(parents=True, exist_ok=True)
        return path

    for candidate in candidate_values:
        if not candidate:
            continue
        path = Path(candidate).expanduser()
        if path.exists():
            return path.resolve()

    fallback = Path(next(candidate for candidate in candidate_values if candidate)).expanduser().resolve()
    if create_parent:
        fallback.parent.mkdir(parents=True, exist_ok=True)
    return fallback


def is_complete_lora_dir(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "tokenizer").is_dir()
        and (path / "text_encoder_lora").exists()
        and (path / "unet_lora").exists()
    )


def collect_checkpoints(experiment_dir: Path):
    if not experiment_dir.is_dir():
        raise FileNotFoundError(f"Experiment directory not found: {experiment_dir}")

    epoch_dirs = {}
    for child in sorted(experiment_dir.iterdir()):
        if not child.is_dir() or not child.name.startswith("checkpoint-epoch-"):
            continue
        try:
            epoch = int(child.name.rsplit("-", 1)[-1])
        except ValueError:
            continue
        if is_complete_lora_dir(child):
            epoch_dirs[epoch] = child

    final_dir = experiment_dir if is_complete_lora_dir(experiment_dir) else None
    return {"epochs": epoch_dirs, "final": final_dir}


def resolve_compact_checkpoints(checkpoints_a, checkpoints_b):
    common_epochs = sorted(set(checkpoints_a["epochs"]).intersection(checkpoints_b["epochs"]))
    missing = [epoch for epoch in COMPACT_EPOCHS if epoch not in common_epochs]
    if missing:
        raise FileNotFoundError(
            f"Compact checkpoints missing from the shared epoch set: {missing}. "
            f"Available shared checkpoints: {common_epochs}"
        )

    selected = [
        {
            "label": f"E{epoch}",
            "kind": "epoch",
            "epoch": epoch,
            "path_a": checkpoints_a["epochs"][epoch],
            "path_b": checkpoints_b["epochs"][epoch],
        }
        for epoch in COMPACT_EPOCHS
    ]

    if checkpoints_a["final"] is not None and checkpoints_b["final"] is not None:
        selected.append(
            {
                "label": "Final",
                "kind": "final",
                "epoch": None,
                "path_a": checkpoints_a["final"],
                "path_b": checkpoints_b["final"],
            }
        )
        return selected

    highest_common_epoch = max(common_epochs)
    if highest_common_epoch <= COMPACT_EPOCHS[-1]:
        raise FileNotFoundError(
            "Neither experiment has a final weights directory, and there is no shared checkpoint later than epoch-60 "
            "to use as the last compact column."
        )

    selected.append(
        {
            "label": f"E{highest_common_epoch}",
            "kind": "epoch_fallback",
            "epoch": highest_common_epoch,
            "path_a": checkpoints_a["epochs"][highest_common_epoch],
            "path_b": checkpoints_b["epochs"][highest_common_epoch],
        }
    )
    return selected


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)


def tensor_to_rgb_image(pixel_values: torch.Tensor) -> np.ndarray:
    image = (pixel_values.detach().cpu() * 0.5 + 0.5).clamp(0.0, 1.0)
    image_np = image.permute(1, 2, 0).numpy()
    return np.clip(np.round(image_np * 255.0), 0, 255).astype(np.uint8)


def tensor_mask_to_uint8(mask_values: torch.Tensor) -> np.ndarray:
    mask_np = mask_values.detach().cpu().squeeze(0).numpy()
    return ((mask_np > 0.5).astype(np.uint8)) * 255


def build_overlay_image(image_np: np.ndarray, defect_mask_np: np.ndarray) -> np.ndarray:
    overlay = image_np.copy()
    mask_bool = defect_mask_np > 127
    overlay[mask_bool] = (
        0.6 * overlay[mask_bool] + 0.4 * np.array([255, 80, 80], dtype=np.float32)
    ).astype(np.uint8)
    return overlay


def serialize_path(path: Path) -> str:
    return str(path.expanduser().resolve())


def build_candidate_pool(
    *,
    normal_dir: Path,
    train_dir: Path,
    stats_cache: Path,
    component_token: str,
    defect_token: str,
    candidate_count: int,
    base_seed: int,
):
    dataset = NormalComponentDataset(data_dir=str(normal_dir), size=TARGET_SIZE, target_comp=component_token)
    if len(dataset) == 0:
        raise ValueError(f"No normal samples found for component token {component_token}.")

    sample_count = min(int(candidate_count), len(dataset))
    if sample_count <= 0:
        raise ValueError("candidate_count must be a positive integer.")
    candidate_indices = list(range(len(dataset)))
    rng = random.Random(int(base_seed))
    rng.shuffle(candidate_indices)
    selected_indices = candidate_indices[:sample_count]

    mask_engine = DefectMaskEngine(train_dir=train_dir, cache_file=stats_cache, target_size=TARGET_SIZE)
    mask_engine.load_or_compute_stats([{"defect": defect_token, "comp": component_token}])
    if stats_cache.exists():
        with open(stats_cache, "r", encoding="utf-8") as f:
            mask_engine.stats_cache = json.load(f)

    candidates = []
    for sample_rank, dataset_index in enumerate(selected_indices):
        sample = dataset[dataset_index]
        image_np = tensor_to_rgb_image(sample["pixel_values"])
        component_mask_np = tensor_mask_to_uint8(sample["mask_values"])
        mask_seed = int(base_seed) + sample_rank

        seed_everything(mask_seed)
        mask_params = mask_engine.sample_generation_params(defect_token)
        defect_mask_np, mask_details = mask_engine.generate_dynamic_mask_with_params(
            component_mask_np,
            defect_token,
            mask_params,
            return_details=True,
        )

        candidates.append(
            {
                "dataset_index": int(dataset_index),
                "sample_rank": int(sample_rank),
                "image_path": sample["image_path"],
                "component_mask_path": sample["component_mask_path"],
                "mask_seed": mask_seed,
                "mask_area": int(cv2.countNonZero(defect_mask_np)),
                "mask_params": mask_params,
                "mask_details": mask_details,
                "image_np": image_np,
                "component_mask_np": component_mask_np,
                "defect_mask_np": defect_mask_np,
            }
        )

    median_area = float(np.median([candidate["mask_area"] for candidate in candidates]))
    representative = min(
        candidates,
        key=lambda candidate: (
            abs(candidate["mask_area"] - median_area),
            candidate["sample_rank"],
            candidate["dataset_index"],
        ),
    )
    return candidates, representative, median_area


def build_pipe(model_source: str, weight_dtype: torch.dtype, device: torch.device, lora_dir: Path):
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        model_source,
        torch_dtype=weight_dtype,
        local_files_only=True,
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.tokenizer = CLIPTokenizer.from_pretrained(str(lora_dir / "tokenizer"))
    pipe.text_encoder.resize_token_embeddings(len(pipe.tokenizer))
    pipe.text_encoder = PeftModel.from_pretrained(pipe.text_encoder, str(lora_dir / "text_encoder_lora"))
    pipe.unet = PeftModel.from_pretrained(pipe.unet, str(lora_dir / "unet_lora"))
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    pipe.enable_attention_slicing()
    pipe.unet.eval()
    pipe.text_encoder.eval()
    return pipe


def find_token_index(tokenizer, prompt: str, target_token: str, device: torch.device) -> int:
    prompt_ids = tokenizer(
        [prompt],
        truncation=True,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    ).input_ids.to(device)
    target_token_id = tokenizer.convert_tokens_to_ids(target_token)
    for idx, token_id in enumerate(prompt_ids[0]):
        if token_id.item() == target_token_id:
            return idx
    return -1


def build_image_tensor(image_np: np.ndarray, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    tensor = torch.from_numpy(image_np).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    return tensor.to(device=device, dtype=dtype)


def build_mask_tensor(mask_np: np.ndarray, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    tensor = torch.from_numpy(mask_np).float().unsqueeze(0).unsqueeze(0) / 255.0
    return tensor.to(device=device, dtype=dtype)


def resolve_pipe_device(pipe) -> torch.device:
    return next(pipe.unet.parameters()).device


def extract_defect_attention(
    *,
    pipe,
    prompt: str,
    defect_token: str,
    image_np: np.ndarray,
    defect_mask_np: np.ndarray,
    seed: int,
    num_inference_steps: int,
    guidance_scale: float,
    negative_prompt: str | None,
):
    attn_store = AttentionStore()
    unet_to_hook = pipe.unet.base_model.model if hasattr(pipe.unet, "base_model") else pipe.unet
    attn_store.register_to_unet(unet_to_hook)
    pipe_device = resolve_pipe_device(pipe)

    token_index = find_token_index(pipe.tokenizer, prompt, defect_token, pipe_device)
    if token_index < 0:
        raise ValueError(f"Failed to locate token {defect_token!r} in prompt: {prompt}")

    capture_mode = "cfg_conditional" if float(guidance_scale) > 1.0 else "direct"
    attn_store.clear()
    attn_store.set_target_token_indices([token_index], capture_mode=capture_mode)

    image_tensor = build_image_tensor(image_np, pipe_device, pipe.unet.dtype)
    defect_mask_tensor = build_mask_tensor(defect_mask_np, pipe_device, pipe.unet.dtype)
    generator = torch.Generator(device=pipe_device).manual_seed(int(seed))

    pipe_kwargs = {
        "prompt": [prompt],
        "image": image_tensor,
        "mask_image": defect_mask_tensor,
        "num_inference_steps": int(num_inference_steps),
        "guidance_scale": float(guidance_scale),
        "generator": generator,
    }
    if negative_prompt:
        pipe_kwargs["negative_prompt"] = [negative_prompt]

    with torch.no_grad():
        pipe(**pipe_kwargs)

    attention_map = attn_store.get_aggregated_attention(target_size=TARGET_SIZE)
    attn_store.clear()
    if attention_map is None:
        raise RuntimeError("No defect attention map was captured during the inpainting trajectory.")
    return attention_map[0, 0].detach().float().cpu().numpy()


def collect_attention_maps(
    *,
    model_source: str,
    weight_dtype: torch.dtype,
    device: torch.device,
    representative_sample,
    prompt: str,
    defect_token: str,
    checkpoint_slots,
    num_inference_steps: int,
    guidance_scale: float,
    negative_prompt: str | None,
    base_seed: int,
):
    method_specs = [
        ("DefectFill", checkpoint_slots, "path_a"),
        ("MSD-Inpainting", checkpoint_slots, "path_b"),
    ]
    method_to_maps = {"DefectFill": [], "MSD-Inpainting": []}

    for method_name, slots, path_key in method_specs:
        for slot in slots:
            lora_dir = slot[path_key]
            print(f"[{method_name}] extracting {slot['label']} attention from {lora_dir}")
            pipe = build_pipe(model_source, weight_dtype, device, lora_dir)
            try:
                attention_map = extract_defect_attention(
                    pipe=pipe,
                    prompt=prompt,
                    defect_token=defect_token,
                    image_np=representative_sample["image_np"],
                    defect_mask_np=representative_sample["defect_mask_np"],
                    seed=base_seed,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    negative_prompt=negative_prompt,
                )
            finally:
                del pipe
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            method_to_maps[method_name].append(
                {
                    "label": slot["label"],
                    "kind": slot["kind"],
                    "epoch": slot["epoch"],
                    "lora_dir": serialize_path(lora_dir),
                    "attention_map": attention_map,
                }
            )

    return method_to_maps


def compute_shared_color_range(method_to_maps):
    stacked = np.concatenate(
        [item["attention_map"].reshape(-1) for maps in method_to_maps.values() for item in maps],
        axis=0,
    )
    vmin = float(np.percentile(stacked, 1.0))
    vmax = float(np.percentile(stacked, 99.0))
    if vmax <= vmin:
        vmax = float(stacked.max()) if stacked.size else 1.0
        vmin = float(stacked.min()) if stacked.size else 0.0
        if vmax <= vmin:
            vmax = vmin + 1e-6
    return vmin, vmax


def draw_mask_contour(ax, defect_mask_np: np.ndarray, color: str = "white"):
    contours, _ = cv2.findContours(defect_mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        contour = contour.squeeze(1)
        if contour.ndim != 2 or contour.shape[0] < 2:
            continue
        ax.plot(contour[:, 0], contour[:, 1], color=color, linewidth=0.6)


def save_compact_figure(
    *,
    output_dir: Path,
    representative_sample,
    checkpoint_slots,
    method_to_maps,
    vmin: float,
    vmax: float,
    dpi: int,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    overlay_np = build_overlay_image(
        representative_sample["image_np"],
        representative_sample["defect_mask_np"],
    )

    titles = ["Mask"] + [slot["label"] for slot in checkpoint_slots]
    method_order = ["DefectFill", "MSD-Inpainting"]
    fig, axes = plt.subplots(
        nrows=2,
        ncols=1 + len(checkpoint_slots),
        figsize=(7.1, 2.6),
        dpi=int(dpi),
    )

    for row_idx, method_name in enumerate(method_order):
        for col_idx in range(1 + len(checkpoint_slots)):
            ax = axes[row_idx, col_idx]
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

            if row_idx == 0:
                ax.set_title(titles[col_idx], fontsize=8, pad=3)

            if col_idx == 0:
                ax.imshow(overlay_np)
                draw_mask_contour(ax, representative_sample["defect_mask_np"], color="white")
            else:
                attention_item = method_to_maps[method_name][col_idx - 1]
                ax.imshow(attention_item["attention_map"], cmap="jet", vmin=vmin, vmax=vmax)
                draw_mask_contour(ax, representative_sample["defect_mask_np"], color="white")

        axes[row_idx, 0].set_ylabel(method_name, fontsize=8, rotation=90, labelpad=10)

    fig.subplots_adjust(left=0.06, right=0.995, top=0.90, bottom=0.06, wspace=0.02, hspace=0.06)
    png_path = output_dir / "lens_scratch_evolution_compact.png"
    pdf_path = output_dir / "lens_scratch_evolution_compact.pdf"
    fig.savefig(png_path, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return png_path, pdf_path


def build_manifest(
    *,
    args,
    train_config,
    infer_config,
    weights_root: Path,
    normal_dir: Path,
    train_dir: Path,
    stats_cache: Path,
    representative_sample,
    candidates,
    median_area: float,
    checkpoint_slots,
    method_to_maps,
    vmin: float,
    vmax: float,
    output_paths,
    num_inference_steps: int,
    guidance_scale: float,
    negative_prompt: str,
):
    return {
        "train_config": serialize_path(Path(args.train_config)),
        "infer_config": serialize_path(Path(args.infer_config)),
        "weights_root": serialize_path(weights_root),
        "normal_dir": serialize_path(normal_dir),
        "train_dir": serialize_path(train_dir),
        "stats_cache": serialize_path(stats_cache),
        "component_token": args.component_token,
        "defect_token": args.defect_token,
        "base_seed": int(args.base_seed),
        "candidate_count": int(args.candidate_count),
        "num_inference_steps": int(num_inference_steps),
        "guidance_scale": float(guidance_scale),
        "negative_prompt": negative_prompt,
        "shared_color_range": {"vmin": vmin, "vmax": vmax},
        "compact_checkpoints": [
            {
                "label": slot["label"],
                "kind": slot["kind"],
                "epoch": slot["epoch"],
                "experiment_a_path": serialize_path(slot["path_a"]),
                "experiment_b_path": serialize_path(slot["path_b"]),
            }
            for slot in checkpoint_slots
        ],
        "selected_sample": {
            "dataset_index": representative_sample["dataset_index"],
            "sample_rank": representative_sample["sample_rank"],
            "image_path": representative_sample["image_path"],
            "component_mask_path": representative_sample["component_mask_path"],
            "mask_seed": representative_sample["mask_seed"],
            "mask_area": representative_sample["mask_area"],
            "mask_params": representative_sample["mask_params"],
            "mask_details": representative_sample["mask_details"],
            "median_candidate_area": median_area,
        },
        "candidate_pool": [
            {
                "dataset_index": candidate["dataset_index"],
                "sample_rank": candidate["sample_rank"],
                "image_path": candidate["image_path"],
                "component_mask_path": candidate["component_mask_path"],
                "mask_seed": candidate["mask_seed"],
                "mask_area": candidate["mask_area"],
                "mask_params": candidate["mask_params"],
                "mask_details": candidate["mask_details"],
            }
            for candidate in candidates
        ],
        "methods": {
            method_name: [
                {
                    "label": item["label"],
                    "kind": item["kind"],
                    "epoch": item["epoch"],
                    "lora_dir": item["lora_dir"],
                }
                for item in items
            ]
            for method_name, items in method_to_maps.items()
        },
        "outputs": {
            "png": serialize_path(output_paths["png"]),
            "pdf": serialize_path(output_paths["pdf"]),
        },
        "model_id": train_config["paths"]["model_id"],
        "normal_dir_from_config": infer_config["paths"].get("normal_dir"),
    }


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    train_config = load_yaml(args.train_config)
    infer_config = load_yaml(args.infer_config)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    weights_root = choose_existing_path(
        args.weights_root,
        [
            repo_root / "defectfill_lora_weights",
            repo_root.parent / "defectfill_lora_weights",
            train_config["paths"].get("output_dir"),
        ],
        label="weights_root",
    )
    normal_dir = choose_existing_path(
        args.normal_dir,
        [
            infer_config["paths"].get("normal_dir"),
            repo_root / "data" / "CCM-Defect" / "normal_components",
            repo_root.parent / "data" / "CCM-Defect" / "normal_components",
        ],
        label="normal_dir",
    )
    train_dir = choose_existing_path(
        args.train_dir,
        [
            infer_config["paths"].get("train_dir"),
            train_config["paths"].get("data_dir"),
            repo_root / "data" / "CCM-Defect" / "defect_train_concept",
            repo_root.parent / "data" / "CCM-Defect" / "defect_train_concept",
        ],
        label="train_dir",
    )
    stats_cache = choose_existing_path(
        args.stats_cache,
        [
            output_dir / "lens_scratch_stats_cache.json",
            infer_config["paths"].get("stats_cache"),
        ],
        label="stats_cache",
        create_parent=True,
    )

    num_inference_steps = int(
        args.num_inference_steps
        if args.num_inference_steps is not None
        else infer_config.get("inference", {}).get("num_inference_steps", 30)
    )
    guidance_scale = float(
        args.guidance_scale
        if args.guidance_scale is not None
        else infer_config.get("inference", {}).get("guidance_scale", 7.5)
    )
    negative_prompt = (
        args.negative_prompt
        if args.negative_prompt is not None
        else infer_config.get("inference", {}).get("negative_prompt", "")
    )

    experiment_a_dir = weights_root / args.experiment_a
    experiment_b_dir = weights_root / args.experiment_b
    checkpoints_a = collect_checkpoints(experiment_a_dir)
    checkpoints_b = collect_checkpoints(experiment_b_dir)
    checkpoint_slots = resolve_compact_checkpoints(checkpoints_a, checkpoints_b)
    print("Selected checkpoints:", ", ".join(slot["label"] for slot in checkpoint_slots))

    candidates, representative_sample, median_area = build_candidate_pool(
        normal_dir=normal_dir,
        train_dir=train_dir,
        stats_cache=stats_cache,
        component_token=args.component_token,
        defect_token=args.defect_token,
        candidate_count=args.candidate_count,
        base_seed=args.base_seed,
    )
    print(
        "Selected representative sample:",
        representative_sample["image_path"],
        f"(mask area={representative_sample['mask_area']}, median area={median_area:.1f})",
    )

    requested_device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if requested_device == "cuda" and not torch.cuda.is_available():
        requested_device = "cpu"
    device = torch.device(requested_device)
    weight_dtype = resolve_weight_dtype(train_config.get("training", {}).get("mixed_precision", "no"))
    if device.type == "cpu":
        weight_dtype = torch.float32
    model_source = resolve_model_source(train_config["paths"])

    prompt = f"a photo of {args.component_token} with {args.defect_token}"
    method_to_maps = collect_attention_maps(
        model_source=model_source,
        weight_dtype=weight_dtype,
        device=device,
        representative_sample=representative_sample,
        prompt=prompt,
        defect_token=args.defect_token,
        checkpoint_slots=checkpoint_slots,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        negative_prompt=negative_prompt,
        base_seed=args.base_seed,
    )

    vmin, vmax = compute_shared_color_range(method_to_maps)
    png_path, pdf_path = save_compact_figure(
        output_dir=output_dir,
        representative_sample=representative_sample,
        checkpoint_slots=checkpoint_slots,
        method_to_maps=method_to_maps,
        vmin=vmin,
        vmax=vmax,
        dpi=args.dpi,
    )

    manifest = build_manifest(
        args=args,
        train_config=train_config,
        infer_config=infer_config,
        weights_root=weights_root,
        normal_dir=normal_dir,
        train_dir=train_dir,
        stats_cache=stats_cache,
        representative_sample=representative_sample,
        candidates=candidates,
        median_area=median_area,
        checkpoint_slots=checkpoint_slots,
        method_to_maps=method_to_maps,
        vmin=vmin,
        vmax=vmax,
        output_paths={"png": png_path, "pdf": pdf_path},
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        negative_prompt=negative_prompt,
    )
    manifest_path = output_dir / "lens_scratch_evolution_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"Saved compact figure to {png_path}")
    print(f"Saved compact figure to {pdf_path}")
    print(f"Saved manifest to {manifest_path}")


if __name__ == "__main__":
    main()
