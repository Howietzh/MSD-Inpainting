import argparse
import gc
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from diffusers import DDIMScheduler, StableDiffusionInpaintPipeline
from peft import PeftModel
from PIL import Image
from transformers import CLIPTokenizer

from dataset.normal_dataset import NormalComponentDataset
from utils.mask_ops import DefectMaskEngine
from utils.runtime import resolve_model_source, resolve_weight_dtype


TARGET_SIZE = 512
MASK_INSET_RATIO = 0.22
EXPERIMENTS = (
    ("defectfill_origin", "defectfill_origin"),
    ("increase_text_encoder_learning_rates", "msd_inpainting"),
)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate qualitative samples with shared masks for two LoRA experiments.")
    parser.add_argument("--train-config", type=str, default="configs/train_config.yaml")
    parser.add_argument("--infer-config", type=str, default="configs/inference_config.yaml")
    parser.add_argument("--weights-root", type=str, default="defectfill_lora_weights")
    parser.add_argument("--normal-dir", type=str, default=None)
    parser.add_argument("--train-dir", type=str, default=None)
    parser.add_argument("--stats-cache", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="qualitative_figure")
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--negative-prompt", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-tasks", type=int, default=None)
    return parser.parse_args()


def load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def choose_existing_path(explicit_value, candidate_values, create_parent: bool = False) -> Path:
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


def resolve_best_lora_dir(experiment_dir: Path) -> tuple[Path, str]:
    if is_complete_lora_dir(experiment_dir):
        return experiment_dir, "final"

    candidates = []
    for child in experiment_dir.glob("checkpoint-epoch-*"):
        if not child.is_dir() or not is_complete_lora_dir(child):
            continue
        try:
            epoch = int(child.name.rsplit("-", 1)[-1])
        except ValueError:
            continue
        candidates.append((epoch, child))

    if not candidates:
        raise FileNotFoundError(
            f"No complete final weights or checkpoint-epoch-* weights found under {experiment_dir}"
        )

    epoch, path = max(candidates, key=lambda item: item[0])
    return path, f"checkpoint-epoch-{epoch}"


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


def select_normal_sample(normal_dir: Path, component_token: str, seed: int):
    dataset = NormalComponentDataset(data_dir=str(normal_dir), size=TARGET_SIZE, target_comp=component_token)
    if len(dataset) == 0:
        raise ValueError(f"No normal samples found for component token {component_token}.")
    dataset_index = random.Random(int(seed)).randrange(len(dataset))
    sample = dataset[dataset_index]
    return {
        "dataset_index": int(dataset_index),
        "image_path": sample["image_path"],
        "component_mask_path": sample["component_mask_path"],
        "image_np": tensor_to_rgb_image(sample["pixel_values"]),
        "component_mask_np": tensor_mask_to_uint8(sample["mask_values"]),
    }


def build_shared_mask(mask_engine, component_mask_np: np.ndarray, defect_token: str, seed: int):
    seed_everything(seed)
    params = mask_engine.sample_generation_params(defect_token)
    defect_mask_np, details = mask_engine.generate_dynamic_mask_with_params(
        component_mask_np,
        defect_token,
        params,
        return_details=True,
    )
    return defect_mask_np, params, details


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


def build_image_tensor(image_np: np.ndarray, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    tensor = torch.from_numpy(image_np).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    return tensor.to(device=device, dtype=dtype)


def build_mask_tensor(mask_np: np.ndarray, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    tensor = torch.from_numpy(mask_np).float().unsqueeze(0).unsqueeze(0) / 255.0
    return tensor.to(device=device, dtype=dtype)


def resolve_pipe_device(pipe) -> torch.device:
    return next(pipe.unet.parameters()).device


def run_generation(
    *,
    pipe,
    prompt: str,
    image_np: np.ndarray,
    defect_mask_np: np.ndarray,
    seed: int,
    num_inference_steps: int,
    guidance_scale: float,
    negative_prompt: str,
) -> Image.Image:
    pipe_device = resolve_pipe_device(pipe)
    image_tensor = build_image_tensor(image_np, pipe_device, pipe.unet.dtype)
    mask_tensor = build_mask_tensor(defect_mask_np, pipe_device, pipe.unet.dtype)
    generator = torch.Generator(device=pipe_device).manual_seed(int(seed))
    kwargs = {
        "prompt": [prompt],
        "image": image_tensor,
        "mask_image": mask_tensor,
        "num_inference_steps": int(num_inference_steps),
        "guidance_scale": float(guidance_scale),
        "generator": generator,
    }
    if negative_prompt:
        kwargs["negative_prompt"] = [negative_prompt]

    with torch.no_grad():
        return pipe(**kwargs).images[0]


def add_mask_inset(image: Image.Image, defect_mask_np: np.ndarray, ratio: float = MASK_INSET_RATIO) -> Image.Image:
    output = image.convert("RGB").copy()
    width, height = output.size
    inset_size = max(32, int(round(width * ratio)))
    border = max(2, inset_size // 32)
    margin = max(8, inset_size // 12)

    mask_rgb = Image.fromarray(defect_mask_np, mode="L").resize(
        (inset_size, inset_size),
        resample=Image.Resampling.NEAREST,
    ).convert("RGB")
    tile = Image.new("RGB", (inset_size + 2 * border, inset_size + 2 * border), "white")
    tile.paste(mask_rgb, (border, border))
    output.paste(tile, (width - tile.width - margin, height - tile.height - margin))
    return output


def safe_token(token: str) -> str:
    return token.strip("<>").replace("/", "_")


def serialize_path(path: Path) -> str:
    return str(path.expanduser().resolve())


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    train_config = load_yaml(args.train_config)
    infer_config = load_yaml(args.infer_config)
    infer_options = infer_config.get("inference", {})

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    weights_root = choose_existing_path(
        args.weights_root,
        [
            repo_root / "defectfill_lora_weights",
            repo_root.parent / "defectfill_lora_weights",
            train_config["paths"].get("output_dir"),
        ],
    )
    normal_dir = choose_existing_path(
        args.normal_dir,
        [
            infer_config["paths"].get("normal_dir"),
            repo_root / "data" / "CCM-Defect" / "normal_components",
            repo_root.parent / "data" / "CCM-Defect" / "normal_components",
        ],
    )
    train_dir = choose_existing_path(
        args.train_dir,
        [
            infer_config["paths"].get("train_dir"),
            train_config["paths"].get("data_dir"),
            repo_root / "data" / "CCM-Defect" / "defect_train_concept",
            repo_root.parent / "data" / "CCM-Defect" / "defect_train_concept",
        ],
    )
    stats_cache = choose_existing_path(
        args.stats_cache,
        [
            output_dir / "defect_stats_cache.json",
            infer_config["paths"].get("stats_cache"),
        ],
        create_parent=True,
    )

    num_inference_steps = int(
        args.num_inference_steps
        if args.num_inference_steps is not None
        else infer_options.get("num_inference_steps", 30)
    )
    guidance_scale = float(
        args.guidance_scale
        if args.guidance_scale is not None
        else infer_options.get("guidance_scale", 7.5)
    )
    negative_prompt = (
        args.negative_prompt
        if args.negative_prompt is not None
        else infer_options.get("negative_prompt", "")
    )
    base_seed = int(args.base_seed)

    requested_device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if requested_device == "cuda" and not torch.cuda.is_available():
        requested_device = "cpu"
    device = torch.device(requested_device)
    weight_dtype = resolve_weight_dtype(train_config.get("training", {}).get("mixed_precision", "no"))
    if device.type == "cpu":
        weight_dtype = torch.float32

    model_source = resolve_model_source(train_config["paths"])
    tasks = list(infer_config.get("tasks", []))
    if args.max_tasks is not None:
        tasks = tasks[: max(0, int(args.max_tasks))]
    if not tasks:
        raise ValueError("No qualitative tasks were selected.")

    mask_engine = DefectMaskEngine(train_dir=train_dir, cache_file=stats_cache, target_size=TARGET_SIZE)
    mask_engine.load_or_compute_stats(tasks)
    if stats_cache.exists():
        with open(stats_cache, "r", encoding="utf-8") as f:
            mask_engine.stats_cache = json.load(f)

    resolved_experiments = []
    for experiment_dir_name, output_label in EXPERIMENTS:
        lora_dir, weight_source = resolve_best_lora_dir(weights_root / experiment_dir_name)
        resolved_experiments.append(
            {
                "experiment": experiment_dir_name,
                "label": output_label,
                "lora_dir": lora_dir,
                "weight_source": weight_source,
            }
        )

    manifest = {
        "train_config": serialize_path(Path(args.train_config)),
        "infer_config": serialize_path(Path(args.infer_config)),
        "weights_root": serialize_path(weights_root),
        "normal_dir": serialize_path(normal_dir),
        "train_dir": serialize_path(train_dir),
        "stats_cache": serialize_path(stats_cache),
        "output_dir": serialize_path(output_dir),
        "base_seed": base_seed,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "negative_prompt": negative_prompt,
        "mask_inset_ratio": MASK_INSET_RATIO,
        "experiments": [
            {
                "experiment": item["experiment"],
                "label": item["label"],
                "lora_dir": serialize_path(item["lora_dir"]),
                "weight_source": item["weight_source"],
            }
            for item in resolved_experiments
        ],
        "tasks": [],
    }

    for task_idx, task in enumerate(tasks):
        defect_token = task["defect"]
        component_token = task["comp"]
        sample_seed = base_seed + task_idx
        mask_seed = base_seed + task_idx * 1000
        generation_seed = base_seed + task_idx * 10000
        prompt = f"a photo of {component_token} with {defect_token}"
        print(f"[Task {task_idx}] {defect_token} on {component_token}")

        sample = select_normal_sample(normal_dir, component_token, sample_seed)
        defect_mask_np, mask_params, mask_details = build_shared_mask(
            mask_engine,
            sample["component_mask_np"],
            defect_token,
            mask_seed,
        )

        task_outputs = []
        for experiment in resolved_experiments:
            print(f"  generating with {experiment['label']} from {experiment['weight_source']}")
            pipe = build_pipe(model_source, weight_dtype, device, experiment["lora_dir"])
            try:
                generated = run_generation(
                    pipe=pipe,
                    prompt=prompt,
                    image_np=sample["image_np"],
                    defect_mask_np=defect_mask_np,
                    seed=generation_seed,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    negative_prompt=negative_prompt,
                )
            finally:
                del pipe
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            final_image = add_mask_inset(generated, defect_mask_np)
            base_name = f"{task_idx:02d}_{safe_token(defect_token)}_{safe_token(component_token)}_{experiment['label']}"
            output_path = output_dir / f"{base_name}.png"
            final_image.save(output_path)
            task_outputs.append(
                {
                    "experiment": experiment["experiment"],
                    "label": experiment["label"],
                    "weight_source": experiment["weight_source"],
                    "lora_dir": serialize_path(experiment["lora_dir"]),
                    "output_path": serialize_path(output_path),
                }
            )

        manifest["tasks"].append(
            {
                "task_index": task_idx,
                "defect_token": defect_token,
                "component_token": component_token,
                "prompt": prompt,
                "sample_seed": sample_seed,
                "mask_seed": mask_seed,
                "generation_seed": generation_seed,
                "normal_dataset_index": sample["dataset_index"],
                "normal_image_path": sample["image_path"],
                "component_mask_path": sample["component_mask_path"],
                "mask_area": int(cv2.countNonZero(defect_mask_np)),
                "mask_params": mask_params,
                "mask_details": mask_details,
                "outputs": task_outputs,
            }
        )

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"Saved qualitative outputs to {output_dir}")
    print(f"Saved manifest to {manifest_path}")


if __name__ == "__main__":
    main()
