import os
import importlib
import importlib.util
from pathlib import Path

import torch


def resolve_weight_dtype(mixed_precision: str) -> torch.dtype:
    mixed_precision = (mixed_precision or "no").lower()
    if mixed_precision == "fp16":
        return torch.float16
    if mixed_precision == "bf16":
        return torch.bfloat16
    return torch.float32


def resolve_pretrained_variant(mixed_precision: str) -> str | None:
    mixed_precision = (mixed_precision or "no").lower()
    if mixed_precision == "fp16":
        return "fp16"
    return None


def _is_diffusers_model_dir(path: Path) -> bool:
    required_entries = ["model_index.json", "tokenizer", "text_encoder", "vae", "unet", "scheduler"]
    return path.is_dir() and all((path / entry).exists() for entry in required_entries)


def _find_cached_model_dir(model_id: str) -> str | None:
    if "/" not in model_id:
        return None

    org_name, repo_name = model_id.split("/", 1)
    home = Path.home()
    cache_roots = []
    modelscope_cache = os.environ.get("MODELSCOPE_CACHE")
    hf_home = os.environ.get("HF_HOME")
    if modelscope_cache:
        cache_roots.append(Path(modelscope_cache))
    cache_roots.extend([
        home / ".cache" / "modelscope",
        Path("/home/doctor/.cache/modelscope"),
    ])
    if hf_home:
        cache_roots.append(Path(hf_home) / "hub")
    cache_roots.extend([
        home / ".cache" / "huggingface" / "hub",
        Path("/home/doctor/.cache/huggingface/hub"),
    ])

    for root in cache_roots:
        candidates = [
            root / "hub" / "models" / org_name / repo_name,
            root / "hub" / org_name / repo_name,
            root / org_name / repo_name,
            root / f"models--{org_name}--{repo_name}",
        ]

        for candidate in candidates:
            if _is_diffusers_model_dir(candidate):
                return str(candidate.resolve())

            snapshots_dir = candidate / "snapshots"
            if snapshots_dir.is_dir():
                for snapshot_dir in sorted(snapshots_dir.iterdir(), reverse=True):
                    if _is_diffusers_model_dir(snapshot_dir):
                        return str(snapshot_dir.resolve())

    return None


def resolve_model_source(paths_config: dict) -> str:
    model_id = paths_config["model_id"]
    configured_local_path = paths_config.get("local_model_path")
    env_local_path = os.environ.get("DEFECTFILL_MODEL_DIR")

    for candidate in (configured_local_path, env_local_path, model_id):
        if candidate and Path(candidate).expanduser().is_dir():
            return str(Path(candidate).expanduser().resolve())

    cached_model_dir = _find_cached_model_dir(model_id)
    if cached_model_dir is not None:
        return cached_model_dir

    cache_dir = paths_config.get("model_cache_dir")
    cache_dir = str(Path(cache_dir).expanduser()) if cache_dir else None

    snapshot_download = None
    if importlib.util.find_spec("modelscope") is not None:
        snapshot_download = importlib.import_module("modelscope").snapshot_download

    if snapshot_download is None:
        raise RuntimeError(
            "未找到可用的本地模型目录，且当前环境未安装 modelscope。"
            "请执行以下任一操作后重试："
            "1) 在 configs/train_config.yaml 的 paths.local_model_path 中填写本地模型目录；"
            "2) 设置环境变量 DEFECTFILL_MODEL_DIR；"
            "3) 安装 modelscope 后再自动下载模型。"
        )

    try:
        return snapshot_download(model_id, cache_dir=cache_dir)
    except Exception as exc:
        raise RuntimeError(
            "无法解析预训练模型。请执行以下任一操作后重试："
            "1) 在 configs/train_config.yaml 的 paths.local_model_path 中填写本地模型目录；"
            "2) 设置环境变量 DEFECTFILL_MODEL_DIR；"
            "3) 确保可以访问 ModelScope 以自动下载模型。"
        ) from exc
