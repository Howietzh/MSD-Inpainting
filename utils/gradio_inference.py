import json
import random
from pathlib import Path

import cv2
import lpips
import numpy as np
import torch
import yaml
from PIL import Image
from torchvision import transforms

from diffusers import DDIMScheduler, StableDiffusionInpaintPipeline
from peft import PeftModel
from transformers import CLIPTokenizer

from utils.mask_ops import DefectMaskEngine
from utils.runtime import resolve_model_source, resolve_weight_dtype


class InteractiveDefectFillEngine:
    def __init__(
        self,
        train_config_path: str,
        infer_config_path: str,
        lora_weights: str | None = None,
        normal_dir: str | None = None,
        stats_cache: str | None = None,
        device: str | None = None,
    ):
        self.train_config_path = str(train_config_path)
        self.infer_config_path = str(infer_config_path)

        with open(self.train_config_path, "r", encoding="utf-8") as f:
            self.train_config = yaml.safe_load(f)
        with open(self.infer_config_path, "r", encoding="utf-8") as f:
            self.infer_config_full = yaml.safe_load(f)

        self.paths = dict(self.infer_config_full["paths"])
        if lora_weights:
            self.paths["lora_weights"] = lora_weights
        if normal_dir:
            self.paths["normal_dir"] = normal_dir
        if stats_cache:
            self.paths["stats_cache"] = stats_cache

        requested_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if requested_device == "cuda" and not torch.cuda.is_available():
            requested_device = "cpu"
        self.device = torch.device(requested_device)

        self.weight_dtype = resolve_weight_dtype(self.train_config.get("training", {}).get("mixed_precision", "no"))
        if self.device.type == "cpu":
            self.weight_dtype = torch.float32

        self.component_tokens = list(self.train_config.get("component_tokens", []))
        self.defect_tokens = list(self.train_config.get("defect_tokens", []))
        self.valid_defects_by_component = self._build_valid_defects_by_component()
        self.mask_engine = DefectMaskEngine(
            train_dir=Path(self.paths["train_dir"]),
            cache_file=Path(self.paths["stats_cache"]),
        )
        self.mask_engine.load_or_compute_stats([{"defect": token} for token in self.defect_tokens])
        with open(self.mask_engine.cache_file, "r", encoding="utf-8") as f:
            self.mask_engine.stats_cache = json.load(f)

        self.normal_dir = Path(self.paths["normal_dir"])
        self.normal_records_by_component = self._load_normal_records()
        self.pipe = self._build_pipe(Path(self.paths["lora_weights"]))
        lpips_backbone = self.infer_config_full.get("inference", {}).get("lpips_backbone", "alex")
        self.lpips_model = lpips.LPIPS(net=lpips_backbone).to(self.device)
        self.lpips_model.eval()
        self.lpips_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        self.image_transform = transforms.Compose([
            transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])

    def _build_pipe(self, lora_dir: Path):
        model_source = resolve_model_source(self.train_config["paths"])
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            model_source,
            torch_dtype=self.weight_dtype,
            local_files_only=True,
        )
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.tokenizer = CLIPTokenizer.from_pretrained(str(lora_dir / "tokenizer"))
        pipe.text_encoder.resize_token_embeddings(len(pipe.tokenizer))
        pipe.text_encoder = PeftModel.from_pretrained(pipe.text_encoder, str(lora_dir / "text_encoder_lora"))
        pipe.unet = PeftModel.from_pretrained(pipe.unet, str(lora_dir / "unet_lora"))
        pipe.to(self.device)
        pipe.set_progress_bar_config(disable=True)
        pipe.enable_attention_slicing()
        return pipe

    def _load_normal_records(self):
        metadata_path = self.normal_dir / "metadata.jsonl"
        if not metadata_path.exists():
            raise FileNotFoundError(f"❌ 找不到 metadata 文件: {metadata_path}")

        grouped = {token: [] for token in self.component_tokens}
        with open(metadata_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                record = json.loads(line.strip())
                object_token = record.get("object_token")
                if object_token not in grouped:
                    continue
                image_path = self.normal_dir / record["image_path"]
                component_mask_path = self.normal_dir / record["component_mask_path"]
                if not image_path.exists():
                    raise FileNotFoundError(f"Normal image missing at metadata line {line_idx}: {image_path}")
                if not component_mask_path.exists():
                    raise FileNotFoundError(f"Component mask missing at metadata line {line_idx}: {component_mask_path}")
                grouped[object_token].append(record)
        return grouped

    def _build_valid_defects_by_component(self):
        mapping = {token: [] for token in self.component_tokens}
        for task in self.infer_config_full.get("tasks", []):
            component = task.get("comp")
            defect = task.get("defect")
            if component in mapping and defect and defect not in mapping[component]:
                mapping[component].append(defect)
        return mapping

    def get_component_choices(self):
        return self.component_tokens

    def get_defect_choices(self, component_token=None):
        if component_token is None:
            return self.defect_tokens
        filtered = self.valid_defects_by_component.get(component_token, [])
        return filtered or self.defect_tokens

    def get_normal_image_choices(self, component_token):
        records = self.normal_records_by_component.get(component_token, [])
        return [record["image_path"] for record in records]

    def get_param_spec(self, defect_token):
        kind = self.mask_engine.get_defect_kind(defect_token)
        defaults = self.mask_engine.get_default_param_values(defect_token)
        stats = self.mask_engine.stats_cache.get(defect_token, self.mask_engine._build_default_stats(defect_token))

        if kind == "scratch":
            return {
                "kind": kind,
                "length": {
                    "label": "Length",
                    "value": defaults["length"],
                    "minimum": 1,
                    "maximum": max(defaults["length"] * 2, int(stats["length"]["p90"]) * 2),
                },
                "thickness": {
                    "label": "Thickness",
                    "value": defaults["thickness"],
                    "minimum": 1,
                    "maximum": max(defaults["thickness"] * 2, int(stats["thickness"]["p90"]) * 3),
                },
            }
        if kind == "tear":
            return {
                "kind": kind,
                "length": {
                    "label": "Length",
                    "value": defaults["length"],
                    "minimum": 1,
                    "maximum": max(defaults["length"] * 2, int(stats["length"]["p90"]) * 2),
                },
                "width": {
                    "label": "Width",
                    "value": defaults["width"],
                    "minimum": 1,
                    "maximum": max(defaults["width"] * 2, int(stats["width"]["p90"]) * 3),
                },
            }
        return {
            "kind": kind,
            "radius": {
                "label": "Radius",
                "value": defaults["radius"],
                "minimum": 1,
                "maximum": max(defaults["radius"] * 3, int(stats["radius"]["p90"]) * 3),
            },
            "count": {
                "label": "Count",
                "value": defaults["count"],
                "minimum": 1,
                "maximum": max(defaults["count"] + 3, int(stats["count"]["p90"]) + 3),
            },
        }

    def _select_record(self, component_token, selected_image_path, use_random_image, base_seed):
        records = self.normal_records_by_component.get(component_token, [])
        if not records:
            raise ValueError(f"当前组件 {component_token} 没有可用正常样本。")

        if use_random_image or not selected_image_path:
            rng = random.Random(int(base_seed))
            return rng.choice(records)

        for record in records:
            if record["image_path"] == selected_image_path:
                return record
        raise ValueError(f"未找到所选正常图: {selected_image_path}")

    def _load_record_assets(self, record):
        image = Image.open(self.normal_dir / record["image_path"]).convert("RGB")
        component_mask = Image.open(self.normal_dir / record["component_mask_path"]).convert("L")
        image = image.resize((512, 512), resample=Image.BILINEAR)
        component_mask = component_mask.resize((512, 512), resample=Image.NEAREST)
        image_np = np.array(image)
        comp_mask_np = (np.array(component_mask) > 0).astype(np.uint8) * 255
        return image, component_mask, image_np, comp_mask_np

    def preview_record(self, component_token, selected_image_path, use_random_image, base_seed):
        record = self._select_record(component_token, selected_image_path, use_random_image, base_seed)
        image, component_mask, _, _ = self._load_record_assets(record)
        return image, component_mask, record["image_path"]

    def _build_visualization(self, init_img_np, defect_mask_np, generated_img_pil):
        mask_bool = defect_mask_np > 127
        overlay = init_img_np.copy()
        overlay[mask_bool] = (
            0.6 * overlay[mask_bool] + 0.4 * np.array([255, 80, 80], dtype=np.float32)
        ).astype(np.uint8)
        generated_img_np = np.array(generated_img_pil)
        canvas = np.concatenate([init_img_np, overlay, generated_img_np], axis=1)
        return Image.fromarray(canvas), Image.fromarray(overlay)

    def _prepare_lpips_crop_pair(self, img_orig, img_gen, mask_np):
        coords = cv2.findNonZero(mask_np)
        if coords is None:
            return None
        x, y, w, h = cv2.boundingRect(coords)
        if h <= 0 or w <= 0:
            return None

        orig_crop = cv2.resize(img_orig[y:y + h, x:x + w], (224, 224), interpolation=cv2.INTER_CUBIC)
        gen_crop = cv2.resize(img_gen[y:y + h, x:x + w], (224, 224), interpolation=cv2.INTER_CUBIC)
        return self.lpips_transform(orig_crop), self.lpips_transform(gen_crop)

    def _calculate_lpips_score(self, img_orig_np, img_gen_np, mask_np):
        crop_pair = self._prepare_lpips_crop_pair(img_orig_np, img_gen_np, mask_np)
        if crop_pair is None:
            return 0.0

        orig_tensor, gen_tensor = crop_pair
        with torch.no_grad():
            score = self.lpips_model(
                orig_tensor.unsqueeze(0).to(self.device),
                gen_tensor.unsqueeze(0).to(self.device),
            )
        return float(score.item())

    def _build_mask_tensor(self, mask_np):
        return torch.from_numpy(mask_np).float().unsqueeze(0).unsqueeze(0) / 255.0

    def _build_image_tensor(self, image):
        return self.image_transform(image).unsqueeze(0)

    def _resolve_actual_params(
        self,
        defect_token,
        random_use_cache_range,
        length,
        thickness,
        width,
        radius,
        count,
    ):
        if random_use_cache_range:
            return self.mask_engine.sample_generation_params(defect_token)

        kind = self.mask_engine.get_defect_kind(defect_token)
        if kind == "scratch":
            return {"length": int(length), "thickness": int(thickness)}
        if kind == "tear":
            return {"length": int(length), "width": int(width)}
        return {"radius": int(radius), "count": int(count)}

    def generate_mask_preview(
        self,
        component_token,
        defect_token,
        selected_image_path,
        use_random_image,
        base_seed,
        refresh_index,
        length,
        thickness,
        width,
        radius,
        count,
        random_use_cache_range,
    ):
        record = self._select_record(component_token, selected_image_path, use_random_image, base_seed)
        image, component_mask, image_np, comp_mask_np = self._load_record_assets(record)

        mask_seed = int(base_seed) + int(refresh_index)
        random.seed(mask_seed)
        np.random.seed(mask_seed)
        torch.manual_seed(mask_seed)

        actual_params = self._resolve_actual_params(
            defect_token,
            random_use_cache_range,
            length,
            thickness,
            width,
            radius,
            count,
        )
        defect_mask_np, mask_details = self.mask_engine.generate_dynamic_mask_with_params(
            comp_mask_np,
            defect_token,
            actual_params,
            return_details=True,
        )
        defect_mask_pil = Image.fromarray(defect_mask_np, mode="L")
        _, overlay_image = self._build_visualization(image_np, defect_mask_np, image)

        mask_payload = {
            "component_token": component_token,
            "defect_token": defect_token,
            "selected_image_path": record["image_path"],
            "prompt": f"a photo of {component_token} with {defect_token}",
            "seed": int(base_seed),
            "mask_seed": mask_seed,
            "refresh_index": int(refresh_index),
            "random_use_cache_range": bool(random_use_cache_range),
            "mask_params": actual_params,
            "mask_details": mask_details,
            "defect_mask": defect_mask_np.tolist(),
        }

        info = {
            "stage": "mask_ready",
            "object_token": component_token,
            "defect_token": defect_token,
            "normal_image_path": record["image_path"],
            "prompt": mask_payload["prompt"],
            "seed": int(base_seed),
            "mask_seed": mask_seed,
            "refresh_index": int(refresh_index),
            "random_use_cache_range": bool(random_use_cache_range),
            "mask_params": actual_params,
        }
        if mask_details:
            info.update(mask_details)

        return {
            "original": image,
            "component_mask": component_mask,
            "defect_mask": defect_mask_pil,
            "overlay": overlay_image,
            "info": info,
            "mask_payload": mask_payload,
            "selected_image_path": record["image_path"],
        }

    def generate_from_mask(
        self,
        mask_payload,
        num_inference_steps,
        guidance_scale,
        negative_prompt,
        num_lfs_samples,
    ):
        if not mask_payload:
            raise ValueError("请先生成并确认 defect mask。")

        component_token = mask_payload["component_token"]
        defect_token = mask_payload["defect_token"]
        record = self._select_record(
            component_token=component_token,
            selected_image_path=mask_payload["selected_image_path"],
            use_random_image=False,
            base_seed=mask_payload["seed"],
        )
        image, component_mask, image_np, _ = self._load_record_assets(record)
        defect_mask_np = np.array(mask_payload["defect_mask"], dtype=np.uint8)
        defect_mask_pil = Image.fromarray(defect_mask_np, mode="L")
        init_img_tensor = self._build_image_tensor(image).to(self.device, dtype=self.weight_dtype)
        mask_tensor = self._build_mask_tensor(defect_mask_np).to(self.device, dtype=self.weight_dtype)
        prompt = mask_payload["prompt"]

        candidate_gallery = []
        candidate_scores = []
        candidate_images = []
        best_idx = 0
        best_score = float("-inf")

        for sample_idx in range(int(num_lfs_samples)):
            generator = torch.Generator(device=self.device).manual_seed(int(mask_payload["seed"]) + sample_idx)
            output_image = self.pipe(
                prompt=[prompt],
                negative_prompt=[negative_prompt],
                image=init_img_tensor,
                mask_image=mask_tensor,
                num_inference_steps=int(num_inference_steps),
                guidance_scale=float(guidance_scale),
                generator=[generator],
            ).images[0]
            output_np = np.array(output_image)
            score = self._calculate_lpips_score(image_np, output_np, defect_mask_np)
            candidate_scores.append(score)
            candidate_images.append(output_image)
            candidate_gallery.append((output_image, f"Candidate {sample_idx} | LPIPS {score:.4f}"))
            if score > best_score:
                best_score = score
                best_idx = sample_idx

        best_image = candidate_images[best_idx]
        triptych, overlay_image = self._build_visualization(image_np, defect_mask_np, best_image)
        info = {
            "stage": "generation_done",
            "object_token": component_token,
            "defect_token": defect_token,
            "normal_image_path": record["image_path"],
            "prompt": prompt,
            "seed": int(mask_payload["seed"]),
            "random_use_cache_range": bool(mask_payload["random_use_cache_range"]),
            "mask_params": mask_payload["mask_params"],
            "candidate_scores": candidate_scores,
            "best_candidate_index": best_idx,
            "best_score": best_score,
        }
        if mask_payload.get("mask_details"):
            info.update(mask_payload["mask_details"])

        return {
            "original": image,
            "component_mask": component_mask,
            "defect_mask": defect_mask_pil,
            "overlay": overlay_image,
            "candidates": candidate_gallery,
            "best": best_image,
            "triptych": triptych,
            "info": info,
            "selected_image_path": record["image_path"],
        }
