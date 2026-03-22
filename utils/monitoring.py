import io

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.utils as vutils
from PIL import Image


class TokenDriftMonitor:
    def __init__(self, text_encoder, tokenizer, defect_tokens, component_tokens):
        self.tokenizer = tokenizer
        self.groups = {
            "Defect": defect_tokens,
            "Component": component_tokens,
        }
        embedding_weight = text_encoder.get_input_embeddings().weight.detach().cpu()
        self.initial_embeddings = {}
        for tokens in self.groups.values():
            for token in tokens:
                token_id = tokenizer.convert_tokens_to_ids(token)
                self.initial_embeddings[token] = embedding_weight[token_id].clone()

    def log_token_drift(self, text_encoder, visualizer, step: int):
        if visualizer.writer is None:
            return

        embedding_weight = text_encoder.get_input_embeddings().weight.detach().cpu()
        for group_name, tokens in self.groups.items():
            for token in tokens:
                token_id = self.tokenizer.convert_tokens_to_ids(token)
                current_embedding = embedding_weight[token_id]
                drift = torch.norm(current_embedding - self.initial_embeddings[token], p=2).item()
                safe_token = token.strip("<>")
                visualizer.writer.add_scalar(
                    f"TokenDrift/{group_name}/{safe_token}",
                    drift,
                    step,
                )


class TensorBoardVisualizer:
    ATTENTION_CMAP = "magma"

    def __init__(self, accelerator):
        self.accelerator = accelerator
        self.writer = self._get_writer()

    def _get_writer(self):
        for tracker in self.accelerator.trackers:
            if tracker.name == "tensorboard":
                return tracker.tracker if hasattr(tracker, "tracker") else tracker.writer
        return None

    def _normalize_heatmap(self, tensor):
        tensor = tensor.detach().float()
        heatmap_min = tensor.amin(dim=(-2, -1), keepdim=True)
        heatmap_max = tensor.amax(dim=(-2, -1), keepdim=True)
        return (tensor - heatmap_min) / (heatmap_max - heatmap_min + 1e-8)

    def _apply_colormap(self, tensor, cmap_name=None):
        B, C, H, W = tensor.shape
        tensor = tensor.detach().float().clamp(0.0, 1.0)
        tensor_np = tensor.cpu().numpy()
        colored_images = []
        cmap = plt.get_cmap(cmap_name or self.ATTENTION_CMAP)

        for i in range(B):
            img = tensor_np[i, 0]
            colored = cmap(img)[:, :, :3]
            colored_images.append(torch.from_numpy(colored).permute(2, 0, 1))

        return torch.stack(colored_images).to(tensor.device)

    def log_attention_maps(self, attention_map, defect_mask, epoch):
        if self.writer is None or attention_map is None:
            return

        mask_vis = F.interpolate(defect_mask.detach().float(), size=attention_map.shape[-2:], mode="nearest")
        mask_vis_3c = mask_vis.repeat(1, 3, 1, 1)
        normalized_attention = self._normalize_heatmap(attention_map)
        attn_colored = self._apply_colormap(normalized_attention)
        comparison = torch.cat([mask_vis_3c, attn_colored], dim=3)
        grid = vutils.make_grid(comparison, nrow=1, normalize=False, pad_value=0.5)
        self.writer.add_image("Alignment/Mask_vs_Attention_Heatmap", grid, epoch)

    def log_random_box_diagnostics(
        self,
        input_image,
        defect_mask,
        random_mask,
        step,
        max_samples: int = 4,
    ):
        if self.writer is None:
            return

        if input_image.dim() == 3:
            input_image = input_image.unsqueeze(0)
        if defect_mask.dim() == 3:
            defect_mask = defect_mask.unsqueeze(0)
        if random_mask.dim() == 3:
            random_mask = random_mask.unsqueeze(0)

        batch_size = min(input_image.shape[0], max_samples)
        input_vis = input_image[:batch_size].detach().float().clamp(0.0, 1.0)
        defect_mask_vis = F.interpolate(
            defect_mask[:batch_size].detach().float(),
            size=input_vis.shape[-2:],
            mode="nearest",
        )
        random_mask_vis = F.interpolate(
            random_mask[:batch_size].detach().float(),
            size=input_vis.shape[-2:],
            mode="nearest",
        )

        defect_overlay = input_vis.clone()
        defect_overlay[:, 0] = torch.maximum(defect_overlay[:, 0], defect_mask_vis[:, 0] * 0.95)
        defect_overlay[:, 1] = defect_overlay[:, 1] * (1.0 - 0.45 * defect_mask_vis[:, 0])
        defect_overlay[:, 2] = defect_overlay[:, 2] * (1.0 - 0.45 * defect_mask_vis[:, 0])

        random_box_overlay = input_vis.clone()
        random_box_overlay = random_box_overlay * (1.0 - random_mask_vis)

        comparison = torch.cat([input_vis, defect_overlay, random_box_overlay], dim=3)
        grid = vutils.make_grid(comparison, nrow=1, normalize=False, pad_value=0.5)
        self.writer.add_image("Alignment/RandomBox_Diagnostics", grid, step)

    def log_combined_inference_panel(
        self,
        tag,
        input_image,
        defect_mask,
        component_mask,
        generated_image,
        defect_attention_map,
        component_attention_map,
        step,
    ):
        if self.writer is None:
            return

        if input_image.dim() == 3:
            input_image = input_image.unsqueeze(0)
        if defect_mask.dim() == 3:
            defect_mask = defect_mask.unsqueeze(0)
        if component_mask.dim() == 3:
            component_mask = component_mask.unsqueeze(0)
        if generated_image.dim() == 3:
            generated_image = generated_image.unsqueeze(0)

        input_vis = input_image.detach().float().clamp(0.0, 1.0)
        gen_vis = generated_image.detach().float().clamp(0.0, 1.0)

        defect_mask_vis = F.interpolate(defect_mask.detach().float(), size=input_vis.shape[-2:], mode="nearest")
        component_mask_vis = F.interpolate(component_mask.detach().float(), size=input_vis.shape[-2:], mode="nearest")

        defect_overlay = input_vis.clone()
        defect_overlay[:, 0] = torch.maximum(defect_overlay[:, 0], defect_mask_vis[:, 0] * 0.95)
        defect_overlay[:, 1] = defect_overlay[:, 1] * (1.0 - 0.45 * defect_mask_vis[:, 0])
        defect_overlay[:, 2] = defect_overlay[:, 2] * (1.0 - 0.45 * defect_mask_vis[:, 0])

        if defect_attention_map is not None:
            defect_attention_map = F.interpolate(
                defect_attention_map.detach().float(),
                size=input_vis.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            defect_attention_vis = self._apply_colormap(self._normalize_heatmap(defect_attention_map))
        else:
            defect_attention_vis = defect_mask_vis.repeat(1, 3, 1, 1)

        if component_attention_map is not None:
            component_attention_map = F.interpolate(
                component_attention_map.detach().float(),
                size=input_vis.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            component_attention_vis = self._apply_colormap(self._normalize_heatmap(component_attention_map))
        else:
            component_attention_vis = component_mask_vis.repeat(1, 3, 1, 1)

        comparison = torch.cat(
            [input_vis, defect_overlay, gen_vis, defect_attention_vis, component_attention_vis],
            dim=3,
        )
        grid = vutils.make_grid(comparison, nrow=1, normalize=False, pad_value=0.5)
        self.writer.add_image(tag, grid, step)

    def log_token_confusion_matrix(self, text_encoder, tokenizer, custom_tokens, epoch, tag, title):
        if self.writer is None or not custom_tokens:
            return

        token_ids = [tokenizer.convert_tokens_to_ids(tok) for tok in custom_tokens]
        embeddings = text_encoder.get_input_embeddings().weight.detach()
        target_embeddings = embeddings[token_ids]
        normalized_emb = F.normalize(target_embeddings, p=2, dim=1)
        sim_matrix = torch.matmul(normalized_emb, normalized_emb.T).cpu().numpy()

        fig, ax = plt.subplots(figsize=(8, 8))
        cax = ax.matshow(sim_matrix, cmap="viridis", vmin=-1.0, vmax=1.0)
        fig.colorbar(cax, fraction=0.046, pad=0.04)

        ax.set_xticks(range(len(custom_tokens)))
        ax.set_yticks(range(len(custom_tokens)))
        ax.xaxis.set_ticks_position("bottom")
        ax.set_xticklabels(custom_tokens, rotation=45, ha="right", fontsize=11, fontweight="bold")
        ax.set_yticklabels(custom_tokens, fontsize=11, fontweight="bold")

        for i in range(len(custom_tokens)):
            for j in range(len(custom_tokens)):
                text_color = "white" if sim_matrix[i, j] < 0.5 else "black"
                ax.text(
                    j,
                    i,
                    f"{sim_matrix[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=12,
                    fontweight="bold",
                )

        plt.title(f"{title} (Epoch {epoch})", pad=20, fontsize=14, fontweight="bold")
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)

        img = Image.open(buf).convert("RGB")
        img_tensor = T.ToTensor()(img)
        self.writer.add_image(tag, img_tensor, epoch)
