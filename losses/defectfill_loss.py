import torch
import torch.nn as nn
import torch.nn.functional as F


class DefectFillLoss(nn.Module):
    def __init__(
        self,
        lambda_rec=1.0,
        lambda_attn_def=0.2,
        lambda_attn_comp=0.05,
        defect_class_weights=None,
        dilation_kernel_size=5,
        use_defect_sensitive_weighting=True,
    ):
        super().__init__()
        self.lambda_rec = float(lambda_rec)
        self.lambda_attn_def = float(lambda_attn_def)
        self.lambda_attn_comp = float(lambda_attn_comp)
        self.defect_class_weights = defect_class_weights or {}
        self.dilation_kernel_size = int(dilation_kernel_size)
        self.use_defect_sensitive_weighting = bool(use_defect_sensitive_weighting)

        if self.dilation_kernel_size <= 0 or self.dilation_kernel_size % 2 == 0:
            raise ValueError("dilation_kernel_size must be a positive odd integer.")

    def _normalize_attention_map(self, attention_map):
        attn_min = attention_map.amin(dim=(-2, -1), keepdim=True)
        attn_max = attention_map.amax(dim=(-2, -1), keepdim=True)
        return (attention_map - attn_min) / (attn_max - attn_min + 1e-6)

    def _to_latent_binary_mask(self, mask, size):
        latent_mask = F.interpolate(mask.float(), size=size, mode="nearest")
        return (latent_mask > 0.5).float()

    def _dilate_mask(self, mask):
        padding = self.dilation_kernel_size // 2
        return F.max_pool2d(mask, kernel_size=self.dilation_kernel_size, stride=1, padding=padding)

    def _resolve_defect_weights(self, defect_tokens, device, dtype):
        weights = []
        available_tokens = sorted(self.defect_class_weights.keys())
        for defect_token in defect_tokens:
            if defect_token not in self.defect_class_weights:
                raise KeyError(
                    f"Missing class weight for defect token {defect_token!r}. Available tokens: {available_tokens}"
                )
            weights.append(float(self.defect_class_weights[defect_token]))

        return torch.tensor(weights, device=device, dtype=dtype).view(-1, 1, 1, 1)

    def _weighted_mse(self, pred, target, weight_map):
        squared_error = (pred - target) ** 2
        weighted_error = squared_error * weight_map
        denom = weight_map.sum() * pred.shape[1]
        if denom <= 0:
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        return weighted_error.sum() / denom

    def _weighted_attention_mse(self, attention_map, target_mask, weight_map):
        normalized_attention = self._normalize_attention_map(attention_map.float())
        squared_error = (normalized_attention - target_mask.float()) ** 2
        weighted_error = squared_error * weight_map
        denom = weight_map.sum()
        if denom <= 0:
            return torch.tensor(0.0, device=attention_map.device, dtype=attention_map.dtype)
        return weighted_error.sum() / denom

    def forward(
        self,
        model_pred,
        target_noise,
        defect_mask,
        component_mask,
        defect_tokens,
        defect_attention_map=None,
        component_attention_map=None,
    ):
        _, _, h, w = model_pred.shape
        latent_size = (h, w)

        latent_defect_mask = self._to_latent_binary_mask(defect_mask, latent_size)
        latent_component_mask = self._to_latent_binary_mask(component_mask, latent_size)
        latent_dilated_mask = self._dilate_mask(latent_defect_mask) * latent_component_mask
        latent_dilated_mask = (latent_dilated_mask > 0.5).float()

        defect_weights = self._resolve_defect_weights(
            defect_tokens, model_pred.device, model_pred.dtype
        )
        if self.use_defect_sensitive_weighting:
            reconstruction_weight_map = 1.0 + defect_weights * latent_dilated_mask
        else:
            reconstruction_weight_map = torch.ones_like(latent_defect_mask)

        loss_rec = self._weighted_mse(model_pred, target_noise, reconstruction_weight_map)

        loss_attn_def = torch.tensor(0.0, device=model_pred.device, dtype=model_pred.dtype)
        if defect_attention_map is not None:
            defect_attention_weights = 1.0 + defect_weights * latent_dilated_mask
            loss_attn_def = self._weighted_attention_mse(
                defect_attention_map,
                latent_dilated_mask,
                defect_attention_weights,
            )

        loss_attn_comp = torch.tensor(0.0, device=model_pred.device, dtype=model_pred.dtype)
        if component_attention_map is not None:
            loss_attn_comp = F.mse_loss(
                self._normalize_attention_map(component_attention_map.float()),
                latent_component_mask.float(),
            )

        total_loss = (
            (self.lambda_rec * loss_rec)
            + (self.lambda_attn_def * loss_attn_def)
            + (self.lambda_attn_comp * loss_attn_comp)
        )

        return total_loss, {
            "loss_rec": loss_rec.item(),
            "loss_attn_def": loss_attn_def.item(),
            "loss_attn_comp": loss_attn_comp.item(),
            "mean_defect_weight": defect_weights.mean().item(),
        }
