import torch
import torch.nn.functional as F
import torch.nn as nn

class DefectFillLoss(nn.Module):
    def __init__(self, lambda_def=0.5, lambda_obj=0.2, lambda_attn=0.2, object_loss_alpha=0.25):
        super().__init__()
        self.lambda_def = lambda_def
        self.lambda_obj = lambda_obj
        self.lambda_attn = lambda_attn
        self.object_loss_alpha = object_loss_alpha

    def _normalize_attention_map(self, attention_map):
        attn_min = attention_map.amin(dim=(-2, -1), keepdim=True)
        attn_max = attention_map.amax(dim=(-2, -1), keepdim=True)
        return (attention_map - attn_min) / (attn_max - attn_min + 1e-6)

    def _masked_mse(self, model_pred, target_noise, mask):
        mse_error = (model_pred - target_noise) ** 2
        weighted_error = mse_error * mask
        denom = mask.sum() * model_pred.shape[1]
        if denom <= 0:
            return torch.tensor(0.0, device=model_pred.device, dtype=model_pred.dtype)
        return weighted_error.sum() / denom

    def forward(self, defect_model_pred, object_model_pred, target_noise, defect_mask, attention_map=None):
        _, _, h, w = defect_model_pred.shape

        latent_defect_mask = F.interpolate(defect_mask, size=(h, w), mode="nearest")
        object_weight_mask = latent_defect_mask + self.object_loss_alpha * (1.0 - latent_defect_mask)

        # L_def: 缺陷去噪损失
        loss_def = self._masked_mse(defect_model_pred, target_noise, latent_defect_mask)

        # L_obj: 论文 Eq.(7) 的加权全图损失
        loss_obj = self._masked_mse(object_model_pred, target_noise, object_weight_mask)

        # L_attn: 交叉注意力特征对齐损失
        loss_attn = torch.tensor(0.0, device=defect_model_pred.device, dtype=defect_model_pred.dtype)
        if attention_map is not None:
            normalized_attention_map = self._normalize_attention_map(attention_map.float())
            loss_attn = F.mse_loss(normalized_attention_map, latent_defect_mask.float())

        total_loss = (self.lambda_def * loss_def) + (self.lambda_obj * loss_obj) + (self.lambda_attn * loss_attn)
        return total_loss, {"loss_def": loss_def.item(), "loss_obj": loss_obj.item(), "loss_attn": loss_attn.item()}
