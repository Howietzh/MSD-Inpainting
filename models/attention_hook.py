import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention


class StoreCrossAttnProcessor:
    def __init__(self):
        self.attention_maps = {}
        self.target_token_indices = None
        self.capture_mode = "direct"

    def _resolve_batch_index(self, batch_idx: int, token_count: int, batch_size: int) -> int:
        if (
            self.capture_mode == "cfg_conditional"
            and token_count > 0
            and batch_size == token_count * 2
        ):
            return batch_idx + token_count
        return batch_idx

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        temb: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        else:
            batch_size = hidden_states.shape[0]
            height = width = None

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
            sequence_length = hidden_states.shape[1]
        else:
            sequence_length = encoder_hidden_states.shape[1]
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        if self.target_token_indices:
            _, hw, _ = attention_probs.shape
            if input_ndim == 4:
                h, w = height, width
            else:
                h = w = int(hw ** 0.5)
            attn_map = attention_probs.view(batch_size, attn.heads, hw, -1)
            for target_name, token_indices in self.target_token_indices.items():
                sample_maps = []
                token_count = len(token_indices)
                for batch_idx, token_index in enumerate(token_indices):
                    attn_batch_idx = self._resolve_batch_index(batch_idx, token_count, batch_size)
                    if (
                        attn_batch_idx < 0
                        or attn_batch_idx >= attn_map.shape[0]
                        or token_index < 0
                        or token_index >= attn_map.shape[-1]
                    ):
                        sample_attn = torch.zeros(
                            1,
                            h,
                            w,
                            device=attn_map.device,
                            dtype=attn_map.dtype,
                        )
                    else:
                        sample_attn = attn_map[attn_batch_idx, :, :, token_index].view(attn.heads, h, w).mean(dim=0, keepdim=True)
                    sample_maps.append(sample_attn)
                self.attention_maps.setdefault(target_name, []).append(torch.stack(sample_maps, dim=0))

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

class AttentionStore:
    def __init__(self):
        self.processor = StoreCrossAttnProcessor()

    def register_to_unet(self, unet):
        attn_processors = {}
        for name, _ in unet.attn_processors.items():
            if "up_blocks" in name and "attn2" in name:
                attn_processors[name] = self.processor
            else:
                attn_processors[name] = unet.attn_processors[name]
        unet.set_attn_processor(attn_processors)

    def set_target_token_indices(self, token_indices, capture_mode: str = "direct"):
        self.processor.target_token_indices = {"default": token_indices}
        self.processor.capture_mode = capture_mode

    def set_named_target_token_indices(self, token_indices_by_name, capture_mode: str = "direct"):
        self.processor.target_token_indices = {
            name: indices for name, indices in token_indices_by_name.items() if indices
        }
        self.processor.capture_mode = capture_mode

    def _aggregate_single_target(self, attention_maps, target_size: int = 64):
        if not attention_maps:
            return None

        aggregated_map = None
        count = 0
        for target_attn in attention_maps:
            resized_attn = F.interpolate(target_attn, size=(target_size, target_size), mode="bilinear", align_corners=False)
            if aggregated_map is None:
                aggregated_map = resized_attn
            else:
                aggregated_map += resized_attn
            count += 1
        return aggregated_map / count

    def get_aggregated_attention(self, target_size: int = 64, target_name: str = "default"):
        return self._aggregate_single_target(self.processor.attention_maps.get(target_name, []), target_size=target_size)

    def get_aggregated_attentions(self, target_size: int = 64):
        return {
            target_name: self._aggregate_single_target(attention_maps, target_size=target_size)
            for target_name, attention_maps in self.processor.attention_maps.items()
        }

    def clear(self):
        self.processor.attention_maps = {}
        self.processor.target_token_indices = None
        self.processor.capture_mode = "direct"
