def get_ablation_config(config: dict) -> dict:
    return config.get("ablation", {}) or {}


def use_textual_inversion(config: dict) -> bool:
    return bool(get_ablation_config(config).get("use_textual_inversion", True))


def use_defect_sensitive_loss(config: dict) -> bool:
    return bool(get_ablation_config(config).get("use_defect_sensitive_loss", True))


def use_dual_mask_attention(config: dict) -> bool:
    return bool(get_ablation_config(config).get("use_dual_mask_attention", True))


def build_conditioning_prompt(config: dict, component_token: str, defect_token: str) -> str:
    if use_textual_inversion(config):
        component_text = component_token
        defect_text = defect_token
    else:
        phrases = config.get("token_init_phrases", {}) or {}
        component_text = phrases.get(component_token, component_token.strip("<>").replace("_", " "))
        defect_text = phrases.get(defect_token, defect_token.strip("<>").replace("_", " "))
    return f"a photo of {component_text} with {defect_text}"
