import torch
from peft import LoraConfig, get_peft_model
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import UNet2DConditionModel


def _resolve_init_phrase(token: str, token_init_phrases: dict[str, str]) -> str:
    if token in token_init_phrases:
        return token_init_phrases[token]
    raise KeyError(
        f"token_init_phrases 缺少 {token} 的初始化映射，请在配置文件中显式指定。"
    )


def _initialize_new_token_embeddings(
    tokenizer: CLIPTokenizer,
    text_encoder: CLIPTextModel,
    new_token_specs: list[tuple[str, int]],
    base_vocab_size: int,
    token_init_phrases: dict[str, str],
):
    if not new_token_specs:
        return

    embedding_weight = text_encoder.get_input_embeddings().weight.data
    print("🧩 初始化新增 Token Embedding:")
    with torch.no_grad():
        for token, token_id in new_token_specs:
            init_phrase = _resolve_init_phrase(token, token_init_phrases)
            phrase_token_ids = tokenizer.encode(init_phrase, add_special_tokens=False)
            phrase_token_ids = [idx for idx in phrase_token_ids if idx < base_vocab_size]

            if not phrase_token_ids:
                raise ValueError(
                    f"初始化短语 \"{init_phrase}\" 无法被 tokenizer 映射到基础词表，请为 {token} 提供可分词的短语。"
                )

            init_embedding = embedding_weight[phrase_token_ids].mean(dim=0)

            embedding_weight[token_id].copy_(init_embedding)
            print(f"  - {token:32s} <- \"{init_phrase}\"")


def setup_lora_and_tokens(
    tokenizer: CLIPTokenizer,
    text_encoder: CLIPTextModel,
    unet: UNet2DConditionModel,
    config: dict,
):
    use_textual_inversion = bool(
        (config.get("ablation", {}) or {}).get("use_textual_inversion", True)
    )
    # 1. 扩充词表：合并缺陷和组件 tokens
    defect_tokens = config.get("defect_tokens", [])
    component_tokens = config.get("component_tokens", [])
    token_init_phrases = config.get("token_init_phrases", {})
    all_new_tokens = defect_tokens + component_tokens if use_textual_inversion else []
    base_vocab_size = len(tokenizer)

    num_added = tokenizer.add_tokens(all_new_tokens)
    text_encoder.resize_token_embeddings(len(tokenizer))

    seen_token_ids = set()
    new_token_specs = []
    for token in all_new_tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id >= base_vocab_size and token_id not in seen_token_ids:
            new_token_specs.append((token, token_id))
            seen_token_ids.add(token_id)

    _initialize_new_token_embeddings(
        tokenizer,
        text_encoder,
        new_token_specs,
        base_vocab_size,
        token_init_phrases,
    )

    # 提取 ID 以供后续定位使用
    defect_token_ids = (
        [tokenizer.convert_tokens_to_ids(tok) for tok in defect_tokens]
        if use_textual_inversion
        else []
    )
    component_token_ids = (
        [tokenizer.convert_tokens_to_ids(tok) for tok in component_tokens]
        if use_textual_inversion
        else []
    )

    print(
        f"✅ 成功向词表添加了 {num_added} 个新概念 Token "
        f"(缺陷: {len(defect_token_ids)} 个, 组件: {len(component_token_ids)} 个, "
        f"textual inversion: {use_textual_inversion})。"
    )

    # 2. 注入 Text Encoder LoRA，并只训练新增 token 对应的 embedding 行
    rank = config["lora"]["rank"]
    dropout = float(config["lora"].get("dropout", 0.0))
    te_lora_config = LoraConfig(
        r=rank,
        lora_alpha=config["lora"]["alpha"],
        lora_dropout=dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        trainable_token_indices=[token_id for _, token_id in new_token_specs] or None,
    )
    text_encoder = get_peft_model(text_encoder, te_lora_config)

    # 3. 注入 U-Net LoRA (挂载在交叉注意力层)
    unet_lora_config = LoraConfig(
        r=rank,
        lora_alpha=config["lora"]["alpha"],
        lora_dropout=dropout,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    )
    unet = get_peft_model(unet, unet_lora_config)

    return tokenizer, text_encoder, unet, defect_token_ids, component_token_ids
