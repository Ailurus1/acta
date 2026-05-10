from __future__ import annotations

TARGET_LAYER_PRESETS: dict[str, list[str]] = {
    "distilbert/distilbert-base-uncased": [
        "distilbert.transformer.layer.*.attention.q_lin",
        "distilbert.transformer.layer.*.attention.k_lin",
        "distilbert.transformer.layer.*.attention.v_lin",
        "distilbert.transformer.layer.*.attention.out_lin",
        "distilbert.transformer.layer.*.ffn.lin1",
        "distilbert.transformer.layer.*.ffn.lin2",
    ],
    "distilbert/distilbert-base-uncased-finetuned-sst-2-english": [
        "distilbert.transformer.layer.*.attention.q_lin",
        "distilbert.transformer.layer.*.attention.k_lin",
        "distilbert.transformer.layer.*.attention.v_lin",
        "distilbert.transformer.layer.*.attention.out_lin",
        "distilbert.transformer.layer.*.ffn.lin1",
        "distilbert.transformer.layer.*.ffn.lin2",
    ],
    "albert/albert-base-v2": [
        "albert.encoder.albert_layer_groups.*.albert_layers.*.attention.query",
        "albert.encoder.albert_layer_groups.*.albert_layers.*.attention.key",
        "albert.encoder.albert_layer_groups.*.albert_layers.*.attention.value",
        "albert.encoder.albert_layer_groups.*.albert_layers.*.attention.dense",
        "albert.encoder.albert_layer_groups.*.albert_layers.*.ffn",
        "albert.encoder.albert_layer_groups.*.albert_layers.*.ffn_output",
    ],
    "google-bert/bert-base-uncased": [
        "bert.encoder.layer.*.attention.self.query",
        "bert.encoder.layer.*.attention.self.key",
        "bert.encoder.layer.*.attention.self.value",
        "bert.encoder.layer.*.attention.output.dense",
        "bert.encoder.layer.*.intermediate.dense",
        "bert.encoder.layer.*.output.dense",
    ],
    "answerdotai/ModernBERT-base": [
        "model.layers.*.attn.Wqkv",
        "model.layers.*.attn.Wo",
        "model.layers.*.mlp.Wi",
        "model.layers.*.mlp.Wo",
    ],
    "gpt2": [
        "transformer.h.*.attn.c_attn",
        "transformer.h.*.attn.c_proj",
        "transformer.h.*.mlp.c_fc",
        "transformer.h.*.mlp.c_proj",
    ],
    "google/gemma-2-2b-it": [
        "model.layers.*.self_attn.q_proj",
        "model.layers.*.self_attn.k_proj",
        "model.layers.*.self_attn.v_proj",
        "model.layers.*.self_attn.o_proj",
        "model.layers.*.mlp.gate_proj",
        "model.layers.*.mlp.up_proj",
        "model.layers.*.mlp.down_proj",
    ],
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0": [
        "model.layers.*.self_attn.q_proj",
        "model.layers.*.self_attn.k_proj",
        "model.layers.*.self_attn.v_proj",
        "model.layers.*.self_attn.o_proj",
        "model.layers.*.mlp.gate_proj",
        "model.layers.*.mlp.up_proj",
        "model.layers.*.mlp.down_proj",
    ],
    "google/vit-base-patch16-224": [
        "vit.encoder.layer.*.attention.attention.query",
        "vit.encoder.layer.*.attention.attention.key",
        "vit.encoder.layer.*.attention.attention.value",
        "vit.encoder.layer.*.attention.output.dense",
        "vit.encoder.layer.*.intermediate.dense",
        "vit.encoder.layer.*.output.dense",
    ],
    "google/vit-base-patch32-224": [
        "vit.encoder.layer.*.attention.attention.query",
        "vit.encoder.layer.*.attention.attention.key",
        "vit.encoder.layer.*.attention.attention.value",
        "vit.encoder.layer.*.attention.output.dense",
        "vit.encoder.layer.*.intermediate.dense",
        "vit.encoder.layer.*.output.dense",
    ],
    "facebook/deit-tiny-patch16-224": [
        "deit.encoder.layer.*.attention.attention.query",
        "deit.encoder.layer.*.attention.attention.key",
        "deit.encoder.layer.*.attention.attention.value",
        "deit.encoder.layer.*.attention.output.dense",
        "deit.encoder.layer.*.intermediate.dense",
        "deit.encoder.layer.*.output.dense",
        "vit.encoder.layer.*.attention.attention.query",
        "vit.encoder.layer.*.attention.attention.key",
        "vit.encoder.layer.*.attention.attention.value",
        "vit.encoder.layer.*.attention.output.dense",
        "vit.encoder.layer.*.intermediate.dense",
        "vit.encoder.layer.*.output.dense",
    ],
    # Whisper / wav2vec2: include both nested-prefix and root-relative patterns so hooks match
    # whether loaded via AutoModel or wrappers where submodule names differ.
    "openai/whisper-tiny": [
        "model.encoder.layers.*.self_attn.q_proj",
        "encoder.layers.*.self_attn.q_proj",
        "model.encoder.layers.*.self_attn.k_proj",
        "encoder.layers.*.self_attn.k_proj",
        "model.encoder.layers.*.self_attn.v_proj",
        "encoder.layers.*.self_attn.v_proj",
        "model.encoder.layers.*.self_attn.out_proj",
        "encoder.layers.*.self_attn.out_proj",
        "model.encoder.layers.*.fc1",
        "encoder.layers.*.fc1",
        "model.encoder.layers.*.fc2",
        "encoder.layers.*.fc2",
    ],
    "facebook/wav2vec2-base-960h": [
        "wav2vec2.encoder.layers.*.attention.q_proj",
        "encoder.layers.*.attention.q_proj",
        "wav2vec2.encoder.layers.*.attention.k_proj",
        "encoder.layers.*.attention.k_proj",
        "wav2vec2.encoder.layers.*.attention.v_proj",
        "encoder.layers.*.attention.v_proj",
        "wav2vec2.encoder.layers.*.attention.out_proj",
        "encoder.layers.*.attention.out_proj",
        "wav2vec2.encoder.layers.*.feed_forward.intermediate_dense",
        "encoder.layers.*.feed_forward.intermediate_dense",
        "wav2vec2.encoder.layers.*.feed_forward.output_dense",
        "encoder.layers.*.feed_forward.output_dense",
    ],
}


def resolve_target_layer_preset(model_id: str) -> list[str] | None:
    mid = (model_id or "").strip()
    if not mid:
        return None
    if mid in TARGET_LAYER_PRESETS:
        return list(TARGET_LAYER_PRESETS[mid])
    mid_low = mid.lower()
    for key, patterns in TARGET_LAYER_PRESETS.items():
        if key.lower() == mid_low:
            return list(patterns)
    return None


def format_preset_for_input(model_id: str) -> str:
    pats = resolve_target_layer_preset(model_id)
    if not pats:
        return ""
    return ", ".join(pats)
