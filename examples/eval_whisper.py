from __future__ import annotations


import numpy as np
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from acta import AutoAnalyzer


def _preferred_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    model_id = "openai/whisper-tiny"
    processor = WhisperProcessor.from_pretrained(model_id)
    full_model = WhisperForConditionalGeneration.from_pretrained(model_id)
    encoder = full_model.get_encoder()

    model = AutoAnalyzer(
        encoder,
        dump_stats_path="./whisper_encoder_analysis",
        target_layers="*out_proj*",
        draw_charts=True,
        verbose=True,
        tokenizer=None,
        asr_chunk_labels=True,
        vit_reg_patch_labels=False,
        chart_3d_max_tokens=24,
    )
    model.eval()
    device = _preferred_device()
    model.to(device)

    sr = 16000
    duration_s = 2.0
    t = np.linspace(0.0, duration_s, int(sr * duration_s), dtype=np.float32)
    audio = 0.12 * (
        np.sin(2.0 * np.pi * 220.0 * t) + 0.5 * np.sin(2.0 * np.pi * 440.0 * t)
    ).astype(np.float32)

    inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
    feats = inputs.input_features.to(device)

    with torch.no_grad():
        _ = model(feats)


if __name__ == "__main__":
    main()
