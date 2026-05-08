from __future__ import annotations

import io
import os
import warnings

import requests
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

from acta import AutoAnalyzer

IMAGE_URL = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSts3LZY5r_gDJHykCjup1i4ckZrX7Ed7TT2A&s"


def _preferred_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _quiet_hf() -> None:
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    warnings.filterwarnings("ignore")
    try:
        from transformers.utils import logging as tr_logging

        tr_logging.set_verbosity_error()
    except Exception:
        pass
    try:
        from huggingface_hub.utils import logging as hf_logging

        hf_logging.set_verbosity_error()
    except Exception:
        pass


def load_image_from_url(url: str, timeout: float = 30.0) -> Image.Image:
    headers = {"User-Agent": "Mozilla/5.0 (compatible; acta-eval/1.0)"}
    response = requests.get(url, timeout=timeout, headers=headers)
    response.raise_for_status()
    return Image.open(io.BytesIO(response.content)).convert("RGB")


def main() -> None:
    _quiet_hf()

    model_name = "facebook/deit-tiny-patch16-224"
    processor = AutoImageProcessor.from_pretrained(model_name)
    base_model = AutoModelForImageClassification.from_pretrained(model_name)

    model = AutoAnalyzer(
        base_model,
        dump_stats_path="./vit_activations_analysis",
        target_layers="*layer.*.layernorm_before",
        draw_charts=True,
        verbose=True,
        tokenizer=None,
        vit_reg_patch_labels=True,
        asr_chunk_labels=False,
    )
    model.eval()
    device = _preferred_device()
    model.to(device)

    image = load_image_from_url(IMAGE_URL)
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        _ = model(**inputs)


if __name__ == "__main__":
    main()
