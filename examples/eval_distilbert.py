from __future__ import annotations

import torch
from transformers import AutoModel, AutoTokenizer

from acta import AutoAnalyzer


def _preferred_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = _preferred_device()
    base_model = AutoModel.from_pretrained(model_name).to(device)

    model = AutoAnalyzer(
        base_model,
        dump_stats_path="./distilbert_activations_analysis",
        target_layers="*",
        draw_charts=True,
        verbose=True,
        tokenizer=tokenizer,
    )
    model.eval()

    text = "DistilBERT is useful for many NLP tasks."
    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        _ = model(**inputs)


if __name__ == "__main__":
    main()
