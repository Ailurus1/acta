from __future__ import annotations

import torch
from transformers import AutoModel, AutoTokenizer

from acta import AutoAnalyzer


def main() -> None:
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModel.from_pretrained(model_name)

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
    inputs = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        _ = model(**inputs)


if __name__ == "__main__":
    main()
