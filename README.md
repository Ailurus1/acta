# acta

**acta** is a Python library for automatic analysis of neural network activations in PyTorch models. Wrap a model with `AutoAnalyzer`, run `forward` or `generate`, and collect per-layer statistics, outlier flags, and charts without manually instrumenting tensors.

## What it can do

- **Hook-based collection** on selected layers (glob/regex patterns or all leaf modules) during `forward()` and `generate()`.
- **Per-channel + per-token statistics**: mean, variance, quantiles, and kurtosis with streaming aggregation.
- **Token-level outlier detection** using several rules:
  - LLM.int8()-style soft/hard thresholds (layer and hidden-dimension fractions)
  - Inter-quartile (IQR) outliers
- **Run report** in `stats.json`: Top-1/2/3/10 and percentile tails of |activation|, global kurtosis, and max–median ratio (MRR).
- **Artifacts per run** under `.acta_dump_results/<timestamp>/`:
  - `stats.json` — full statistics and outlier payload
  - `acta_results.csv` — detailed per-token flags and attribution columns
  - Charts (optional): operator-wise activation tops, kurtosis/MRR across blocks, 3D token–layer views, and more
- **Stdout summary table** at exit: `idx`, `token`, `token_id`, `layer`, `channel_dim`, `outliers detected`
- **CLI** (`acta`): easy experiment management from command line
- **Web UI** (Dash): load Hugging Face or local models for CLM, MLM, ViT, and ASR tasks

Works with transformers (GPT-2, BERT, Whisper, ViT, etc.) and arbitrary `nn.Module` checkpoints.

## Installation

### pip

```bash
pip install -e .
```

## Quickstart

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from acta import AutoAnalyzer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2").to(device)
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")

model = AutoAnalyzer(
    model,
    tokenizer=tokenizer,
    target_layers=["transformer.h.*.attn.c_proj", "transformer.h.*.mlp.c_fc"],
    draw_charts=True,          # write PNG charts on exit
    verbose=True,
)

inputs = tokenizer("Hello world", return_tensors="pt").to(device)
model.generate(**inputs, max_new_tokens=32)
# Prints a token table to stdout; writes .acta_dump_results/<timestamp>/
```

**CLI**

```bash
acta show                              # list past runs
acta plot .acta_dump_results/<run>/stats.json
acta ui                                # interactive Dash UI
```

See `examples/` for GPT-2, DistilBERT, ViT, and Whisper.

## Citation

If you use ACTA in academic work, please cite:

```bibtex
@software{acta2026,
  title  = {ACTA: Automatic Activation Analysis for PyTorch},
  author = {ACTA Contributors},
  year   = {2026},
  url    = {https://github.com/Ailurus1/acta}
}
```
