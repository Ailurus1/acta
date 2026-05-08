import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from acta import AutoAnalyzer


def _preferred_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    device = _preferred_device()
    model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2").to(device)
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")

    model = AutoAnalyzer(
        model,
        tokenizer=tokenizer,
        target_layers=["*Block"],
        draw_charts=True,
        verbose=True,
    )

    prompts = [
        "Hey there!" "Once upon a time in a land far, far away...",
        "Never gonna give you up",
        "Never gonna let you down",
    ]
    inputs = [tokenizer(prompt, return_tensors="pt").to(device) for prompt in prompts]
    for input in inputs:
        model.generate(**input, max_new_tokens=100)


if __name__ == "__main__":
    main()
