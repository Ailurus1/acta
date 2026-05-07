import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from acta import AutoAnalyzer

def main() -> None:
    model = AutoModelForCausalLM.from_pretrained(
        "openai-community/gpt2",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")

    model = AutoAnalyzer(
        model,
        tokenizer=tokenizer,
        target_layers=["*Block"],
        draw_charts=True,
        verbose=True,
    )

    prompts = [
        "Hey there!"
        "Once upon a time in a land far, far away...",
        "Never gonna give you up",
        "Never gonna let you down"
    ]
    inputs = [tokenizer(prompt, return_tensors="pt").to(model.device) for prompt in prompts]
    outputs = [model.generate(**input, max_new_tokens=100) for input in inputs]

if __name__ == "__main__":
    main()