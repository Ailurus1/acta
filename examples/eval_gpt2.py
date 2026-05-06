import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from acta import AutoAnalyzer

model = AutoModelForCausalLM.from_pretrained(
    "openai-community/gpt2",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")

model = AutoAnalyzer(
    model,
    tokenizer=tokenizer,
    dump_stats_path="./activations_analysis.json",
    target_layers=["*Block"],
    draw_charts=True,
    verbose=True
)

inputs = tokenizer("Summer is warm. Winter is cold.", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=100)