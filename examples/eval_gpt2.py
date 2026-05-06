import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from acta import AutoAnalyzer

model = AutoModelForCausalLM.from_pretrained(
    "openai-community/gpt2-xl",
    device_map="auto"
)
model = AutoAnalyzer(model, dump_stats_path="./activations_analysis.json", target_layers=["*Block"], draw_charts=True)

tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2-xl")
inputs = tokenizer("Once upon a time, there was a magical forest", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=100)