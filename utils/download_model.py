"""
This script downloads a model and tokenizer from Hugging Face, simulating a local setup.
It should only be run once to download the model files.
"""
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MODEL_PATH = "./models/TinyLlama-1.1B-Chat-v1.0"

# First time - this downloads to cache
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Save it locally to simulate getting files from teammate
model.save_pretrained(MODEL_PATH)
tokenizer.save_pretrained(MODEL_PATH)