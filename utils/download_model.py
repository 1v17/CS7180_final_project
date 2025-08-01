"""
This script downloads a model and tokenizer from Hugging Face, simulating a local setup.
It should only be run once to download the model files.
"""
from transformers import AutoModelForCausalLM, AutoTokenizer

# Using an open model that doesn't require authentication
MODEL_NAME = "microsoft/DialoGPT-medium"
MODEL_PATH = "./models/test_local_model"

# First time - this downloads to cache
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Save it locally to simulate getting files from teammate
model.save_pretrained(MODEL_PATH)
tokenizer.save_pretrained(MODEL_PATH)