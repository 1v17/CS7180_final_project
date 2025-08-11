# -*- coding: utf-8 -*-
"""
Klyme Personalized AI Full Parameter Fine-tuning Training Script
================================================================
This script is specifically designed for full parameter fine-tuning of the Llama-3-8B-Instruct model
on NVIDIA 3090/4090 (24GB VRAM), aiming to maximize hardware utilization.

Author: Gemini (Customized for your Klyme project)
Date: August 3, 2025
Fix: Adapted for TRL 0.20.0 API changes
"""

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from trl import SFTTrainer, SFTConfig  # Import SFTConfig
import os

# --- 1. Configuration Parameters ---
# -----------------------------------------------------------------------------
# Model ID, ensure you have access to meta-llama/Meta-Llama-3-8B-Instruct
MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
#MODEL_ID = "Qwen/Qwen3-8B"

# Dataset path, relative to this script location
DATASET_PATH = "./data/Klyme_dataset.jsonl"

# Output directory for saving the trained model
OUTPUT_DIR = "./results/Klyme_model_v1"


# --- Main Training Function ---
# -----------------------------------------------------------------------------
def main():
    """Main training workflow function"""
    
    print("===================================================")
    print("      Klyme Personalized AI Fine-tuning Project - Training Started      ")
    print("===================================================")
    
    # --- 2. Load Dataset ---
    print(f"\n[Step 2/7] Loading dataset from '{DATASET_PATH}'...")
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Error: Dataset file not found! Please ensure '{DATASET_PATH}' exists.")
    
    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
    print(f"✅ Dataset loaded successfully, {len(dataset)} samples in total.")

    # --- 3. Load Tokenizer ---
    print(f"\n[Step 3/7] Loading Tokenizer from '{MODEL_ID}'...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    # For Llama-3, pad_token is the same as eos_token, this is a standard setting
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("✅ Tokenizer loaded successfully.")

    # --- 4. Load Model ---
    # Using bfloat16 mixed precision, which is key for full parameter fine-tuning on 30/40 series GPUs
    print(f"\n[Step 4/7] Loading model from '{MODEL_ID}' (using bfloat16)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto", # Automatically assign model to GPU
        # attn_implementation="flash_attention_2" # Key acceleration: Enable Flash Attention 2 for 4090
    )
    # Note: Using FlashAttention requires additional installation: `pip install flash-attn`
    # For first training, comment out this line to ensure successful execution, uncomment later for maximum speed

    # Disable cache during training to save some VRAM
    model.config.use_cache = False
    print("✅ Model loaded successfully and assigned to GPU.")

    # --- 5. Configure Training Parameters ---
    # These parameters are carefully tuned to squeeze maximum performance from 24GB VRAM
    print("\n[Step 5/7] Configuring training parameters...")
    
    # New API requires SFT-related parameters to be placed in SFTConfig
    training_arguments = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=5,                # Train for 2 epochs, usually sufficient for high-quality small datasets
        per_device_train_batch_size=2,     # Batch size per GPU. Try reducing to 1 if OOM occurs
        gradient_accumulation_steps=8,     # Gradient accumulation, effective batch size is 2*8=16
        gradient_checkpointing=True,       # Enable gradient checkpointing, trade compute time for VRAM
        learning_rate=2e-5,                # Learning rate
        optim="paged_adamw_8bit",          # Use 8-bit paged optimizer to save VRAM
        logging_dir=f"{OUTPUT_DIR}/logs",  # Log directory
        logging_strategy="steps",
        logging_steps=1,                   # Output logs every 1 step
        bf16=True,                         # **Key**: Enable bfloat16 mixed precision training
        tf32=True,                         # **Key**: Enable TF32 on Ampere and newer GPUs for acceleration
        save_strategy="epoch",             # Save model at the end of each epoch
        save_total_limit=5,                # Keep maximum 2 checkpoints
        report_to="tensorboard",           # Report training metrics to TensorBoard
        
        # SFT-specific parameters - these parameters need to be in SFTConfig in new version
        dataset_text_field="messages",     # **Key**: Tell Trainer that core field in dataset is 'messages'
        max_length=2048,                   # Maximum sequence length (note: renamed from max_seq_length to max_length)
        packing=True,                      # **Key**: Pack multiple short sequences, greatly improves training efficiency

        # Advanced settings supported by H100
        dataloader_pin_memory=True,
        fp16_full_eval=False,              # Use bf16 for evaluation
        #load_best_model_at_end=True,
        #metric_for_best_model="eval_loss",
        #greater_is_better=False,
    )
    print("✅ Training parameters configured.")

    # --- 6. Initialize SFTTrainer ---
    # SFTTrainer is the star of trl library, greatly simplifying the fine-tuning process
    print("\n[Step 6/7] Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        args=training_arguments,
        train_dataset=dataset,
        processing_class=tokenizer,        # **Important change**: tokenizer parameter changed to processing_class
    )
    print("✅ Trainer initialized successfully.")

    # --- 7. Start Training ---
    print("\n[Step 7/7] Starting training! Please sit back and relax, this may take some time...")
    print("You can monitor GPU status with `nvidia-smi -l 1` command.")
    print(f"You can also run `tensorboard --logdir={OUTPUT_DIR}/logs` in a new terminal to monitor training curves.")

    trainer.train()

    print("\n🎉🎉🎉 Training completed! 🎉🎉🎉")

    # --- Save Final Model ---
    print(f"\nSaving final model to '{OUTPUT_DIR}'...")
    trainer.save_model()
    print("✅ Model saved successfully! Your AI companion Klyme is ready.")
    print("===================================================")


if __name__ == "__main__":
    main()