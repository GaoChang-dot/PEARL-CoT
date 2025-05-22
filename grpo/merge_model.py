import os
import argparse

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import PeftModel


def set_device():
    """Set CUDA device for distributed training."""
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    print(f"Using GPU {torch.cuda.current_device()}")


def get_model_paths(base_model_type: str) -> tuple[str, str, str]:
    """Return the model paths based on base model type."""
    base_model_path = f"../sft/merged_model/{base_model_type}"
    lora_path = f"{base_model_type}/final"
    save_path = f"merged_model/{base_model_type}/"
    return base_model_path, lora_path, save_path


def merge_and_save_model(base_model_path: str, lora_path: str, save_path: str):
    """Merge the base model with LoRA adapter and save the merged model."""
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, lora_path)

    print("Merging LoRA weights into base model...")
    merged_model = model.merge_and_unload()

    print(f"Saving merged model to '{save_path}'...")
    merged_model.save_pretrained(save_path)

    print("Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(lora_path)
    tokenizer.save_pretrained(save_path)

    print("Model merge and save completed successfully!")


def run_merge(args):
    """Main function to merge and save the model."""
    set_device()
    base_model_path, lora_path, save_path = get_model_paths(args.base_model_type)
    merge_and_save_model(base_model_path, lora_path, save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge a base model with its corresponding LoRA adapter and save the merged model.")
    parser.add_argument("--base_model_type", type=str, required=True, choices=["llama", "qwen"])
    args = parser.parse_args()

    run_merge(args)