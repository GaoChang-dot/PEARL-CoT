import os
import json
import random
import argparse

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import get_peft_model, LoraConfig, TaskType
from datasets import Dataset as HFDataset
from typing import Dict, Any


class SFTDataset(Dataset):
    """Dataset for supervised fine-tuning with causal language models."""
    
    def __init__(self, data, tokenizer, base_model_type: str):
        self.data = data
        self.tokenizer = tokenizer
        self.base_model_type = base_model_type
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        example = self.data[idx]
        return self.format_sample(example)
    
    def format_sample(self, example: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Tokenize a single example depending on the base model type."""
        if self.base_model_type == "llama":
            prompt = (
                "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
                f"{example['prompt'][0]['content']}\n\n"
                "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            )
            response = self.tokenizer(
                f"{example['gold_response']}<|eot_id|>", 
                add_special_tokens=False, 
                max_length=3072
            )
        else:  # Qwen
            prompt = (
                "<|im_start|>user\n"
                f"{example['prompt'][0]['content']}"
                "<|im_end|>\n<|im_start|>assistant\n"
            )
            response = self.tokenizer(
                f"{example['gold_response']}<|im_end|>", 
                add_special_tokens=False, 
                max_length=3072
            )

        instruction = self.tokenizer(prompt, add_special_tokens=False, max_length=3072)
        input_ids = instruction["input_ids"] + response["input_ids"]
        attention_mask = instruction["attention_mask"] + response["attention_mask"]
        labels = [-100] * len(instruction["input_ids"]) + response["input_ids"]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def load_json(file_path: str):
    """Load data from a JSON file."""
    with open(file_path, "r") as f:
        return json.load(f)


def set_device():
    """Set CUDA device for distributed training."""
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    print(f"Using GPU {torch.cuda.current_device()}")


def get_model_and_tokenizer(model_name: str):
    """Load base model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda"
    )

    # Set pad token if not defined
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    print(f"Pad token ID: {tokenizer.pad_token_id}")

    return model, tokenizer


def apply_lora(model):
    """Wrap model with LoRA adapters."""
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        task_type=TaskType.CAUSAL_LM,
    )
    model.enable_input_require_grads()
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.config.use_cache = False
    return model


def get_training_args(args):
    """Create HuggingFace TrainingArguments."""
    return TrainingArguments(
        output_dir=f"sft_lora/{args.base_model_type}/",
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.scheduler_type,
        num_train_epochs=args.epochs,
        save_steps=args.save_steps,
        save_total_limit=args.save_limit,
        logging_steps=args.log_interval,
        logging_dir=f"logs/{args.base_model_type}/",
        report_to="none",
        deepspeed=args.deepspeed_config,
        gradient_checkpointing=args.gradient_checkpointing,
        bf16=args.bf16,
    )


def train(args):
    """Main training function."""
    set_device()

    # Load model and tokenizer
    model_name = "meta-llama/Llama-3.2-1B-Instruct" if args.base_model_type == "llama" else "Qwen/Qwen2.5-1.5B-Instruct"
    model, tokenizer = get_model_and_tokenizer(model_name)
    model = apply_lora(model)

    # Load and prepare dataset
    data = load_json(args.train_file)
    random.shuffle(data)
    print(f"Total training examples: {len(data)}")
    dataset = HFDataset.from_list(data)
    tokenized_dataset = dataset.map(
        lambda x: SFTDataset.format_sample(x, tokenizer, args.base_model_type)
    )

    # Set up training
    training_args = get_training_args(args)
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, 
        padding=True, 
        pad_to_multiple_of=8
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    # Start training
    print(f"\nStarting training for {args.epochs} epochs...")
    trainer.train()

    # Save final model
    save_path = f"sft_lora/{args.base_model_type}/final"
    trainer.model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"\nTraining complete. Model saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LoRA SFT Training")
    parser.add_argument("--base_model_type", type=str, required=True, choices=["llama", "qwen"])
    parser.add_argument("--train_file", type=str, default="../data/sft_dataset/sft_train.json")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--scheduler_type", type=str, default="cosine")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--save_steps", type=int, default=20)
    parser.add_argument("--save_limit", type=int, default=5)
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--deepspeed_config", type=str, default="ds_config.json")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    args = parser.parse_args()

    train(args)