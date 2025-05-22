import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import argparse
from typing import Dict

def set_device():
    """Set the CUDA device based on environment variable LOCAL_RANK."""
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    print(f"Using GPU {torch.cuda.current_device()}")


def load_model_and_tokenizer(model_path: str):
    """Load model and tokenizer from the given path."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    print(f"Pad token ID set to: {tokenizer.pad_token_id}")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda"
    )
    model.eval()
    return model, tokenizer

def format_prompt(example: Dict[str, str], base_model_type: str) -> str:
    """
    Format the input text according to model-specific prompt style.
    """
    input_text = example['input']
    if base_model_type == "llama":
        return (
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{input_text}\n\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )
    else:  # qwen
        return f"<|im_start|>user\n{input_text}<|im_end|>\n<|im_start|>assistant\n"

def generate_response(model, tokenizer, input_text: str, base_model_type: str,
                      max_new_tokens=150, temperature=0.7, top_p=0.8, top_k=20, repetition_penalty=1.1) -> str:
    """
    Generate model response from input text.
    """
    prompt = format_prompt({"input": input_text}, base_model_type)
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    input_len = inputs.input_ids.shape[1]
    generated_ids = outputs[0][input_len:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)

    if base_model_type == "llama":
        return response.split("<|eot_id|>")[0].strip()
    else:
        return response.split("<|im_end|>")[0].strip()

def run_inference(args):
    """Main function to load model, generate responses, and save results."""
    set_device()
    model_path = f"merged_model/{args.base_model_type}/"
    val_data_path = "../data/grpo_dataset/grpo_test.json"

    model, tokenizer = load_model_and_tokenizer(model_path)

    with open(val_data_path, "r", encoding="utf-8") as f:
        val_data = json.load(f)

    results = []
    for i, example in enumerate(val_data):
        input_text = example["prompt"][0]["content"]
        gold_output = example["gold_response"]
        generated_output = generate_response(model, tokenizer, input_text, args.base_model_type)

        # Print first 5 examples for inspection
        if i < 5:
            print(f"\n=== Example {i + 1} ===")
            print(f"Input:\n{input_text}")
            print(f"\nGold Response:\n{gold_output}")
            print(f"\nGenerated Response:\n{generated_output}")
            print("\n" + "=" * 50 + "\n")

        results.append({
            "input": input_text,
            "gold_output": gold_output,
            "generated_output": generated_output
        })

    # Save inference results
    output_path = f"../data/grpo_dataset/{args.base_model_type}_inference.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Inference complete. Results saved to {output_path}")

if __name__ == "__main__":
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Inference with GRPO fine-tuned model.")
    parser.add_argument("--base_model_type", type=str, required=True, choices=["llama", "qwen"])
    args = parser.parse_args()
    run_inference(args)
