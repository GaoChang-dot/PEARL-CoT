import re
import argparse
import torch
import nltk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BertTokenizer,
    BertForSequenceClassification
)
from sentence_transformers import SentenceTransformer, util
from peft import get_peft_model, LoraConfig, TaskType
from trl import GRPOConfig, GRPOTrainer
from datasets import load_dataset
from nltk.translate.meteor_score import meteor_score

nltk.download('punkt')
nltk.download('wordnet')

# --- Extraction utilities ---
def extract_emotion(text: str) -> str:
    match = re.search(r"emotion:(.*?)persona_info", text, re.DOTALL)
    return match.group(1).strip() if match else ""

def extract_persona(text: str) -> str:
    match = re.search(r"persona_info:(.*?)</think>", text, re.DOTALL)
    return match.group(1).strip() if match else ""

def extract_response(text: str) -> str:
    match = re.search(r"<response>(.*?)</response>", text, re.DOTALL)
    return match.group(1).strip() if match else ""

def extract_context(prompt: str) -> str:
    pattern = r"Here is a conversation between a seeker and a supporter.(.*?)Based on the conversation above, please follow the instructions below"
    match = re.search(pattern, prompt, re.DOTALL)
    return match.group(1).strip() if match else ""

# --- Label mapping for scorer ---
id2label = {0: -1, 1: 0, 2: 1}

def format_score_context(conversation: str) -> str:
    # Remove speaker tokens and keep last 8 turns separated by [SEP]
    context = conversation.replace("seeker:", "").replace("supporter:", "")
    turns = context.strip().split("\n")[-8:]
    return " [SEP] ".join(turns)

def score_response(context: str, response: str, model, tokenizer, device, max_length=512) -> int:
    model.eval()
    model.to(device)
    conversation = f"{context.strip()}\n{response.strip()}"
    formatted_context = format_score_context(conversation)
    encoding = tokenizer(
        formatted_context,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    with torch.no_grad():
        outputs = model(
            input_ids=encoding["input_ids"].to(device),
            attention_mask=encoding["attention_mask"].to(device)
        )
        logits = outputs.logits
        pred = torch.argmax(logits, dim=1).item()
    return id2label[pred]

# --- Reward functions ---
def emotion_reward(prompts, completions, **kwargs):
    gold = kwargs.get("gold_response")
    preds = [extract_emotion(c[0]['content']) for c in completions]
    labels = [extract_emotion(g) for g in gold]
    rewards = [1.0 if p == l else 0.0 for p, l in zip(preds, labels)]
    print("Emotion Reward:", rewards)
    return rewards

def persona_reward(prompts, completions, **kwargs):
    gold = kwargs.get("gold_response")
    preds = [extract_persona(c[0]['content']) for c in completions]
    labels = [extract_persona(g) for g in gold]
    pred_embed = sim_model.encode(preds, convert_to_tensor=True)
    label_embed = sim_model.encode(labels, convert_to_tensor=True)
    sim_scores = util.pytorch_cos_sim(pred_embed, label_embed).diag().tolist()
    rewards = [1.0 if s > 0.5 else 0.0 for s in sim_scores]
    print("Persona Reward:", rewards)
    return rewards

def meteor_relevance_reward(prompts, completions, **kwargs):
    gold = kwargs.get("gold_response")
    hyps = [c[0]['content'] for c in completions]
    preds = [extract_response(h) for h in hyps]
    refs = [extract_response(gold[0])]
    tokenized_ref = nltk.word_tokenize(refs[0])
    rewards = []
    for hyp in preds:
        tokenized_hyp = nltk.word_tokenize(hyp)
        score = meteor_score([tokenized_ref], tokenized_hyp)
        rewards.append(round(score, 1))
    print("METEOR Reward:", rewards)
    return rewards

def score_reward(prompts, completions, **kwargs):
    gold = kwargs.get("gold_response")
    context = extract_context(prompts[0][-1]["content"])
    responses = [extract_response(c[0]['content']) for c in completions]
    rewards = [float(score_response(context, r, score_model, score_tokenizer, device)) for r in responses]
    print("Score Reward:", rewards)
    return rewards

def main(args):
    base_model_type = args.base_model_type
    model_path = f"../sft/merged_model/{base_model_type}/"
    output_dir = f"{base_model_type}/"

    global device, sim_model, score_model, score_tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading SentenceTransformer model for persona reward...")
    sim_model = SentenceTransformer('sentence-transformers/multi-qa-distilbert-cos-v1')

    print("Loading score model and tokenizer...")
    score_tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")
    score_model = BertForSequenceClassification.from_pretrained("google-bert/bert-base-uncased", num_labels=3, problem_type="single_label_classification")
    score_model.load_state_dict(torch.load("../scorer/best_model.pth"))

    print("Loading base tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="flash_attention_2"
    )

    print("Applying LoRA...")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        task_type=TaskType.CAUSAL_LM
    )
    model.enable_input_require_grads()
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.config.use_cache = False

    print("Loading datasets...")
    train_dataset = load_dataset("json", data_files={"train": "../data/grpo_dataset/grpo_train.json"})["train"]
    eval_dataset = load_dataset("json", data_files={"validation": "../data/grpo_dataset/grpo_dev.json"})["validation"]

    print("Configuring GRPO training...")
    training_args = GRPOConfig(
        output_dir=output_dir,
        learning_rate=1e-4,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type='cosine',
        logging_steps=1,
        bf16=True,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        num_generations=4,
        max_prompt_length=3072,
        max_completion_length=150,
        num_train_epochs=3,
        save_steps=50,
        max_grad_norm=0.1,
        report_to="none",
        deepspeed="ds_config.json"
    )

    print("Initializing GRPOTrainer...")
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            emotion_reward,
            persona_reward,
            score_reward,
            meteor_relevance_reward,
        ],
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )

    print("Starting training...")
    trainer.train(resume_from_checkpoint=False)

    print("Saving final LoRA weights...")
    save_path = output_dir + "final"
    trainer.model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a language model with GRPO multi-reward")
    parser.add_argument("--base_model_type", type=str, required=True, choices=["llama", "qwen"])
    args = parser.parse_args()
    main(args)
