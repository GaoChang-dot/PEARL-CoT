import json
import argparse
from tqdm import tqdm, trange

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from sklearn.metrics import accuracy_score
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    get_scheduler
)


class ConversationDataset(Dataset):
    """Dataset for helpfulness scoring task."""

    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        # Remap labels: -1 → 0, 0 → 1, 1 → 2
        self.label_map = {-1: 0, 0: 1, 1: 2}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item["input"]
        label = self.label_map[item["output"]]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def validate(model, device, val_loader):
    """Evaluate the model on validation data."""
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for _, batch in tqdm(enumerate(val_loader), total=len(val_loader), desc="Validating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    print(f"\n[Validation] Accuracy: {accuracy:.4f}")
    return accuracy


def train(args):
    # Load data
    train_data = load_json(args.train_file)
    val_data = load_json(args.val_file)

    # Load tokenizer and model
    tokenizer = BertTokenizer.from_pretrained(args.model_name)
    model = BertForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=3,
        problem_type="single_label_classification"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Prepare DataLoaders
    train_dataset = ConversationDataset(train_data, tokenizer, args.max_length)
    val_dataset = ConversationDataset(val_data, tokenizer, args.max_length)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=len(train_loader) * args.epochs
    )

    best_accuracy = 0
    step_count = 0

    for epoch in trange(args.epochs, desc="Epochs"):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        model.train()
        total_loss = 0.0

        for step, batch in tqdm(enumerate(train_loader), total=len(train_loader), desc="Training"):
            step_count += 1
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()

            total_loss += loss.item()

            if (step + 1) % args.log_interval == 0:
                print(f"Step {step + 1} - Loss: {loss.item():.4f}")

            if step_count % args.val_interval == 0:
                print(f"\n[Step {step_count}] Running validation...")
                accuracy = validate(model, device, val_loader)

                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    torch.save(model.state_dict(), args.save_path)
                    print(f"Model saved with improved accuracy: {accuracy:.4f}")

        avg_loss = total_loss / len(train_loader)
        print(f"[Epoch {epoch + 1}] Average Training Loss: {avg_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train helpfulness scorer")
    parser.add_argument("--train_file", type=str, default="../data/score_dataset/score_train.json")
    parser.add_argument("--val_file", type=str, default="../data/score_dataset/score_dev.json")
    parser.add_argument("--model_name", type=str, default="google-bert/bert-base-uncased")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--log_interval", type=int, default=2)
    parser.add_argument("--val_interval", type=int, default=20)
    parser.add_argument("--save_path", type=str, default="best_model.pth")
    args = parser.parse_args()

    train(args)
