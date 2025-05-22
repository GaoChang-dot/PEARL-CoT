import json
import argparse
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizer,
    BertForSequenceClassification
)
from sklearn.metrics import accuracy_score, classification_report


class ConversationDataset(Dataset):
    """Dataset for conversation scoring."""

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
    """Load data from a JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def inference(model, device, test_loader):
    """Run model inference and print evaluation metrics."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader, total=len(test_loader), desc="Testing"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    accuracy = accuracy_score(all_labels, all_preds)
    print(f"\n[Test] Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, digits=4))


def main(args):
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer and model
    tokenizer = BertTokenizer.from_pretrained(args.model_name)
    model = BertForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=3,
        problem_type="single_label_classification"
    )
    model.load_state_dict(torch.load(args.weight_file, map_location=device))

    # Load test data and create DataLoader
    test_data = load_json(args.test_file)
    test_dataset = ConversationDataset(test_data, tokenizer, args.max_length)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # Run inference
    inference(model, device, test_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test helpfulness scorer")
    parser.add_argument("--test_file", type=str, default="../data/score_dataset/score_test.json")
    parser.add_argument("--model_name", type=str, default="google-bert/bert-base-uncased")
    parser.add_argument("--weight_file", type=str, default="best_model.pth")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=512)
    args = parser.parse_args()

    main(args)