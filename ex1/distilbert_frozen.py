"""
Exercice 1 - DistilBERT gelé + linear probe
L'encodeur est frozen, on entraîne seulement la couche linéaire.
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from utils import seed_everything, load_ag_news_subset, compute_classification_metrics, get_device
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertTokenizer, DistilBertModel

SEEDS = [42, 123, 456]
TRAIN_SIZE = 4000
TEST_SIZE = 1000
MAX_LEN = 128
NUM_CLASSES = 4
BATCH_SIZE = 32
EPOCHS = 5
LR = 1e-3


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx], truncation=True, padding="max_length",
            max_length=self.max_len, return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long)
        }


class FrozenDistilBertClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.encoder = DistilBertModel.from_pretrained("distilbert-base-uncased")
        # on gèle tous les paramètres de l'encodeur
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # vecteur du premier token [CLS]
        cls_output = outputs.last_hidden_state[:, 0, :]
        return self.classifier(cls_output)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            logits = model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch["label"].numpy())
            all_probs.extend(probs.cpu().numpy())
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def main():
    device = get_device()
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    all_results = []

    for seed in SEEDS:
        seed_everything(seed)
        print(f"\n--- Seed {seed} ---")

        train_texts, train_labels, test_texts, test_labels = load_ag_news_subset(
            train_size=TRAIN_SIZE, test_size=TEST_SIZE, seed=42
        )

        train_ds = TextDataset(train_texts, train_labels, tokenizer, MAX_LEN)
        test_ds = TextDataset(test_texts, test_labels, tokenizer, MAX_LEN)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

        model = FrozenDistilBertClassifier(NUM_CLASSES).to(device)
        # on n'optimise que le classifieur
        optimizer = torch.optim.Adam(model.classifier.parameters(), lr=LR)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(EPOCHS):
            loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            print(f"  Epoch {epoch+1}/{EPOCHS} - Loss: {loss:.4f}")

        y_true, y_pred, y_proba = evaluate(model, test_loader, device)
        metrics = compute_classification_metrics(y_true, y_pred, y_proba)
        all_results.append(metrics)

        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

    print("\n" + "="*60)
    print("DistilBERT frozen — Résultats (mean ± std)")
    print("="*60)
    for key in all_results[0]:
        vals = [r[key] for r in all_results]
        print(f"  {key}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")


if __name__ == "__main__":
    main()
