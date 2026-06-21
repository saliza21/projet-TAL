"""
Exercice 1 - Petit Transformer entraîné from scratch
2 couches, 2 têtes d'attention. Résultats modestes car peu de données.
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from utils import seed_everything, load_ag_news_subset, compute_classification_metrics, get_device
import numpy as np
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter

SEEDS = [42, 123, 456]
TRAIN_SIZE = 4000
TEST_SIZE = 1000
MAX_LEN = 128
EMBED_DIM = 64
NUM_HEADS = 2
NUM_LAYERS = 2
NUM_CLASSES = 4
BATCH_SIZE = 64
EPOCHS = 10
LR = 1e-3


def build_vocab(texts, max_vocab=15000):
    counter = Counter()
    for text in texts:
        counter.update(text.lower().split())
    vocab = {"<pad>": 0, "<unk>": 1}
    for word, _ in counter.most_common(max_vocab):
        vocab[word] = len(vocab)
    return vocab


def texts_to_indices(texts, vocab, max_len):
    result = []
    for text in texts:
        tokens = text.lower().split()[:max_len]
        indices = [vocab.get(t, vocab["<unk>"]) for t in tokens]
        indices += [vocab["<pad>"]] * (max_len - len(indices))
        result.append(indices)
    return np.array(result)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, num_classes, max_len=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_enc = PositionalEncoding(embed_dim, max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=embed_dim * 4, dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # masque de padding
        pad_mask = (x == 0)
        emb = self.embedding(x)
        emb = self.pos_enc(emb)
        out = self.transformer(emb, src_key_padding_mask=pad_mask)

        # mean pooling (on ignore le padding)
        mask = (~pad_mask).unsqueeze(-1).float()
        pooled = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        return self.fc(self.dropout(pooled))


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            logits = model(X)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.numpy())
            all_probs.extend(probs.cpu().numpy())
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def main():
    device = get_device()
    all_results = []

    for seed in SEEDS:
        seed_everything(seed)
        print(f"\n--- Seed {seed} ---")

        train_texts, train_labels, test_texts, test_labels = load_ag_news_subset(
            train_size=TRAIN_SIZE, test_size=TEST_SIZE, seed=42
        )

        vocab = build_vocab(train_texts)
        X_train = texts_to_indices(train_texts, vocab, MAX_LEN)
        X_test = texts_to_indices(test_texts, vocab, MAX_LEN)

        train_ds = TensorDataset(torch.LongTensor(X_train), torch.LongTensor(train_labels))
        test_ds = TensorDataset(torch.LongTensor(X_test), torch.LongTensor(test_labels))
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

        model = TransformerClassifier(
            len(vocab), EMBED_DIM, NUM_HEADS, NUM_LAYERS, NUM_CLASSES, MAX_LEN
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(EPOCHS):
            loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            if (epoch + 1) % 2 == 0:
                print(f"  Epoch {epoch+1}/{EPOCHS} - Loss: {loss:.4f}")

        y_true, y_pred, y_proba = evaluate(model, test_loader, device)
        metrics = compute_classification_metrics(y_true, y_pred, y_proba)
        all_results.append(metrics)

        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

    print("\n" + "="*60)
    print("Transformer from scratch — Résultats (mean ± std)")
    print("="*60)
    for key in all_results[0]:
        vals = [r[key] for r in all_results]
        print(f"  {key}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")


if __name__ == "__main__":
    main()
