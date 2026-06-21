"""
Exercice 1 - BiLSTM avec embeddings GloVe
Tokenization maison, padding, BiLSTM + couche linéaire.
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from utils import seed_everything, load_ag_news_subset, compute_classification_metrics, get_device
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter

SEEDS = [42, 123, 456]
TRAIN_SIZE = 4000
TEST_SIZE = 1000
MAX_LEN = 128
EMBED_DIM = 100
HIDDEN_DIM = 64
NUM_CLASSES = 4
BATCH_SIZE = 64
EPOCHS = 8
LR = 1e-3


def build_vocab(texts, max_vocab=15000):
    """Construit un vocabulaire à partir des textes."""
    counter = Counter()
    for text in texts:
        counter.update(text.lower().split())
    vocab = {"<pad>": 0, "<unk>": 1}
    for word, _ in counter.most_common(max_vocab):
        vocab[word] = len(vocab)
    return vocab


def texts_to_indices(texts, vocab, max_len):
    """Convertit les textes en indices + padding."""
    result = []
    for text in texts:
        tokens = text.lower().split()[:max_len]
        indices = [vocab.get(t, vocab["<unk>"]) for t in tokens]
        # padding
        indices += [vocab["<pad>"]] * (max_len - len(indices))
        result.append(indices)
    return np.array(result)


def load_glove_embeddings(vocab, dim=100):
    """Charge les embeddings GloVe pour notre vocabulaire."""
    from torchtext.vocab import GloVe
    glove = GloVe(name="6B", dim=dim)

    matrix = np.random.normal(0, 0.1, (len(vocab), dim))
    matrix[0] = 0  # padding
    found = 0
    for word, idx in vocab.items():
        if word in glove.stoi:
            matrix[idx] = glove[word].numpy()
            found += 1
    print(f"  GloVe: {found}/{len(vocab)} mots trouvés")
    return torch.FloatTensor(matrix)


class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, pretrained_emb=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if pretrained_emb is not None:
            self.embedding.weight.data.copy_(pretrained_emb)

        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        emb = self.embedding(x)
        _, (hidden, _) = self.lstm(emb)
        # hidden: (2, batch, hidden_dim) -> concat forward et backward
        hidden = torch.cat((hidden[0], hidden[1]), dim=1)
        out = self.dropout(hidden)
        return self.fc(out)


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
    print(f"Device: {device}")

    all_results = []

    for seed in SEEDS:
        seed_everything(seed)
        print(f"\n--- Seed {seed} ---")

        train_texts, train_labels, test_texts, test_labels = load_ag_news_subset(
            train_size=TRAIN_SIZE, test_size=TEST_SIZE, seed=42
        )

        # Vocabulaire et embeddings
        vocab = build_vocab(train_texts)
        X_train = texts_to_indices(train_texts, vocab, MAX_LEN)
        X_test = texts_to_indices(test_texts, vocab, MAX_LEN)

        glove_emb = load_glove_embeddings(vocab, EMBED_DIM)

        # Datasets PyTorch
        train_ds = TensorDataset(
            torch.LongTensor(X_train), torch.LongTensor(train_labels)
        )
        test_ds = TensorDataset(
            torch.LongTensor(X_test), torch.LongTensor(test_labels)
        )
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

        # Modèle
        model = BiLSTMClassifier(
            len(vocab), EMBED_DIM, HIDDEN_DIM, NUM_CLASSES, glove_emb
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        criterion = nn.CrossEntropyLoss()

        # Entraînement
        for epoch in range(EPOCHS):
            loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            if (epoch + 1) % 2 == 0:
                print(f"  Epoch {epoch+1}/{EPOCHS} - Loss: {loss:.4f}")

        # Évaluation
        y_true, y_pred, y_proba = evaluate(model, test_loader, device)
        metrics = compute_classification_metrics(y_true, y_pred, y_proba)
        all_results.append(metrics)

        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

    # Résultats
    print("\n" + "="*60)
    print("BiLSTM + GloVe — Résultats (mean ± std)")
    print("="*60)
    for key in all_results[0]:
        vals = [r[key] for r in all_results]
        print(f"  {key}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")


if __name__ == "__main__":
    main()
