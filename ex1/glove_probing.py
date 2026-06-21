"""
Exercice 1 - Embeddings statiques : GloVe moyenné + Logistic Regression
On fait la moyenne des vecteurs GloVe de chaque mot du doc.
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from utils import seed_everything, load_ag_news_subset, compute_classification_metrics
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

SEEDS = [42, 123, 456]
TRAIN_SIZE = 4000
TEST_SIZE = 1000
GLOVE_DIM = 100


def load_glove(dim=100):
    """Charge les embeddings GloVe depuis torchtext."""
    from torchtext.vocab import GloVe
    glove = GloVe(name="6B", dim=dim)
    return glove


def text_to_glove_vector(text, glove, dim=100):
    """Moyenne des vecteurs GloVe pour les mots du texte."""
    tokens = text.lower().split()
    vectors = []
    for token in tokens:
        if token in glove.stoi:
            vectors.append(glove[token].numpy())
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(dim)


def main():
    print("Chargement de GloVe...")
    glove = load_glove(GLOVE_DIM)

    all_results = []

    for seed in SEEDS:
        seed_everything(seed)
        print(f"\n--- Seed {seed} ---")

        train_texts, train_labels, test_texts, test_labels = load_ag_news_subset(
            train_size=TRAIN_SIZE, test_size=TEST_SIZE, seed=42
        )

        # Vectorisation
        print("Vectorisation des textes...")
        X_train = np.array([text_to_glove_vector(t, glove, GLOVE_DIM) for t in train_texts])
        X_test = np.array([text_to_glove_vector(t, glove, GLOVE_DIM) for t in test_texts])

        # Classification
        clf = LogisticRegression(max_iter=1000, random_state=seed)
        clf.fit(X_train, train_labels)

        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)

        metrics = compute_classification_metrics(test_labels, y_pred, y_proba)
        all_results.append(metrics)

        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

    # Résultats
    print("\n" + "="*60)
    print("GloVe + Probing — Résultats (mean ± std)")
    print("="*60)
    for key in all_results[0]:
        vals = [r[key] for r in all_results]
        print(f"  {key}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")


if __name__ == "__main__":
    main()
