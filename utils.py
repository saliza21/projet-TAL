"""
Fonctions utilitaires partagées entre tous les scripts du projet.
Seeding, métriques, chargement des données.
"""

import random
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score
)


def seed_everything(seed):
    """Fixe toutes les sources d'aléa pour la reproductibilité."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_classification_metrics(y_true, y_pred, y_proba=None, num_classes=4):
    """
    Calcule toutes les métriques demandées pour l'exercice 1.
    y_true: labels réels
    y_pred: labels prédits
    y_proba: probabilités (pour AUROC), shape (n, num_classes)
    """
    metrics = {}
    metrics["micro_acc"] = accuracy_score(y_true, y_pred)

    # macro accuracy = moyenne des accuracy par classe
    per_class_acc = []
    for cls in range(num_classes):
        mask = np.array(y_true) == cls
        if mask.sum() > 0:
            per_class_acc.append(accuracy_score(
                np.array(y_true)[mask], np.array(y_pred)[mask]
            ))
    metrics["macro_acc"] = np.mean(per_class_acc)

    metrics["micro_f1"] = f1_score(y_true, y_pred, average="micro")
    metrics["macro_f1"] = f1_score(y_true, y_pred, average="macro")

    if y_proba is not None:
        try:
            metrics["auroc"] = roc_auc_score(
                y_true, y_proba, multi_class="ovr", average="macro"
            )
        except ValueError:
            metrics["auroc"] = float("nan")
    else:
        metrics["auroc"] = float("nan")

    return metrics


def load_ag_news_subset(train_size=4000, test_size=1000, seed=42):
    """
    Charge AG News et sous-échantillonne avec un seed fixe.
    On utilise le même sous-ensemble pour tous les modèles.
    """
    from datasets import load_dataset
    ds = load_dataset("fancyzhx/ag_news")

    rng = np.random.RandomState(seed)
    train_idx = rng.choice(len(ds["train"]), size=train_size, replace=False)
    test_idx = rng.choice(len(ds["test"]), size=test_size, replace=False)

    train_texts = [ds["train"][int(i)]["text"] for i in train_idx]
    train_labels = [ds["train"][int(i)]["label"] for i in train_idx]
    test_texts = [ds["test"][int(i)]["text"] for i in test_idx]
    test_labels = [ds["test"][int(i)]["label"] for i in test_idx]

    return train_texts, train_labels, test_texts, test_labels


# ---------- Métriques pour l'exercice 2 (ROUGE, embedding sim) ----------

def compute_rouge_n(candidate, reference, n=1):
    """Calcule ROUGE-N (precision, recall, F1) entre deux strings."""
    def get_ngrams(tokens, n):
        return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

    cand_tokens = candidate.lower().split()
    ref_tokens = reference.lower().split()

    if len(cand_tokens) < n or len(ref_tokens) < n:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    cand_ngrams = get_ngrams(cand_tokens, n)
    ref_ngrams = get_ngrams(ref_tokens, n)

    # comptage avec plafond (modified precision style)
    ref_counts = {}
    for ng in ref_ngrams:
        ref_counts[ng] = ref_counts.get(ng, 0) + 1

    cand_counts = {}
    for ng in cand_ngrams:
        cand_counts[ng] = cand_counts.get(ng, 0) + 1

    matches = 0
    for ng, count in cand_counts.items():
        matches += min(count, ref_counts.get(ng, 0))

    precision = matches / len(cand_ngrams) if cand_ngrams else 0.0
    recall = matches / len(ref_ngrams) if ref_ngrams else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1}


def lcs_length(a, b):
    """Plus longue sous-séquence commune (programmation dynamique)."""
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i-1] == b[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]


def compute_rouge_l(candidate, reference):
    """Calcule ROUGE-L basé sur la LCS."""
    cand_tokens = candidate.lower().split()
    ref_tokens = reference.lower().split()

    if not cand_tokens or not ref_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    lcs = lcs_length(cand_tokens, ref_tokens)
    precision = lcs / len(cand_tokens) if cand_tokens else 0.0
    recall = lcs / len(ref_tokens) if ref_tokens else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1}


def print_metrics_table(results_dict):
    """Affiche un tableau récapitulatif des résultats."""
    print("\n" + "="*70)
    for name, vals in results_dict.items():
        print(f"  {name}: {vals:.4f}")
    print("="*70)
