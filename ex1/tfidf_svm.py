"""
Exercice 1 - Baseline lexicale : TF-IDF + SVM linéaire
Pas besoin de GPU, très rapide.
"""

import sys, os, json
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from utils import seed_everything, load_ag_news_subset, compute_classification_metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
import numpy as np

SEEDS = [42, 123, 456]
TRAIN_SIZE = 4000
TEST_SIZE = 1000

all_results = []

for seed in SEEDS:
    seed_everything(seed)
    print(f"\n--- Seed {seed} ---")

    # Chargement des données (même sous-ensemble pour tous les modèles)
    train_texts, train_labels, test_texts, test_labels = load_ag_news_subset(
        train_size=TRAIN_SIZE, test_size=TEST_SIZE, seed=42  # seed fixe pour les données
    )

    # TF-IDF
    vectorizer = TfidfVectorizer(max_features=10000, stop_words="english")
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)

    # SVM avec calibration pour avoir des probabilités (nécessaire pour AUROC)
    svm = LinearSVC(max_iter=5000, random_state=seed)
    clf = CalibratedClassifierCV(svm, cv=3)
    clf.fit(X_train, train_labels)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)

    metrics = compute_classification_metrics(test_labels, y_pred, y_proba)
    all_results.append(metrics)

    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

# Moyenne et écart-type
print("\n" + "="*60)
print("TF-IDF + SVM — Résultats (mean ± std)")
print("="*60)
for key in all_results[0]:
    vals = [r[key] for r in all_results]
    print(f"  {key}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")
