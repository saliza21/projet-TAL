# 🧠 Projet TAL — Classification de texte & Résumé automatique

> Projet de Traitement Automatique du Langage (TAL)  
> Master 1 Intelligence Artificielle & Sciences des Données en Santé  
> Université de Caen Normandie — Mai 2026

**Auteur :** Salma Benamar

---

## 📌 Description

Ce projet explore deux tâches fondamentales du TAL :

- **Exercice 1** — Classification de texte sur le dataset AG News (4 classes) en comparant 5 architectures allant de la baseline lexicale jusqu'aux transformers pré-entraînés
- **Exercice 2** — Résumé automatique de dialogues sur le dataset SAMSum en comparant approches zero-shot et fine-tuning de T5

---

## 📁 Structure du projet

```
Projet_Tal/
├── utils.py                        # Fonctions partagées (métriques, seeding, chargement)
├── rapport_TAL.pdf                 # Rapport complet du projet
├── ex1/                            # Classification de texte
│   ├── tfidf_svm.py                # Baseline : TF-IDF + SVM linéaire
│   ├── glove_probing.py            # GloVe moyenné + Régression logistique
│   ├── bilstm_glove.py             # BiLSTM avec embeddings GloVe
│   ├── transformer_scratch.py      # Transformer entraîné from scratch
│   ├── distilbert_frozen.py        # DistilBERT avec couches gelées
│   └── distilbert_finetuned.py     # DistilBERT fine-tuné complet
└── ex2/                            # Résumé automatique
    ├── t5_zeroshot.py              # T5-small zero-shot
    ├── t5_finetuned.py             # T5-small fine-tuné (beam search)
    └── t5_finetuned_sampling.py    # T5-small fine-tuné (sampling)
```

---

## 🧪 Exercice 1 — Classification de texte (AG News)

### Dataset
- **AG News** : 4 classes (World, Sports, Business, Sci/Tech)
- 4 000 exemples d'entraînement, 1 000 de test
- 3 seeds (42, 123, 456) pour garantir la reproductibilité

### Modèles comparés

| Modèle | Fichier | Description |
|--------|---------|-------------|
| TF-IDF + SVM | `tfidf_svm.py` | Baseline lexicale, sans GPU |
| GloVe + LR | `glove_probing.py` | Embeddings statiques moyennés |
| BiLSTM + GloVe | `bilstm_glove.py` | BiLSTM (hidden=64, embed=100) |
| Transformer scratch | `transformer_scratch.py` | 2 couches, 2 têtes, embed=64 |
| DistilBERT frozen | `distilbert_frozen.py` | Couches gelées, tête seule entraînée |
| DistilBERT fine-tuné | `distilbert_finetuned.py` | Tout le modèle, lr=2e-5 |

### Métriques évaluées
- Micro accuracy, Macro accuracy
- F1-score (macro), AUROC

---

## 📝 Exercice 2 — Résumé automatique (SAMSum)

### Dataset
- **SAMSum** : résumés de dialogues (type SMS/chat)
- 2 000 exemples d'entraînement, 200 de test

### Modèles comparés

| Modèle | Fichier | Description |
|--------|---------|-------------|
| T5-small zero-shot | `t5_zeroshot.py` | Préfixe "summarize:", beam search |
| T5-small fine-tuné | `t5_finetuned.py` | 3 epochs, lr=3e-4, beam search |
| T5-small fine-tuné + sampling | `t5_finetuned_sampling.py` | Même modèle, décodage par sampling |

### Métriques évaluées
- ROUGE-1, ROUGE-2, ROUGE-L
- Similarité sémantique (SentenceTransformer)

---

## 🛠️ Technologies utilisées

- **Python 3.13**
- **PyTorch** — BiLSTM, Transformer from scratch, fine-tuning
- **HuggingFace Transformers** — DistilBERT, T5
- **scikit-learn** — TF-IDF, SVM, métriques
- **SentenceTransformer** — similarité sémantique
- **NumPy / Pandas**

---

## ▶️ Installation

```bash
pip install torch transformers datasets scikit-learn sentence-transformers numpy pandas
```

### Lancer un modèle

```bash
# Exercice 1 — Baseline TF-IDF + SVM
python ex1/tfidf_svm.py

# Exercice 1 — DistilBERT fine-tuné
python ex1/distilbert_finetuned.py

# Exercice 2 — T5 zero-shot
python ex2/t5_zeroshot.py

# Exercice 2 — T5 fine-tuné
python ex2/t5_finetuned.py
```

---

## 🔁 Reproductibilité

Tous les modèles utilisent 3 seeds fixes (42, 123, 456) via la fonction `seed_everything()` dans `utils.py`, qui fixe `random`, `numpy`, `torch` et `cudnn` pour garantir des résultats reproductibles.

---

## 👤 Auteur

**Salma Benamar** — [@saliza21](https://github.com/saliza21)  
Master 1 IA & Data Science en Santé — Université de Caen Normandie
