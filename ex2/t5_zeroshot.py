"""
Exercice 2 - T5-small en zero-shot avec le préfixe "summarize: "
Beam search, déterministe -> pas besoin de plusieurs runs.
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from utils import seed_everything, compute_rouge_n, compute_rouge_l
import numpy as np
import torch
from datasets import load_dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sentence_transformers import SentenceTransformer

SEED = 42
TEST_SIZE = 200
MAX_INPUT_LEN = 512
MAX_OUTPUT_LEN = 100
BEAM_SIZE = 4


def load_samsum_test(test_size, seed=42):
    ds = load_dataset("knkarthick/samsum")
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(ds["test"]), size=test_size, replace=False)
    dialogues = [ds["test"][int(i)]["dialogue"] for i in idx]
    summaries = [ds["test"][int(i)]["summary"] for i in idx]
    return dialogues, summaries


def compute_embedding_similarity(candidates, references, sim_model):
    """Cosinus entre sentence embeddings."""
    cand_embs = sim_model.encode(candidates, normalize_embeddings=True)
    ref_embs = sim_model.encode(references, normalize_embeddings=True)
    sims = np.sum(cand_embs * ref_embs, axis=1)
    return sims


def main():
    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Chargement du modèle T5-small...")
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small").to(device)
    model.eval()

    print("Chargement de SAMSum test...")
    dialogues, references = load_samsum_test(TEST_SIZE)

    print("Chargement du modèle de similarité...")
    sim_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # Génération
    candidates = []
    print("Génération des résumés...")
    for i, dialogue in enumerate(dialogues):
        input_text = "summarize: " + dialogue
        inputs = tokenizer(input_text, return_tensors="pt", max_length=MAX_INPUT_LEN, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_length=MAX_OUTPUT_LEN,
                num_beams=BEAM_SIZE,
                do_sample=False,
                early_stopping=True
            )
        decoded = tokenizer.decode(output[0], skip_special_tokens=True)
        candidates.append(decoded)

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{TEST_SIZE} générés")

    # Calcul des métriques
    rouge1_scores, rouge2_scores, rougel_scores = [], [], []
    for cand, ref in zip(candidates, references):
        rouge1_scores.append(compute_rouge_n(cand, ref, n=1)["f1"])
        rouge2_scores.append(compute_rouge_n(cand, ref, n=2)["f1"])
        rougel_scores.append(compute_rouge_l(cand, ref)["f1"])

    emb_sims = compute_embedding_similarity(candidates, references, sim_model)

    # Résultats
    print("\n" + "="*60)
    print("T5-small zero-shot (beam=4) — Résultats")
    print("="*60)
    print(f"  ROUGE-1: {np.mean(rouge1_scores):.4f}")
    print(f"  ROUGE-2: {np.mean(rouge2_scores):.4f}")
    print(f"  ROUGE-L: {np.mean(rougel_scores):.4f}")
    print(f"  EmbSim:  {np.mean(emb_sims):.4f}")

    # Exemples qualitatifs
    print("\n--- Exemples qualitatifs ---")
    for i in [0, 1, 2]:
        print(f"\n[Exemple {i+1}]")
        print(f"Dialogue: {dialogues[i][:200]}...")
        print(f"Référence: {references[i]}")
        print(f"Généré: {candidates[i]}")
        print(f"ROUGE-1: {rouge1_scores[i]:.3f}, EmbSim: {emb_sims[i]:.3f}")

    # Déterministe -> un seul run suffit, pas d'écart-type
    print("\nNote: décodage déterministe (beam search), pas de std reporté.")


if __name__ == "__main__":
    main()
