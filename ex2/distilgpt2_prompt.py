"""
Exercice 2 - distilgpt2 en mode prompt (decoder-only)
On formule la tâche comme une continuation de texte.
Déterministe (greedy) -> un seul run.
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from utils import seed_everything, compute_rouge_n, compute_rouge_l
import numpy as np
import torch
from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from sentence_transformers import SentenceTransformer

SEED = 42
TEST_SIZE = 200
MAX_INPUT_LEN = 700  # gpt2 a un contexte de 1024 tokens
MAX_OUTPUT_LEN = 80

# Template de prompt
PROMPT_TEMPLATE = """Below is a conversation between two or more people. Write a short summary of the conversation in one paragraph.

Conversation:
{dialogue}

Summary:"""


def load_samsum_test(test_size, seed=42):
    ds = load_dataset("knkarthick/samsum")
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(ds["test"]), size=test_size, replace=False)
    dialogues = [ds["test"][int(i)]["dialogue"] for i in idx]
    summaries = [ds["test"][int(i)]["summary"] for i in idx]
    return dialogues, summaries


def compute_embedding_similarity(candidates, references, sim_model):
    cand_embs = sim_model.encode(candidates, normalize_embeddings=True)
    ref_embs = sim_model.encode(references, normalize_embeddings=True)
    return np.sum(cand_embs * ref_embs, axis=1)


def main():
    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Chargement de distilgpt2...")
    tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained("distilgpt2").to(device)
    model.eval()

    print("Chargement des données et du modèle de similarité...")
    dialogues, references = load_samsum_test(TEST_SIZE)
    sim_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    candidates = []
    print("Génération des résumés...")
    for i, dialogue in enumerate(dialogues):
        prompt = PROMPT_TEMPLATE.format(dialogue=dialogue)

        inputs = tokenizer(
            prompt, return_tensors="pt",
            max_length=MAX_INPUT_LEN, truncation=True
        )
        input_len = inputs["input_ids"].shape[1]
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=MAX_OUTPUT_LEN,
                do_sample=False,  # greedy
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.encode("\n\n")[0] if "\n\n" in tokenizer.get_vocab() else tokenizer.eos_token_id,
            )

        # extraire seulement la partie générée (après le prompt)
        generated_ids = output[0][input_len:]
        decoded = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        # prendre la première phrase / premier paragraphe
        if "\n" in decoded:
            decoded = decoded.split("\n")[0].strip()
        candidates.append(decoded)

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{TEST_SIZE}")

    # Métriques
    r1, r2, rl = [], [], []
    for cand, ref in zip(candidates, references):
        r1.append(compute_rouge_n(cand, ref, n=1)["f1"])
        r2.append(compute_rouge_n(cand, ref, n=2)["f1"])
        rl.append(compute_rouge_l(cand, ref)["f1"])

    emb_sims = compute_embedding_similarity(candidates, references, sim_model)

    print("\n" + "="*60)
    print("distilgpt2 prompt-based (greedy) — Résultats")
    print("="*60)
    print(f"  ROUGE-1: {np.mean(r1):.4f}")
    print(f"  ROUGE-2: {np.mean(r2):.4f}")
    print(f"  ROUGE-L: {np.mean(rl):.4f}")
    print(f"  EmbSim:  {np.mean(emb_sims):.4f}")

    # Exemples
    print("\n--- Exemples qualitatifs ---")
    for i in [0, 1, 2]:
        print(f"\n[Exemple {i+1}]")
        print(f"Dialogue: {dialogues[i][:200]}...")
        print(f"Référence: {references[i]}")
        print(f"Généré: {candidates[i][:200]}")
        print(f"ROUGE-1: {r1[i]:.3f}, EmbSim: {emb_sims[i]:.3f}")

    print("\nNote: décodage déterministe (greedy), pas de std reporté.")


if __name__ == "__main__":
    main()
