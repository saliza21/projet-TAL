"""
Exercice 2 - T5-small fine-tuné + nucleus sampling
Même modèle que t5_finetuned.py mais avec top-p sampling au lieu de beam.
Stochastique -> 3 runs avec seeds différents.
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from utils import seed_everything, compute_rouge_n, compute_rouge_l
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sentence_transformers import SentenceTransformer

SEEDS = [42, 123, 456]
TRAIN_SIZE = 2000
TEST_SIZE = 200
MAX_INPUT_LEN = 512
MAX_OUTPUT_LEN = 100
BATCH_SIZE = 8
EPOCHS = 3
LR = 3e-4

# Paramètres de sampling
TOP_P = 0.9
TEMPERATURE = 0.8


class SamSumDataset(Dataset):
    def __init__(self, dialogues, summaries, tokenizer, max_input, max_output):
        self.dialogues = dialogues
        self.summaries = summaries
        self.tokenizer = tokenizer
        self.max_input = max_input
        self.max_output = max_output

    def __len__(self):
        return len(self.dialogues)

    def __getitem__(self, idx):
        input_text = "summarize: " + self.dialogues[idx]
        target_text = self.summaries[idx]
        source = self.tokenizer(input_text, max_length=self.max_input, truncation=True, padding="max_length", return_tensors="pt")
        target = self.tokenizer(target_text, max_length=self.max_output, truncation=True, padding="max_length", return_tensors="pt")
        labels = target["input_ids"].squeeze(0).clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        return {
            "input_ids": source["input_ids"].squeeze(0),
            "attention_mask": source["attention_mask"].squeeze(0),
            "labels": labels
        }


def load_samsum_subsets(train_size, test_size, seed=42):
    ds = load_dataset("knkarthick/samsum")
    rng = np.random.RandomState(seed)
    train_idx = rng.choice(len(ds["train"]), size=train_size, replace=False)
    test_idx = rng.choice(len(ds["test"]), size=test_size, replace=False)
    return (
        [ds["train"][int(i)]["dialogue"] for i in train_idx],
        [ds["train"][int(i)]["summary"] for i in train_idx],
        [ds["test"][int(i)]["dialogue"] for i in test_idx],
        [ds["test"][int(i)]["summary"] for i in test_idx],
    )


def compute_embedding_similarity(candidates, references, sim_model):
    cand_embs = sim_model.encode(candidates, normalize_embeddings=True)
    ref_embs = sim_model.encode(references, normalize_embeddings=True)
    return np.sum(cand_embs * ref_embs, axis=1)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    sim_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    train_dial, train_sum, test_dial, test_sum = load_samsum_subsets(TRAIN_SIZE, TEST_SIZE, seed=42)

    all_results = []

    for seed in SEEDS:
        seed_everything(seed)
        print(f"\n--- Seed {seed} ---")

        model = T5ForConditionalGeneration.from_pretrained("t5-small").to(device)
        train_ds = SamSumDataset(train_dial, train_sum, tokenizer, MAX_INPUT_LEN, MAX_OUTPUT_LEN)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

        # Entraînement (identique à t5_finetuned)
        model.train()
        for epoch in range(EPOCHS):
            total_loss = 0
            for batch in train_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                total_loss += loss.item()
            print(f"  Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss/len(train_loader):.4f}")

        # Génération avec nucleus sampling
        model.eval()
        candidates = []
        for dialogue in test_dial:
            input_text = "summarize: " + dialogue
            inputs = tokenizer(input_text, return_tensors="pt", max_length=MAX_INPUT_LEN, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_length=MAX_OUTPUT_LEN,
                    do_sample=True,
                    top_p=TOP_P,
                    temperature=TEMPERATURE,
                    no_repeat_ngram_size=3,
                )
            decoded = tokenizer.decode(output[0], skip_special_tokens=True)
            candidates.append(decoded)

        # Métriques
        r1, r2, rl = [], [], []
        for cand, ref in zip(candidates, test_sum):
            r1.append(compute_rouge_n(cand, ref, n=1)["f1"])
            r2.append(compute_rouge_n(cand, ref, n=2)["f1"])
            rl.append(compute_rouge_l(cand, ref)["f1"])
        emb_sims = compute_embedding_similarity(candidates, test_sum, sim_model)

        result = {"rouge1": np.mean(r1), "rouge2": np.mean(r2), "rougel": np.mean(rl), "embsim": np.mean(emb_sims)}
        all_results.append(result)
        print(f"  R1={result['rouge1']:.4f} R2={result['rouge2']:.4f} RL={result['rougel']:.4f} Emb={result['embsim']:.4f}")

    print("\n" + "="*60)
    print(f"T5-small fine-tuned + nucleus (p={TOP_P}, T={TEMPERATURE}) — mean ± std")
    print("="*60)
    for key in all_results[0]:
        vals = [r[key] for r in all_results]
        print(f"  {key}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")


if __name__ == "__main__":
    main()
