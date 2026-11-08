# LLM in Conversational Recommender Systems and User Preference Reasoning (ReDial)

A multi-turn conversational recommender: the system chats with the user, infers what they
like from the dialogue, recommends **real items from a closed catalogue** with an
explanation — and does not hallucinate item names that do not exist.

This is the experiment code for a Master's thesis (Computer Science, Phenikaa University).
The research question: can a large language model, grounded in retrieved item knowledge
(RAG) and checked by a deterministic verifier, infer user preferences from multi-turn
dialogue and generate recommendations that are accurate, trustworthy, and explainable?

- **Backbone:** Qwen2.5-7B-Instruct, 4-bit (QLoRA) — runs on a Colab T4.
- **Data:** ReDial (Li et al., NeurIPS 2018), movie domain only, English only.
- **Approach:** explicit preference state → retrieval of a closed candidate set → constrained
  generation → deterministic verifier (a recommendation is only emitted if it is in the
  retrieved set).

> **Status: work in progress.** This repo shows the work done so far — data pipeline and
> prompting baselines are implemented and measured. LoRA training, the KBRD/KGSF/UniCRS
> baselines, human evaluation, and the final test-set run are not done yet. See
> [Status](#status) below.

## Status

| Part | State |
|---|---|
| Download ReDial + count from the downloaded files | ✅ Done |
| Map each item name to item knowledge (genre, actors, director, plot) | ✅ Done |
| Dialogue-based split: train 9,005 / valid 1,001 / test 1,342 dialogues, leakage 0 | ✅ Done |
| Data cleaning (dialogue cleanup, title normalisation, like/dislike labels) | ✅ Done |
| Prompting baseline (ranking / RAG / free generation), valid n=200 | ✅ Done — see [Preliminary results](#preliminary-results) |
| LoRA fine-tuning (r=16, α=32, lr 2e-4, 1 epoch) | ⏳ Smoke test done (200 examples, adapter saved); full training not started |
| Baselines KBRD / KGSF / UniCRS (re-run on the same split) | ❌ Not done — needs CRSLab, a separate environment |
| Human evaluation (naturalness, coherence, explanation quality) | ❌ Not done |
| Test set (1,342 dialogues) | 🔒 Untouched — run once, for the final configuration only |

## Preliminary results

> **PRELIMINARY — do not cite as final results.** Measured on the validation split only
> (n = 200 examples), a single seed (42), one backbone (Qwen2.5-7B-Instruct, 4-bit,
> temperature 0.0). No mean ± std over multiple seeds yet, no significance test, no test
> set. Final results will replace these numbers.

Ranking setup: at a recommender turn, the model ranks 100 candidates (1 ground truth + 99
random negatives, fixed seed). Random baseline Recall@10 = 0.10.

| Method | Recall@1 | Recall@10 | NDCG@10 | Recall@50 | MRR@10 |
|---|---|---|---|---|---|
| Prompting only (ranking) | 0.125 | **0.395** | **0.245** | 0.640 | 0.199 |
| Prompting + RAG (TF-IDF retriever) | 0.000 | 0.015 | 0.008 | 0.015 | 0.006 |
| Free generation | — | — | — | — | — |

Prompting reaches Recall@10 = 0.395 — **nearly 4× the random baseline (0.10)** — with no
fine-tuning at all.

| Dialogue quality (free generation, n=200) | Value |
|---|---|
| Out-of-catalogue rate (hallucination: recommended items not in the catalogue) | **0.206 (20.6%)** |
| Distinct-2 (response diversity) | 0.419 |
| Mean recommended items per example | 2.94 |

**What the numbers say so far:**

- Prompting alone already works surprisingly well on ReDial — the signal is real.
- RAG fails, but the failure is in the **retriever, not the generator**: the title-only
  TF-IDF retriever puts the ground-truth item in the top-100 candidates in only 2% of cases
  (`retrieval_recall@100 = 0.02`). Fixing retrieval (embedding or knowledge-graph based) is
  the main next step — the RAG idea is not abandoned.
- Free generation hallucinates 20.6% of the time (about 1 in 5 recommended items is made
  up). That is exactly what the verifier is for: only items from the retrieved set may be
  recommended.

Full metric files and predictions: [`results/`](results/) (see
`pred_valid_*_metrics.json`, `pred_valid_*.jsonl`, `valid_*.json` run logs).

## Quickstart

Runs on Google Colab with a T4 GPU. Full step-by-step instructions (Vietnamese) are in
[`COLAB.md`](COLAB.md).

```bash
# 1. Check the GPU (Colab: Runtime > Change runtime type > GPU, T4 or better)
nvidia-smi

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download ReDial, count from the downloaded files, split (no GPU needed, ~1 min)
python 01_download_count.py
# -> data/data_card.json : dialogue / utterance / movie counts counted from the files

# 4. Build evaluation / training examples (no GPU needed)
python 02_build_examples.py

# 5. Inspect the prompt without spending GPU
python 03_prompting_baseline.py --dry_run --n 2

# 6. Run 20 validation examples, ranking mode (GPU, a few minutes)
python 03_prompting_baseline.py --split valid --n 20 --mode ranking

# 7. Recompute metrics from a predictions file (no GPU needed)
python 05_evaluate.py --predictions results/pred_valid_ranking_n200.jsonl
```

Or open `thuc-nghiem-LLM-CRS-Colab.ipynb` in Colab — it runs the same scripts cell by cell.

## Repository structure

```
.
├── 01_download_count.py          # Download ReDial, count dialogues/utterances/movies, split 90/10 by dialogue
├── 02_build_examples.py          # Build ranking / RAG / free-generation examples from the split
├── 03_prompting_baseline.py      # Prompting baseline: --mode ranking | rag | free (+ --adapter for LoRA eval)
├── 04_lora_train.py              # QLoRA fine-tuning (r=16, α=32, lr 2e-4, 1 epoch; --resume, --per_device_batch)
├── 05_evaluate.py                # Recompute metrics from a predictions .jsonl (no GPU)
├── common.py                     # Shared metrics (Recall/NDCG/MRR, Distinct-n), run logging
├── config.yaml                   # Single config for every step (backbone, data, LoRA, paths)
├── requirements.txt              # transformers, peft, bitsandbytes, accelerate, scikit-learn, pandas, pyyaml, tqdm
├── thuc-nghiem-LLM-CRS-Colab.ipynb  # Colab notebook running the same scripts
├── COLAB.md                      # Detailed Colab run guide (Vietnamese; the original README)
└── results/                      # Measured outputs: metrics JSONs, predictions, run logs
```

## Requirements

- Python 3.10+ (Colab default is fine)
- GPU with ~15 GB VRAM for the 4-bit model (Tesla T4 on Colab works; ≥24 GB if you want
  to disable 4-bit loading)
- `pip install -r requirements.txt` (Colab already ships torch)

## Data

ReDial (Li et al., NeurIPS 2018, CC BY 4.0), downloaded by `01_download_count.py` from the
official link. Counts **counted from the downloaded files** (`data/data_card.json`):

| Split | Dialogues | Utterances |
|---|---|---|
| Train file (split 90/10 by dialogue) | 10,006 → train 9,005 / valid 1,001 | 182,150 |
| Official test file (untouched) | 1,342 | 23,952 |
| Catalogue | 6,925 movies | — |
| Leakage (conversation overlap between splits) | 0 | — |

The raw dataset is **not** committed to this repo — the script downloads it. One catalogue
entry (movie id `50`) has a null title; the code handles it.

## Rules this repo follows (so the numbers stay trustworthy)

- One metric implementation for all methods (`common.py`) — no per-method metric code.
- Split by dialogue, never by utterance. The test set is run **once**, for the final
  configuration only; tuning happens on validation.
- No hand-typed numbers: every number in the results table comes from a `results/`
  metrics file produced by the code.
- Baselines are re-run on the same split, never copied from other papers.

## Known limitations (stated plainly)

- The TF-IDF retriever (titles only) is too weak — ReDial has no genre/actor fields in the
  dialogue data itself, so retrieval by title words fails. This is the main open problem.
- KBRD/KGSF/UniCRS baselines are not in this repo (they need CRSLab in a separate
  old-PyTorch environment).
- Everything so far is a single seed on 200 validation examples. No human evaluation yet.

## Citation

```bibtex
@inproceedings{li2018redial,
  title={Towards Deep Conversational Recommendations},
  author={Li, Raymond and Ebrahimi, Samira and Lee, Robert and Chen, Jimmy and Yu, Zhou and Nevatia, Ram and Radev, Dragomir},
  booktitle={NeurIPS},
  year={2018}
}
```

Related: Lewis et al. (RAG, NeurIPS 2020); Hu et al. (LoRA, ICLR 2022).

## License

MIT — see [LICENSE](LICENSE). Author: Vũ Minh Phong, Phenikaa University, 2026.
