# -*- coding: utf-8 -*-
"""Buoc 3: Prompting baseline - chay LLM ma KHONG huan luyen lai. CAN GPU (Colab).
Dung:
  # Xem prompt truoc, khong ton GPU:
  python 03_prompting_baseline.py --dry_run --n 2
  # Chay that tren 20 vi du valid (smoke test):
  python 03_prompting_baseline.py --split valid --n 20 --mode ranking
  # Ban RAG (retriever TF-IDF thay vi ung vien ngau nhien):
  python 03_prompting_baseline.py --split valid --n 100 --mode rag
  # Ban sinh tu do + kiem tra ao giac:
  python 03_prompting_baseline.py --split valid --n 20 --mode free

Che do ranking (chinh, dung cho bang ket qua):
  Moi vi du, mo hinh nhan danh sach 100 ung vien (1 dap an dung + 99 phim
  khac, tron thu tu, seed co dinh) va phai xep top 10. Do Recall@10 / NDCG@10
  tren bang xep hang do. Day la bai toan GOI Y, khong phai bai viet van tu do.
Che do rag: 100 ung vien lay tu retriever TF-IDF tren TEN PHIM (ReDial goc
  chi co ten + nam, khong co the loai/dien vien - do la han che that, ghi ro
  trong bai). Neu retriever khong tim ra dap an dung thi Recall = 0 o vi du
  do - do chinh la loi cua retriever, khong duoc giau.
Che do free: mo hinh tu goi y 3 phim + giai thich; verifier kiem tra tung
  ten phim co trong catalogue khong -> ti le ao giac (out-of-catalogue rate).
"""
import argparse
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (  # noqa: E402
    distinct_n, evaluate_ranking, load_dialogues, save_jsonl, save_run_log,
    title_to_id,
)

try:
    import yaml
except ImportError:
    yaml = None


def load_config(path):
    cfg = {"model": {"name": "Qwen/Qwen2.5-7B-Instruct", "load_in_4bit": True,
                     "dtype": "bfloat16", "max_new_tokens": 200, "temperature": 0.0},
           "data": {"n_candidates": 100}}
    if path and os.path.exists(path) and yaml:
        with open(path, encoding="utf-8") as f:
            user = yaml.safe_load(f) or {}
        cfg["model"].update(user.get("model", {}))
        cfg["data"].update(user.get("data", {}))
    return cfg


def format_history(history, limit_turns=10):
    lines = []
    for m in history[-limit_turns:]:
        who = "Nguoi dung" if m["speaker"] == "seeker" else "Nguoi goi y"
        lines.append(f"{who}: {m['text']}")
    return "\n".join(lines)


def format_preference(pref, catalogue):
    def names(ids):
        return [catalogue.get(i, i) for i in ids]
    liked = names(pref.get("liked", []))
    disliked = names(pref.get("disliked", []))
    s = "So thich suy luan duoc tu hoi thoai (chi tu lich su tren, khong them thong tin ngoai):\n"
    s += f"- Thich: {', '.join(liked) if liked else '(chua ro)'}\n"
    s += f"- Khong thich: {', '.join(disliked) if disliked else '(chua ro)'}"
    return s


def build_ranking_prompt(ex, candidates, catalogue):
    cand_lines = "\n".join(f"{i + 1}. {catalogue.get(mid, mid)}" for i, mid in enumerate(candidates))
    user = (
        "Ban la he thong goi y phim qua hoi thoai. Duoi day la lich su hoi thoai "
        "va so thich da suy luan. Hay chon ra 10 phim PHU HOP NHAT trong danh sach "
        "ung vien, xep theo thu tu phu hop giam dan. CHI DUOC chon phim trong danh "
        "sach ung vien, KHONG DUOC bia them phim ngoai danh sach.\n\n"
        f"{format_history(ex['history'])}\n\n{format_preference(ex['preference'], catalogue)}\n\n"
        f"Danh sach ung vien:\n{cand_lines}\n\n"
        "Tra loi CHI BANG danh sach so thu tu, moi dong 1 so, vi du:\n1\n5\n12"
    )
    return user


def build_free_prompt(ex, catalogue):
    user = (
        "Ban la he thong goi y phim qua hoi thoai. Dua vao lich su va so thich "
        "duoi day, hay goi y dung 3 bo phim CO THAT va giai thich ngan gon vi sao "
        "phu hop. Neu khong chac phim co that thi DUNG goi y phim do.\n\n"
        f"{format_history(ex['history'])}\n\n{format_preference(ex['preference'], catalogue)}\n\n"
        'Tra loi CHI BANG JSON: {"recommendations": [{"title": "<ten phim day du kem nam>", "reason": "<1 cau>"}, ...]}'
    )
    return user


def parse_ranked_numbers(text, candidates):
    nums = []
    for tok in re.findall(r"\b(\d{1,3})\b", text or ""):
        n = int(tok)
        if 1 <= n <= len(candidates) and n not in nums:
            nums.append(n)
    ranked = [candidates[n - 1] for n in nums]
    # Dien them ung vien chua duoc xep (theo thu tu goc) de du 100 - chi de
    # metric top-10 khong bi thieu khi model tra loi thieu so.
    for mid in candidates:
        if mid not in ranked:
            ranked.append(mid)
    return ranked


def parse_free_titles(text, t2i):
    titles, raw = [], text or ""
    try:
        data = json.loads(raw[raw.index("{"):raw.rindex("}") + 1])
        for item in data.get("recommendations", []):
            titles.append(str(item.get("title", "")).strip())
    except Exception:
        titles = re.findall(r'"title"\s*:\s*"([^"]+)"', raw)
    ids, ooc = [], []
    for t in titles:
        key = re.sub(r"\s+", " ", t.strip().lower())
        if key in t2i:
            ids.append(t2i[key])
        else:
            ooc.append(t)
    return ids, ooc, titles


class Retriever:
    """TF-IDF tren TEN PHIM. Don gian, tai tao duoc, khong can API ngoai.
    Han che: chi dung ten phim (vi ReDial goc khong co the loai/dien vien)."""

    def __init__(self, catalogue):
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
        except ImportError as e:
            raise RuntimeError("Thieu scikit-learn. Chay: pip install -r requirements.txt") from e
        # Bo qua phim thieu ten (title None/rong) — neu khong, TF-IDF se sap
        # khi goi .lower() tren None (loi gap khi chay RAG tren Colab 06/11/2026,
        # phim id '50' trong catalogue co title null).
        pairs = [(i, catalogue[i]) for i in catalogue
                 if catalogue[i] and str(catalogue[i]).strip()]
        self.ids = [p[0] for p in pairs]
        self.titles = [p[1] for p in pairs]
        self.vec = TfidfVectorizer(lowercase=True, token_pattern=r"(?u)\b\w+\b")
        self.mat = self.vec.fit_transform(self.titles)

    def retrieve(self, query, k=100):
        q = self.vec.transform([query])
        scores = (self.mat @ q.T).toarray().ravel()
        order = scores.argsort()[::-1][:k]
        return [self.ids[i] for i in order], [float(scores[i]) for i in order]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--split", default="valid", choices=["valid", "test", "train"])
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--mode", default="ranking", choices=["ranking", "rag", "free"])
    ap.add_argument("--output", default=None)
    ap.add_argument("--adapter", default=None,
                    help="Duong dan thu muc adapter LoRA (vd outputs/lora_qwen25_7b). "
                         "Neu co, nap adapter len tren mo hinh goc roi moi danh gia.")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)

    with open(os.path.join(args.data_dir, "catalogue.json"), encoding="utf-8") as f:
        catalogue = json.load(f)
    examples = load_dialogues(os.path.join(args.data_dir, f"examples_{args.split}.jsonl"))
    if args.n:
        examples = examples[:args.n]
    print(f"split={args.split} n={len(examples)} mode={args.mode} model={cfg['model']['name']}")

    rng = random.Random(42)
    # Loai phim thieu ten khoi tap ung vien (vd id '50' co title null) de ten
    # phim khong bi ghi "None" vao prompt.
    all_ids = [i for i in catalogue if catalogue.get(i) and str(catalogue[i]).strip()]
    retriever = Retriever(catalogue) if args.mode == "rag" else None
    t2i = title_to_id(catalogue) if args.mode == "free" else None

    # --- chuan bi ung vien + prompt cho tung vi du (khong can GPU) ---
    jobs = []
    for ex in examples:
        # Lich su trong vi du da thay @id bang ten phim, nen lay id phim da
        # nhac toi tu preference (liked/disliked/seen) de loai khoi ung vien am.
        history_ids = set(ex["preference"].get("liked", []) + ex["preference"].get("disliked", []) + ex["preference"].get("seen", []))
        if args.mode == "rag":
            liked_titles = " ".join(catalogue.get(i, "") for i in ex["preference"].get("liked", []))
            last_seeker = next((m["text"] for m in reversed(ex["history"]) if m["speaker"] == "seeker"), "")
            query = f"{liked_titles} {last_seeker}"
            cands, _ = retriever.retrieve(query, k=cfg["data"]["n_candidates"])
            prompt = build_ranking_prompt(ex, cands, catalogue)
            jobs.append((ex, cands, prompt, ex["target_id"] in cands))
        elif args.mode == "free":
            jobs.append((ex, None, build_free_prompt(ex, catalogue), None))
        else:
            pool = [i for i in all_ids if i != ex["target_id"] and i not in history_ids]
            negs = rng.sample(pool, cfg["data"]["n_candidates"] - 1)
            cands = negs + [ex["target_id"]]
            rng.shuffle(cands)
            prompt = build_ranking_prompt(ex, cands, catalogue)
            jobs.append((ex, cands, prompt, True))

    if args.dry_run:
        for ex, cands, prompt, _ in jobs[:2]:
            print("\n===== PROMPT (target =", ex["target_title"], ") =====")
            print(prompt[:2000])
        print("\nDry run xong: prompt da dung dinh dang tren. Chay that thi bo --dry_run.")
        return

    # --- can GPU moi chay tiep ---
    import torch
    if not torch.cuda.is_available():
        print("LOI: khong thay GPU. Chay tren Colab co GPU (Runtime > Change runtime type > T4 GPU).")
        print("Neu muon xem prompt thi chay voi --dry_run (khong can GPU).")
        sys.exit(2)
    print("GPU:", torch.cuda.get_device_name(0), "| VRAM:", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), "GB")

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    model_name = cfg["model"]["name"]
    tok = AutoTokenizer.from_pretrained(model_name)
    if cfg["model"].get("load_in_4bit"):
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
        model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb, device_map="auto")
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
        print("Da nap adapter LoRA:", args.adapter)
    model.eval()

    tag = "_lora" if args.adapter else ""
    out_path = args.output or os.path.join("outputs", f"pred_{args.split}_{args.mode}{tag}_n{len(examples)}.jsonl")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    done = set()
    if args.resume and os.path.exists(out_path):
        for p in load_dialogues(out_path):
            done.add((p["conversationId"], p["turn_index"]))
        print(f"Resume: da co {len(done)} vi du trong {out_path}, se bo qua.")

    predictions, free_texts = [], []
    for idx, (ex, cands, prompt, gt_in_cands) in enumerate(jobs):
        key = (ex["conversationId"], ex["turn_index"])
        if key in done:
            continue
        messages = [{"role": "user", "content": prompt}]
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok([text], return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=cfg["model"]["max_new_tokens"],
                                 temperature=cfg["model"]["temperature"], do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        raw = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        rec = {"conversationId": ex["conversationId"], "turn_index": ex["turn_index"],
               "target_id": ex["target_id"], "target_title": ex["target_title"], "raw_output": raw}
        if args.mode == "free":
            ids, ooc, titles = parse_free_titles(raw, t2i)
            rec.update({"ranked_ids": ids, "recommended_titles": titles,
                        "out_of_catalogue_titles": ooc, "n_recommended": len(titles)})
            free_texts.append(raw)
        else:
            rec.update({"ranked_ids": parse_ranked_numbers(raw, cands),
                        "candidate_ids": cands, "target_in_candidates": bool(gt_in_cands)})
            predictions.append(rec)
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if (idx + 1) % 10 == 0 or idx == len(jobs) - 1:
            print(f"  {idx + 1}/{len(jobs)} xong, target: {ex['target_title']}")

    # --- danh gia bang 1 ham metric chung ---
    all_preds = load_dialogues(out_path)
    if args.mode == "free":
        ooc_n = sum(len(p.get("out_of_catalogue_titles", [])) for p in all_preds)
        tot_n = sum(p.get("n_recommended", 0) for p in all_preds)
        metrics = {
            "n": len(all_preds),
            "out_of_catalogue_rate": ooc_n / tot_n if tot_n else 0.0,
            "distinct_2_response": distinct_n([p.get("raw_output", "") for p in all_preds], 2),
            "mean_recommended_per_example": tot_n / len(all_preds) if all_preds else 0.0,
        }
    else:
        metrics = evaluate_ranking(all_preds)
        if args.mode == "rag":
            metrics["retrieval_recall@100"] = sum(1 for p in all_preds if p.get("target_in_candidates")) / len(all_preds) if all_preds else 0.0
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    metrics_path = out_path.replace(".jsonl", "_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    save_run_log("runs", f"{args.split}_{args.mode}_n{len(examples)}.json",
                 {"command": " ".join(sys.argv), "config": cfg, "split": args.split,
                  "mode": args.mode, "n": len(examples), "metrics": metrics,
                  "predictions_file": out_path, "metrics_file": metrics_path})
    print("Ket qua:", metrics_path)


if __name__ == "__main__":
    main()
