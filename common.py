# -*- coding: utf-8 -*-
"""Ham dung chung cho moi buoc thuc nghiem LLM-CRS.
Giai thich ngan gon bang tieng Viet o moi ham, de Phong doc code la hieu.

Du lieu goc (da kiem chung ngay 2026-11-06):
- File zip chinh thuc: https://github.com/ReDialData/website/raw/data/redial_dataset.zip
  (link trong README cua repo goc RaymondLi0/conversational-recommendations)
- Trong zip: train_data.jsonl (10.006 hoi thoai), test_data.jsonl (1.342),
  movies_with_mentions.csv (6.924 phim).
- Moi dong jsonl la 1 hoi thoai: movieMentions {id: "Ten phim (nam)"},
  messages [{senderWorkerId, text}], trong text phim duoc viet dang @<id>,
  respondentQuestions / initiatorQuestions {id: {suggested, seen, liked}}.
  Chu y: 2 truong Questions co khi la [] (rong) thay vi dict -> ham norm_questions
  xu ly ca 2 dang. Gia tri liked/seen: 0 = khong, 1/2 = co (muc do khac nhau).
"""
import json
import math
import os
import random
import re
import time
from collections import Counter

MENTION_RE = re.compile(r"@(\d+)")


def norm_questions(x):
    """Questions co the la dict hoac [] -> luon tra ve dict."""
    return x if isinstance(x, dict) else {}


def load_dialogues(path):
    """Doc file .jsonl, moi dong 1 hoi thoai. Tra ve list[dict]."""
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def save_jsonl(rows, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build_catalogue(dialogues, movies_csv=None):
    """catalogue: {movie_id (str): ten phim}. Lay tu movieMentions cua moi
    hoi thoai; neu co file movies_with_mentions.csv thi bo sung phim con thieu.
    Ten phim trung nhau (vd 2 phim cung ten, khac nam) thi giu phim gap dau
    tien trong catalogue title->id, nhung id->title thi giu het."""
    cat = {}
    for d in dialogues:
        for mid, title in (d.get("movieMentions") or {}).items():
            # Bo qua phim thieu ten (title None/rong). Du lieu ReDial co 1 phim
            # nhu vay (id '50'): de lọt vao catalogue se lam sap RAG (TF-IDF)
            # va FREE (title->id) o buoc sau, phat hien khi chay Colab 06/11/2026.
            if title and str(title).strip():
                cat[str(mid)] = title
    if movies_csv and os.path.exists(movies_csv):
        import csv
        with open(movies_csv, encoding="utf-8", errors="ignore") as f:
            for row in csv.DictReader(f):
                mid, title = str(row["movieId"]), row["movieName"]
                if title and str(title).strip():
                    cat.setdefault(mid, title)
    return cat


def title_to_id(catalogue):
    """Anh xa ten chuan hoa -> id, dung de kiem tra phim model goi y co that khong.
    Bo qua phim thieu ten (title None/rong) — neu khong, .strip() se sap o day."""
    out = {}
    for mid, title in catalogue.items():
        if not title or not str(title).strip():
            continue
        key = re.sub(r"\s+", " ", str(title).strip().lower())
        out.setdefault(key, mid)
    return out


def extract_movie_ids(text):
    return MENTION_RE.findall(text or "")


def replace_mentions(text, catalogue):
    """@12345 -> ten phim, de dua hoi thoai cho LLM doc (nguoi doc cung hieu)."""
    def _rep(m):
        return catalogue.get(m.group(1)) or f"@[{m.group(1)}]"
    return MENTION_RE.sub(_rep, text or "")


def speaker_of(msg, dialogue):
    """Nguoi hoi (seeker) la initiator, nguoi goi y (recommender) la respondent."""
    return "recommender" if msg.get("senderWorkerId") == dialogue.get("respondentWorkerId") else "seeker"


def preference_state(history, dialogue, catalogue):
    """Suy luan so thich P_t theo cach KIEM TRA DUOC (khong dung LLM o day):
    - liked: phim seeker nhac toi, va initiatorQuestions ghi liked >= 1
    - disliked: phim seeker nhac toi, seen >= 1 nhung liked == 0
    - seen: cac phim con lai seeker da nhac toi
    Day la heuristic lam co so cho prompt va retriever; no khong phai nhan
    vang (gold label) cua tap du lieu."""
    q = norm_questions(dialogue.get("initiatorQuestions"))
    liked, disliked, seen = [], [], []
    for msg in history:
        if speaker_of(msg, dialogue) != "seeker":
            continue
        for mid in extract_movie_ids(msg.get("text", "")):
            info = q.get(str(mid), {}) or {}
            if info.get("liked", 0) >= 1:
                liked.append(mid)
            elif info.get("seen", 0) >= 1:
                disliked.append(mid)
            else:
                seen.append(mid)
    def _uniq(xs):
        out = []
        for x in xs:
            if x not in out:
                out.append(x)
        return out
    return {"liked": _uniq(liked), "disliked": _uniq(disliked), "seen": _uniq(seen)}


def build_examples(dialogues, catalogue):
    """Bien moi hoi thoai thanh cac vi du danh gia / huan luyen.
    Quy tac (ghi ro de tai tao duoc):
    - Duyet tung tin nhan theo thu tu. Tai tin nhan cua recommender, neu co
      phim @id ma respondentQuestions ghi suggested == 1 VA phim do chua
      duoc nhac toi truoc do trong hoi thoai -> tao 1 vi du:
      history = moi tin nhan TRUOC tin do, target = phim do.
    - Chi lay vi du khi history co it nhat 2 tin nhan.
    - Moi vi du co P_t (preference_state) tinh tu history.
    Nhu vay target khong bao gio nam trong history, khong ro ri dap an."""
    examples = []
    for d in dialogues:
        suggested = {str(k) for k, v in norm_questions(d.get("respondentQuestions")).items()
                     if isinstance(v, dict) and v.get("suggested", 0) == 1}
        if not suggested:
            continue
        mentioned = set()
        msgs = d.get("messages", [])
        for i, msg in enumerate(msgs):
            ids = extract_movie_ids(msg.get("text", ""))
            if speaker_of(msg, d) == "recommender" and len(msgs[:i]) >= 2:
                new_targets = [mid for mid in ids if mid in suggested and mid not in mentioned]
                if new_targets:
                    target = new_targets[0]
                    history = msgs[:i]
                    examples.append({
                        "conversationId": d.get("conversationId"),
                        "turn_index": i,
                        "history": [
                            {"speaker": speaker_of(m, d),
                             "text": replace_mentions(m.get("text", ""), catalogue)}
                            for m in history
                        ],
                        "target_id": str(target),
                        "target_title": catalogue.get(str(target), f"@[{target}]"),
                        "preference": preference_state(history, d, catalogue),
                    })
            mentioned.update(ids)
    return examples


# ------------------- metrics: 1 ham metric cho moi model -------------------
def recall_at_k(ranked_ids, target_id, k):
    return 1.0 if str(target_id) in [str(x) for x in ranked_ids[:k]] else 0.0


def ndcg_at_k(ranked_ids, target_id, k):
    for i, mid in enumerate(ranked_ids[:k]):
        if str(mid) == str(target_id):
            return 1.0 / math.log2(i + 2)
    return 0.0


def mrr_at_k(ranked_ids, target_id, k):
    for i, mid in enumerate(ranked_ids[:k]):
        if str(mid) == str(target_id):
            return 1.0 / (i + 1)
    return 0.0


def distinct_n(texts, n=2):
    """Ti le n-gram khac nhau / tong n-gram: do da dang cau tra loi."""
    grams, total = set(), 0
    for t in texts:
        toks = (t or "").lower().split()
        for i in range(max(0, len(toks) - n + 1)):
            grams.add(tuple(toks[i:i + n])); total += 1
    return len(grams) / total if total else 0.0


def evaluate_ranking(predictions, k_list=(1, 10, 50)):
    """predictions: list dict co 'target_id' va 'ranked_ids' (list id theo thu
    tu). Tra ve dict metric trung binh. Chi goi ham nay cho MOI model, dung
    chung 1 bo code metric."""
    n = len(predictions)
    out = {"n": n}
    for k in k_list:
        out[f"Recall@{k}"] = sum(recall_at_k(p["ranked_ids"], p["target_id"], k) for p in predictions) / n if n else 0.0
        out[f"NDCG@{k}"] = sum(ndcg_at_k(p["ranked_ids"], p["target_id"], k) for p in predictions) / n if n else 0.0
    out["MRR@10"] = sum(mrr_at_k(p["ranked_ids"], p["target_id"], 10) for p in predictions) / n if n else 0.0
    return out


def save_run_log(log_dir, name, payload):
    """Ghi nhat ky chay: ngay, lenh, cau hinh, ket qua. Khong ghi so bang tay."""
    os.makedirs(log_dir, exist_ok=True)
    payload = dict(payload)
    payload["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    path = os.path.join(log_dir, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


def count_stats(dialogues):
    utt = sum(len(d.get("messages", [])) for d in dialogues)
    movies = set()
    for d in dialogues:
        movies.update(str(k) for k in (d.get("movieMentions") or {}))
    return {"dialogues": len(dialogues), "utterances": utt, "distinct_movies_mentioned": len(movies)}
