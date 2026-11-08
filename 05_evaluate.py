# -*- coding: utf-8 -*-
"""Buoc 5: Danh gia file du doan da co. Chay KHONG CAN GPU.
Dung:  python 05_evaluate.py --predictions outputs/pred_valid_ranking_n100.jsonl

Doc file du doan (jsonl, moi dong co target_id + ranked_ids), tinh Recall /
NDCG / MRR bang CUNG 1 ham metric trong common.py nhu buoc 3. Neu 2 lan
chay cho 2 so khac nhau tu cung 1 file thi do loi code - bao ngay.
Voi che do free, tinh them ti le ao giac va Distinct-2.
"""
import argparse
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import distinct_n, evaluate_ranking, load_dialogues  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    args = ap.parse_args()
    preds = load_dialogues(args.predictions)
    if not preds:
        print("File du doan rong:", args.predictions)
        sys.exit(1)
    if "out_of_catalogue_titles" in preds[0]:
        ooc = sum(len(p.get("out_of_catalogue_titles", [])) for p in preds)
        tot = sum(p.get("n_recommended", 0) for p in preds)
        metrics = {"n": len(preds),
                   "out_of_catalogue_rate": ooc / tot if tot else 0.0,
                   "distinct_2_response": distinct_n([p.get("raw_output", "") for p in preds], 2)}
    else:
        metrics = evaluate_ranking(preds)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    out = args.predictions.replace(".jsonl", "_metrics.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print("Da ghi:", out)


if __name__ == "__main__":
    main()
