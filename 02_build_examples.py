# -*- coding: utf-8 -*-
"""Buoc 2: Bien hoi thoai thanh vi du danh gia / huan luyen. Chay KHONG CAN GPU.
Dung:  python 02_build_examples.py [--data_dir data]

Doc dinh nghia vi du trong common.build_examples truoc khi dung so lieu nay.
Dau ra: examples_train.jsonl, examples_valid.jsonl, examples_test.jsonl
trong thu muc data/. Moi vi du: conversationId, turn_index, history (da thay
@id bang ten phim), target_id, target_title, preference {liked, disliked, seen}.
"""
import argparse
import json
import os

from common import build_catalogue, build_examples, load_dialogues, save_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    args = ap.parse_args()

    with open(os.path.join(args.data_dir, "catalogue.json"), encoding="utf-8") as f:
        catalogue = json.load(f)

    summary = {}
    for split in ["train_split", "valid_split", "test_split"]:
        dialogues = load_dialogues(os.path.join(args.data_dir, f"{split}.jsonl"))
        examples = build_examples(dialogues, catalogue)
        out = os.path.join(args.data_dir, f"examples_{split.split('_')[0]}.jsonl")
        save_jsonl(examples, out)
        summary[split] = {"dialogues": len(dialogues), "examples": len(examples)}
        print(f"{split}: {len(dialogues)} hoi thoai -> {len(examples)} vi du -> {out}")

    # In 2 vi du dau tien cua valid de Phong doc thu bang mat
    valid = load_dialogues(os.path.join(args.data_dir, "examples_valid.jsonl"))
    for ex in valid[:2]:
        print("\n--- Vi du ---")
        print("target:", ex["target_title"])
        print("liked:", [catalogue.get(m, m) for m in ex["preference"]["liked"]][:5])
        for m in ex["history"][-4:]:
            print(f"[{m['speaker']}] {m['text'][:160]}")
    print("\nTom tat:", json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
