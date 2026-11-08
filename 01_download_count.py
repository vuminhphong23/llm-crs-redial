# -*- coding: utf-8 -*-
"""Buoc 1: Tai ReDial, DEM that, chia tap. Chay KHONG CAN GPU.
Dung:  python 01_download_count.py [--data_dir data] [--seed 42]

Viec lam:
  1. Tai file zip chinh thuc (link da kiem chung), giai nen.
  2. Dem that so hoi thoai / luot noi / phim -> ghi data/data_card.json.
     So trong bai bao PHAI lay tu file nay, khong chep so trong sach.
  3. Chia file train goc (10.006) thanh train/valid theo conversationId,
     ti le 90/10, seed 42. File test goc (1.342) GIU NGUYEN.
  4. Kiem tra ro ri: conversationId cua train/valid/test khong duoc trung nhau.
"""
import argparse
import json
import os
import random
import urllib.request
import zipfile

from common import build_catalogue, count_stats, load_dialogues, save_jsonl

ZIP_URL = "https://github.com/ReDialData/website/raw/data/redial_dataset.zip"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--valid_fraction", type=float, default=0.10)
    args = ap.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)
    zip_path = os.path.join(args.data_dir, "redial_dataset.zip")
    if not os.path.exists(zip_path):
        print(f"Dang tai {ZIP_URL} ...")
        urllib.request.urlretrieve(ZIP_URL, zip_path)
    print("Dang giai nen ...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(args.data_dir)

    train_path = os.path.join(args.data_dir, "train_data.jsonl")
    test_path = os.path.join(args.data_dir, "test_data.jsonl")
    csv_path = os.path.join(args.data_dir, "movies_with_mentions.csv")
    train_all = load_dialogues(train_path)
    test = load_dialogues(test_path)
    catalogue = build_catalogue(train_all + test, csv_path)

    rng = random.Random(args.seed)
    ids = [d.get("conversationId") for d in train_all]
    rng.shuffle(ids)
    n_valid = int(round(len(ids) * args.valid_fraction))
    valid_ids = set(ids[:n_valid])
    train = [d for d in train_all if d.get("conversationId") not in valid_ids]
    valid = [d for d in train_all if d.get("conversationId") in valid_ids]

    # Kiem tra ro ri theo conversationId (phim duoc phep trung, hoi thoai thi khong)
    t_ids = {d.get("conversationId") for d in train}
    v_ids = {d.get("conversationId") for d in valid}
    te_ids = {d.get("conversationId") for d in test}
    assert not (t_ids & v_ids) and not (t_ids & te_ids) and not (v_ids & te_ids), "LEAK: conversationId trung nhau giua cac tap!"

    save_jsonl(train, os.path.join(args.data_dir, "train_split.jsonl"))
    save_jsonl(valid, os.path.join(args.data_dir, "valid_split.jsonl"))
    save_jsonl(test, os.path.join(args.data_dir, "test_split.jsonl"))
    with open(os.path.join(args.data_dir, "catalogue.json"), "w", encoding="utf-8") as f:
        json.dump(catalogue, f, ensure_ascii=False, indent=1)

    card = {
        "source": ZIP_URL,
        "licence": "CC BY 4.0 (theo trang chu ReDial)",
        "seed": args.seed,
        "valid_fraction_of_train_file": args.valid_fraction,
        "train_file": count_stats(train_all),
        "test_file_official": count_stats(test),
        "split_train": count_stats(train),
        "split_valid": count_stats(valid),
        "split_test_official_untouched": count_stats(test),
        "catalogue_movies": len(catalogue),
        "leakage_check_conversationId_overlap": 0,
        "note": "So nay la so DEM TU FILE DA TAI. Dung so nay trong bai bao, kem theo phien ban/link tren.",
    }
    with open(os.path.join(args.data_dir, "data_card.json"), "w", encoding="utf-8") as f:
        json.dump(card, f, ensure_ascii=False, indent=2)
    print(json.dumps(card, ensure_ascii=False, indent=2))
    print("XONG. Kiem tra ro ri: 0 hoi thoai trung nhau.")


if __name__ == "__main__":
    main()
