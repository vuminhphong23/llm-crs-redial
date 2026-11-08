# -*- coding: utf-8 -*-
"""Buoc 4: LoRA nhe (QLoRA 4-bit) - CAN GPU Colab.
Dung:
  # Chay thu 200 vi du truoc (khoang vai phut tren T4):
  python 04_lora_train.py --max_examples 200 --epochs 1 --output_dir outputs/lora_smoke
  # Chay that tren toan bo train:
  python 04_lora_train.py --epochs 1 --output_dir outputs/lora_qwen25_7b
  # Noi lai tu checkpoint neu Colab ngat giua chung (chi can duong dan checkpoint cuoi):
  python 04_lora_train.py --epochs 1 --output_dir outputs/lora_qwen25_7b --resume outputs/lora_qwen25_7b/checkpoint-400

Cach huan luyen (ghi ro de tai tao duoc):
- Du lieu: vi du train tu 02_build_examples.py. Moi vi du thanh 1 doan chat:
  user = lich su + P_t, assistant = "Toi goi y: <ten phim dung>. Vi ban
  thich <phim liked dau tien>..." (giai thich chi dung thong tin co trong
  lich su, khong bia the loai/dien vien).
- Loss CHI TINH tren phan assistant (completion-only): token cua prompt
  duoc danh dau -100, khong hoc thuoc prompt.
- QLoRA: mo hinh goc 4-bit NF4 dong bang, chi huan luyen adapter LoRA
  r=16, alpha=32, lr 2e-4 (theo config.yaml). Luu adapter + tokenizer ra
  output_dir; chay danh gia bang 03 voi --adapter <duong dan>.
- LUON luu checkpoint len Drive sau khi chay xong (Colab ngat la mat).
- Noi lai sau khi ngat: chay them --resume (noi tu checkpoint moi nhat
  trong output_dir) hoac --resume <duong dan checkpoint>. Khong co co nay
  thi train lai tu dau.
- Do toc do (benchmark): dung --per_device_batch N --grad_accum M de doi
  batch ma khong sua config.yaml; tich N*M giu nguyen = 16 de toan hoc
  khong doi (vi du batch 4 + accum 4).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_dialogues, save_run_log  # noqa: E402

try:
    import yaml
except ImportError:
    yaml = None


def load_config(path):
    cfg = {"model": {"name": "Qwen/Qwen2.5-7B-Instruct", "load_in_4bit": True},
           "lora": {"r": 16, "alpha": 32, "dropout": 0.05, "learning_rate": 0.0002,
                    "epochs": 1, "per_device_batch": 1, "grad_accum": 16, "max_len": 1024,
                    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                                       "gate_proj", "up_proj", "down_proj"]}}
    if path and os.path.exists(path) and yaml:
        with open(path, encoding="utf-8") as f:
            user = yaml.safe_load(f) or {}
        cfg["model"].update(user.get("model", {}))
        cfg["lora"].update(user.get("lora", {}))
    return cfg


def build_texts(examples, catalogue, tokenizer):
    """Tra ve list (prompt_text, full_text). Prompt dung chat template cua Qwen."""
    out = []
    for ex in examples:
        hist = "\n".join(
            f"{'Nguoi dung' if m['speaker'] == 'seeker' else 'Nguoi goi y'}: {m['text']}"
            for m in ex["history"][-10:])
        liked = ex["preference"].get("liked", [])
        why = f" Vi ban thich {catalogue.get(liked[0], liked[0])}." if liked else ""
        user = (f"Dua vao hoi thoai duoi day, hay goi y 1 bo phim phu hop va giai thich ngan gon.\n\n"
                f"{hist}\n\nChi tra loi 1 phim co that.")
        assistant = f"Toi goi y: {ex['target_title']}.{why}"
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True)
        full = prompt + assistant + tokenizer.eos_token
        out.append((prompt, full))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--max_examples", type=int, default=None)
    ap.add_argument("--epochs", type=float, default=None)
    ap.add_argument("--output_dir", default="outputs/lora_qwen25_7b")
    ap.add_argument("--resume", nargs="?", const=True, default=False,
                    help="Noi lai: --resume de noi tu checkpoint moi nhat trong "
                         "output_dir, hoac --resume <duong dan checkpoint>.")
    ap.add_argument("--per_device_batch", type=int, default=None,
                    help="Ghi de config lora.per_device_batch (benchmark toc do).")
    ap.add_argument("--grad_accum", type=int, default=None,
                    help="Ghi de config lora.grad_accum. Giu tich "
                         "per_device_batch*grad_accum = 16 de toan hoc khong doi.")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg["lora"]["epochs"] = args.epochs
    if args.per_device_batch is not None:
        cfg["lora"]["per_device_batch"] = args.per_device_batch
    if args.grad_accum is not None:
        cfg["lora"]["grad_accum"] = args.grad_accum

    import torch
    if not torch.cuda.is_available():
        print("LOI: khong thay GPU. Chay tren Colab co GPU.")
        sys.exit(2)
    print("GPU:", torch.cuda.get_device_name(0))

    with open(os.path.join(args.data_dir, "catalogue.json"), encoding="utf-8") as f:
        catalogue = json.load(f)
    examples = load_dialogues(os.path.join(args.data_dir, "examples_train.jsonl"))
    if args.max_examples:
        examples = examples[:args.max_examples]
    print(f"Train examples: {len(examples)} | epochs: {cfg['lora']['epochs']}")

    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig, Trainer, TrainingArguments)
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    model_name = cfg["model"]["name"]
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb, device_map="auto")
    model = prepare_model_for_kbit_training(model)
    lora_cfg = LoraConfig(r=cfg["lora"]["r"], lora_alpha=cfg["lora"]["alpha"],
                          lora_dropout=cfg["lora"]["dropout"], bias="none",
                          task_type="CAUSAL_LM", target_modules=cfg["lora"]["target_modules"])
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    texts = build_texts(examples, catalogue, tok)
    max_len = cfg["lora"]["max_len"]

    class Dataset(torch.utils.data.Dataset):
        def __len__(self):
            return len(texts)

        def __getitem__(self, i):
            prompt, full = texts[i]
            p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
            f_ids = tok(full, add_special_tokens=False)["input_ids"][:max_len]
            labels = [-100] * len(p_ids) + f_ids[len(p_ids):]
            labels = labels[:max_len]
            return {"input_ids": f_ids, "labels": labels,
                    "attention_mask": [1] * len(f_ids)}

    def collate(batch):
        maxlen = max(len(b["input_ids"]) for b in batch)
        input_ids, labels, attn = [], [], []
        for b in batch:
            pad = maxlen - len(b["input_ids"])
            input_ids.append(b["input_ids"] + [tok.pad_token_id] * pad)
            labels.append(b["labels"] + [-100] * pad)
            attn.append(b["attention_mask"] + [0] * pad)
        return {"input_ids": torch.tensor(input_ids), "labels": torch.tensor(labels),
                "attention_mask": torch.tensor(attn)}

    targs = TrainingArguments(
        output_dir=args.output_dir, num_train_epochs=cfg["lora"]["epochs"],
        per_device_train_batch_size=cfg["lora"]["per_device_batch"],
        gradient_accumulation_steps=cfg["lora"]["grad_accum"],
        learning_rate=cfg["lora"]["learning_rate"], logging_steps=10,
        save_steps=200, save_total_limit=2, bf16=True, optim="paged_adamw_8bit",
        report_to=[], seed=42, gradient_checkpointing=True,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=Dataset(), data_collator=collate)
    import time as _time
    _t0 = _time.time()
    trainer.train(resume_from_checkpoint=args.resume)
    _elapsed = _time.time() - _t0
    _steps = trainer.state.global_step or 0
    print(f"Train xong: {_elapsed/60:.1f} phut, {_steps} buoc, "
          f"{_elapsed/max(_steps, 1):.2f} giay/buoc")
    if torch.cuda.is_available():
        print(f"VRAM dinh: {torch.cuda.max_memory_allocated(0)/1e9:.2f} GB")
    model.save_pretrained(args.output_dir)
    tok.save_pretrained(args.output_dir)
    print("Da luu adapter:", args.output_dir)
    print("NHO: copy thu muc nay len Google Drive ngay truoc khi Colab ngat.")
    save_run_log("runs", "lora_train.json",
                 {"command": " ".join(sys.argv), "config": cfg,
                  "n_train_examples": len(examples), "output_dir": args.output_dir})


if __name__ == "__main__":
    main()
