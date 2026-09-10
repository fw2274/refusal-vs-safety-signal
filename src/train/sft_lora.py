"""bf16 LoRA SFT for Qwen2.5-1.5B-Instruct on 8 GB VRAM.

    python -m src.train.sft_lora --config configs/qwen2.5-1.5b.yaml

Deliberately uses plain HF Trainer with manual label masking rather than TRL's SFTTrainer:
the completion-masking API has churned across TRL versions, and getting the mask wrong (i.e.
training on prompt tokens) silently changes what the model learns.

NO QUANTIZATION. 4-bit QLoRA would perturb the activations this whole project measures.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import yaml
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

from src.prompting import prompt_ids, turn_end_id


class MaskedSFTDataset(Dataset):
    """Prompt tokens get label -100; loss is computed on the completion only."""

    def __init__(self, df: pd.DataFrame, tok, max_len: int):
        self.rows = []
        self.end_id = turn_end_id(tok)
        n_trunc = 0
        for prompt, completion in zip(df["prompt"], df["completion"]):
            p_ids = prompt_ids(tok, prompt)
            c_ids = tok(completion, add_special_tokens=False)["input_ids"] + [self.end_id]
            budget = max_len - len(p_ids)
            if budget < 8:
                n_trunc += 1
                continue  # prompt alone eats the window; drop rather than truncate the prompt
            if len(c_ids) > budget:
                c_ids = c_ids[: budget - 1] + [self.end_id]
                n_trunc += 1
            self.rows.append(
                {
                    "input_ids": p_ids + c_ids,
                    "labels": [-100] * len(p_ids) + c_ids,
                }
            )
        if n_trunc:
            print(f"  {n_trunc} examples truncated or dropped at max_len={max_len}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        return self.rows[i]


def collate(batch: list[dict], pad_id: int) -> dict:
    n = max(len(b["input_ids"]) for b in batch)
    return {
        "input_ids": torch.tensor(
            [b["input_ids"] + [pad_id] * (n - len(b["input_ids"])) for b in batch]
        ),
        "labels": torch.tensor(
            [b["labels"] + [-100] * (n - len(b["labels"])) for b in batch]
        ),
        "attention_mask": torch.tensor(
            [[1] * len(b["input_ids"]) + [0] * (n - len(b["input_ids"])) for b in batch]
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--policy", default=None, help="data subdir; defaults to config data.refusal_policy")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--max-steps", type=int, default=-1, help="smoke test: e.g. 5")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    mcfg, scfg = cfg["model"], cfg["sft"]
    set_seed(scfg["seed"])

    policy_dir = args.policy or cfg["data"]["refusal_policy"]
    proc = Path(cfg["paths"]["processed"]) / policy_dir
    train_df = pd.read_parquet(proc / "sft_train.parquet")
    policy = train_df["policy"].iloc[0]
    run = args.run_name or f"{mcfg['id'].split('/')[-1]}-{policy}"
    out_dir = Path(cfg["paths"]["runs"]) / run
    out_dir.mkdir(parents=True, exist_ok=True)

    if train_df["needs_generation"].any():
        raise SystemExit(
            f"{train_df['needs_generation'].sum()} compliance targets lack a reference "
            "answer. Run src/data/generate_missing.py first."
        )

    print(f"Loading {mcfg['id']} in {mcfg['dtype']} ...")
    tok = AutoTokenizer.from_pretrained(mcfg["id"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        mcfg["id"],
        dtype=getattr(torch, mcfg["dtype"]),
        attn_implementation=mcfg["attn_implementation"],
        device_map={"": 0} if torch.cuda.is_available() else None,
    )
    model.config.use_cache = False

    lora = LoraConfig(
        r=scfg["lora_r"],
        lora_alpha=scfg["lora_alpha"],
        lora_dropout=scfg["lora_dropout"],
        target_modules=scfg["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    if scfg["grad_checkpointing"]:
        model.enable_input_require_grads()  # required: PEFT + checkpointing

    ds = MaskedSFTDataset(train_df, tok, scfg["max_seq_len"])
    print(f"train examples: {len(ds)}")

    # transformers 5.x removed warmup_ratio; derive warmup_steps ourselves.
    eff_batch = scfg["per_device_batch_size"] * scfg["grad_accum"]
    total_steps = max(1, (len(ds) // eff_batch) * scfg["epochs"])
    warmup_steps = max(1, int(scfg["warmup_ratio"] * total_steps))
    print(f"eff. batch {eff_batch} | ~{total_steps} steps | {warmup_steps} warmup")

    targs = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=scfg["per_device_batch_size"],
        gradient_accumulation_steps=scfg["grad_accum"],
        num_train_epochs=scfg["epochs"],
        max_steps=args.max_steps,
        learning_rate=scfg["lr"],
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine",
        bf16=True,
        gradient_checkpointing=scfg["grad_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch",
        logging_steps=5,
        save_strategy="no",
        report_to=[],
        seed=scfg["seed"],
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds,
        data_collator=lambda b: collate(b, tok.pad_token_id),
    )
    result = trainer.train()

    model.save_pretrained(out_dir / "adapter")
    tok.save_pretrained(out_dir / "adapter")
    (out_dir / "train_meta.json").write_text(
        json.dumps(
            {
                "policy": policy,
                "n_train": len(ds),
                "metrics": result.metrics,
                "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)
                if torch.cuda.is_available()
                else None,
                "config": cfg,
            },
            indent=2,
            default=str,
        )
    )
    print(f"\nadapter -> {out_dir / 'adapter'}")
    if torch.cuda.is_available():
        print(f"peak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
