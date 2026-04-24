"""SFT warmup for GRPO training.

Problem: small models (0.6B-1.5B) produce 40-60% invalid JSON on their first
attempts at a structured-action schema. Every generation in a GRPO group
falls back to the same default action → zero reward variance → no gradient.

Solution: 1-2 epochs of supervised fine-tuning on a small set of expert
trajectories teaches the action JSON schema (role, action_type, config_edits
shape, etc). The resulting LoRA adapter is then merged into the base
before GRPO attaches its own LoRA on top — see `grpo_train.py
--sft-adapter-path`.

Expected wall-time on Kaggle T4: ~15 min for 30 trajectories × 2 epochs.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger("sft_warmup")


def load_trajectories(path: str):
    """Load JSONL trajectories. Each row must be ``{"messages": [...]}``.

    Skips comment lines starting with `#` and blank lines so authors can
    annotate the file; requires at least 5 rows total.
    """
    from datasets import Dataset

    traj_path = Path(path)
    if not traj_path.exists():
        raise FileNotFoundError(f"Trajectories not found: {path}")

    rows: List[Dict[str, Any]] = []
    with open(traj_path, "r", encoding="utf-8") as f:
        for line_num, raw in enumerate(f, 1):
            line = raw.strip()
            if not line or line.startswith("#") or line.startswith("//"):
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning("line %d invalid JSON: %s", line_num, e)
                continue
            if "messages" not in data:
                logger.warning("line %d missing 'messages' field, skipping", line_num)
                continue
            rows.append(data)

    logger.info("Loaded %d trajectories from %s", len(rows), path)
    if len(rows) < 5:
        raise ValueError(
            f"Need at least 5 trajectories, got {len(rows)}. Check {path} has "
            f"non-comment JSONL lines with a 'messages' field."
        )
    return Dataset.from_list(rows)


def format_example(example: Dict[str, Any], tokenizer, max_length: int = 1024) -> Dict[str, str]:
    """Render chat-formatted text from trajectory messages for SFT."""
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


def _build_sft_config(output_dir, epochs, batch_size, learning_rate,
                     warmup_ratio, max_seq_length, seed):
    """Build an SFTConfig in a way that tolerates TRL API drift.

    Newer TRL renamed some kwargs; older TRL accepts different combos.
    We start from the most explicit set and fall back on TypeError.
    """
    from trl import SFTConfig

    base_kwargs = dict(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        logging_steps=2,
        save_strategy="epoch",
        report_to="none",
        seed=seed,
    )
    for extra_key, extra_val in (
        ("max_seq_length", max_seq_length),
        ("dataset_text_field", "text"),
    ):
        try:
            SFTConfig(**{**base_kwargs, extra_key: extra_val})
            base_kwargs[extra_key] = extra_val
        except TypeError:
            logger.warning("SFTConfig rejected kwarg %s; proceeding without", extra_key)
    return SFTConfig(**base_kwargs)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description="SFT warmup for GRPO")
    parser.add_argument("--model", required=True,
                        help="Base model id, e.g. unsloth/Qwen3-0.6B-bnb-4bit")
    parser.add_argument("--trajectories", default="data/sft_trajectories.jsonl",
                        help="Path to JSONL file of {'messages': [...]} rows")
    parser.add_argument("--output-dir", required=True,
                        help="Where to save the adapter (adapter_config.json lives at <output-dir>/final)")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    try:
        from unsloth import FastLanguageModel
        from trl import SFTTrainer
    except ImportError as e:
        logger.error(
            "SFT warmup requires GPU deps (torch+CUDA, unsloth, trl). "
            "Install: pip install -e '.[training]'. Import error: %s", e,
        )
        raise SystemExit(1)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "final").mkdir(parents=True, exist_ok=True)

    logger.info("Loading base model: %s", args.model)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )

    logger.info("Configuring LoRA for SFT (r=%d, alpha=%d)", args.lora_r, args.lora_alpha)
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        bias="none",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )

    logger.info("Loading trajectories from %s", args.trajectories)
    dataset = load_trajectories(args.trajectories)

    logger.info("Formatting dataset with chat template")
    dataset = dataset.map(
        lambda ex: format_example(ex, tokenizer, args.max_seq_length),
        remove_columns=[c for c in dataset.column_names if c != "messages"],
    )

    sft_config = _build_sft_config(
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        max_seq_length=args.max_seq_length,
        seed=args.seed,
    )

    # SFTTrainer in newer TRL uses `processing_class`, older used `tokenizer`.
    try:
        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=dataset,
            args=sft_config,
        )
    except TypeError:
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            args=sft_config,
        )

    logger.info("Starting SFT: %d epochs on %d trajectories", args.epochs, len(dataset))
    trainer.train()

    final_dir = out_dir / "final"
    logger.info("Saving adapter to %s", final_dir)
    trainer.model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    logger.info("SFT warmup complete.")
    logger.info(
        "Use in GRPO:  python training/grpo_train.py --sft-adapter-path %s ...",
        final_dir,
    )


if __name__ == "__main__":
    main()
