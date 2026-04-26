"""Plot GRPO reward + loss curves from a TRL trainer_state.json.

Produces outputs/grpo_run1/reward_curve.png with two stacked panels:
  top:    rewards/reward_function/mean per step + ±std band
  bottom: loss + grad_norm + KL on log scale (right axis)

Usage:
  python training/plot_grpo_curve.py \
    --state outputs/grpo_run1/checkpoint-30/trainer_state.json \
    --output outputs/grpo_run1/reward_curve.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--state", required=True, help="path to trainer_state.json")
    p.add_argument("--output", required=True, help="output PNG path")
    args = p.parse_args()

    with open(args.state, encoding="utf-8") as f:
        state = json.load(f)
    log = state.get("log_history", [])
    steps = [e["step"] for e in log]
    reward_mean = [e.get("rewards/reward_function/mean", 0.0) for e in log]
    reward_std = [e.get("rewards/reward_function/std", 0.0) for e in log]
    loss = [e.get("loss", 0.0) for e in log]
    grad_norm = [e.get("grad_norm", 0.0) for e in log]
    kl = [e.get("kl", 0.0) for e in log]

    rm = np.array(reward_mean)
    rs = np.array(reward_std)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    ax1.plot(steps, rm, "o-", color="#2E7D32", label="reward (mean)", linewidth=2)
    ax1.fill_between(steps, rm - rs, rm + rs, color="#2E7D32", alpha=0.18,
                     label="±1 std (per-prompt-group)")
    ax1.axhline(0, color="gray", linewidth=0.5, linestyle=":")
    ax1.set_ylabel("Reward (env step)")
    ax1.set_title("GRPO from raw Qwen3-1.7B-bnb-4bit (30 steps, single-step rollout)")
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.3)

    ax2.plot(steps, loss, "o-", color="#1565C0", label="loss", linewidth=1.5)
    ax2.plot(steps, grad_norm, "s-", color="#F57C00", label="grad_norm", linewidth=1.5)
    ax2.plot(steps, kl, "^-", color="#7B1FA2", label="kl", linewidth=1.5)
    ax2.set_yscale("symlog", linthresh=1e-7)
    ax2.set_xlabel("Training step")
    ax2.set_ylabel("loss / grad_norm / kl (symlog)")
    ax2.legend(loc="best")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"Wrote {out}")
    print(f"Final reward (mean): {rm[-1]:.4f}")
    print(f"Final loss: {loss[-1]:.2e}")
    print(f"Final grad_norm: {grad_norm[-1]:.4f}")
    print(f"Final KL: {kl[-1]:.4f}")


if __name__ == "__main__":
    main()
