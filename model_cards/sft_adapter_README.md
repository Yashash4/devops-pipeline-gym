---
base_model: unsloth/Qwen3-1.7B-bnb-4bit
library_name: peft
tags:
  - trl
  - sft
  - qlora
  - openenv
  - devops
  - incident-response
  - reinforcement-learning
license: apache-2.0
---

# DevOps Pipeline Gym — SFT Adapter

*A 1.7B model that learned to investigate before acting on production incidents.*

[![Live Env Space](https://img.shields.io/badge/%F0%9F%A4%97%20Env%20Space-devops--pipeline--gym-blue)](https://huggingface.co/spaces/yashash045/devops-pipeline-gym)
[![Code](https://img.shields.io/badge/GitHub-Yashash4%2Fdevops--pipeline--gym-black)](https://github.com/Yashash4/devops-pipeline-gym)
[![Blog](https://img.shields.io/badge/Read-BLOG.md-orange)](https://huggingface.co/spaces/yashash045/devops-pipeline-gym/blob/main/BLOG.md)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Yashash4/devops-pipeline-gym/blob/main/devops_pipeline_gym_colab.ipynb)

## What this is

A QLoRA adapter trained on top of `unsloth/Qwen3-1.7B-bnb-4bit` to act as an autonomous on-call engineer inside the [DevOps Pipeline Gym](https://huggingface.co/spaces/yashash045/devops-pipeline-gym) OpenEnv environment.

The environment simulates production incident response across 5 microservices in a dependency graph. The agent rotates between three roles — DEV, SRE, OPS — and must investigate before acting, identify root cause through cascading symptoms, and choose among multiple valid recovery paths. Every reward component is deterministic Python — no LLM judge in the loop, no API calls to anyone.

## What it learned

Trained on **80 expert trajectories** for 2 epochs on a free Kaggle T4 (~30 minutes wall clock). 17.4M trainable parameters (1.69% of base). QLoRA configuration: `r=16, alpha=32, dropout=0.05`, applied to all attention + MLP modules.

The hero result: **a 1.7B model trained on 80 trajectories outperforms 70B-700B frontier models** on the same `judgment_call` task. Same task, same seed family, same prompt format, same scoring rubric. Frontier baselines hit through HF Inference Router (n=3 seeds averaged for frontier, single-seed for our trained model and the 7B notebook baseline):

| Model | Size | Reward on `judgment_call` | Δ ours beats |
|---|---|---:|---:|
| Llama-3.3-70B-Instruct (untrained) | 70B | -1.815 | **+1.771** |
| DeepSeek-V3.1 (untrained) | 671B MoE | -1.580 | **+1.536** |
| Mistral-Large-Instruct-2411 (untrained) | 123B | -1.580 | **+1.536** |
| Qwen2.5-72B-Instruct (untrained) | 72B | -1.232 | **+1.188** |
| GPT-OSS-120B (untrained) | 120B MoE | -1.201 | **+1.157** |
| Qwen2.5-7B-Instruct (untrained, baseline in notebook) | 7B | -1.200 | **+1.156** |
| **Qwen3-1.7B + this SFT adapter (TRAINED)** | **1.7B** | **-0.044** | — |

A 1.7B model trained on 80 expert trajectories beats every untrained model we tested — from a 7B same-family Qwen baseline to the 671B DeepSeek-V3.1 — by **+1.16 to +1.77 reward** on this task. We did not run untrained Qwen3-1.7B as a same-family baseline within budget; the 7B Qwen2.5 row is the closest-size untrained model the demo notebook actually invokes via HF Router.

Frontier models default to either immediate `abort` (DeepSeek, Mistral all return -1.580 across all tasks) or attempted-but-failed action sequences. None succeed at the task without env-specific training. The trained 1.7B knows to investigate first, identify root cause, deploy carefully, and approve only when healthy.

## Quick start

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

base = "unsloth/Qwen3-1.7B-bnb-4bit"
tokenizer = AutoTokenizer.from_pretrained(base)
model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(model, "yashash045/devops-pipeline-gym-sft-adapter", subfolder="final")

# Then drive it through the env at https://yashash045-devops-pipeline-gym.hf.space
# See devops_pipeline_gym_colab.ipynb for an end-to-end runnable example.
```

## Reproduce on free Kaggle T4 (~15 min)

The full eval pipeline runs on free Kaggle T4. See `scripts/kaggle_cell_acc1.py` in the [code repo](https://github.com/Yashash4/devops-pipeline-gym) — it's a single-cell paste that boots the env, downloads this adapter, runs multi-seed eval, and uploads results back to this repo.

## Training stack

- **Framework:** TRL ([`SFTTrainer`](https://huggingface.co/docs/trl/main/en/sft_trainer)) + PEFT (QLoRA) + bitsandbytes (4-bit NF4)
- **Trajectories:** 80 expert demonstrations (chat-template format)
- **Hardware:** Kaggle T4 16GB, free tier
- **Cost:** $0
- **Wall time:** ~30 minutes

## What's the larger work

This adapter is the **trained policy** for the [DevOps Pipeline Gym](https://huggingface.co/spaces/yashash045/devops-pipeline-gym) — an OpenEnv environment built for the Meta PyTorch OpenEnv Hackathon (India 2026). The full submission includes:

- The env (deterministic, no-LLM-judge, role-rotated single policy)
- This SFT adapter (the trained policy)
- A [GRPO refinement adapter](https://huggingface.co/yashash045/devops-pipeline-gym-trained) (RL pipeline proof)
- Frontier baseline comparisons (5 models tested)
- Interactive Colab demo + Gradio "play as the on-call engineer" interface
- A [narrative writeup (BLOG.md)](https://huggingface.co/spaces/yashash045/devops-pipeline-gym/blob/main/BLOG.md)

## Citations

```bibtex
@misc{vonwerra2022trl,
    title        = {{TRL: Transformer Reinforcement Learning}},
    author       = {Leandro von Werra and Younes Belkada and Lewis Tunstall and Edward Beeching and Tristan Thrush and Nathan Lambert and Shengyi Huang and Kashif Rasul and Quentin Gallouédec},
    year         = 2020,
    journal      = {GitHub repository},
    publisher    = {GitHub},
    howpublished = {\url{https://github.com/huggingface/trl}}
}
```

(The TRL citation above is for the open-source library we used to train this adapter — Hugging Face's `trl` package by von Werra et al. Standard academic credit for the training framework.)
