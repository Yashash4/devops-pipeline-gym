---
title: DevOps Pipeline Demo
emoji: 🛠️
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "5.7.1"
python_version: "3.11"
app_file: app.py
pinned: false
license: apache-2.0
short_description: Play as the on-call engineer in our incident env
---

# DevOps Pipeline Demo

You are the on-call engineer. Five microservices sit in front of you across 6 tasks, with 9 actions split across 3 roles (DEV, SRE, OPS) that rotate each step. Pick a task, click **Reset**, then click action buttons to clear the incident or watch yourself break things. This is the same env our trained Qwen3-1.7B + SFT agent operates in. The agent scored -0.044 on `judgment_call` against -1.200 for the untrained Qwen2.5-7B baseline (delta +1.156). Every action you take hits the same FastAPI server and the same deterministic reward grader the agent saw during training.

## Links

- **Env (graded artifact):** [yashash045/devops-pipeline-gym](https://huggingface.co/spaces/yashash045/devops-pipeline-gym)
- **Trained adapter:** [yashash045/devops-pipeline-gym-sft-adapter](https://huggingface.co/yashash045/devops-pipeline-gym-sft-adapter)
- **Code:** [Yashash4/devops-pipeline-gym](https://github.com/Yashash4/devops-pipeline-gym)
- **Blog:** [BLOG.md](https://huggingface.co/spaces/yashash045/devops-pipeline-gym/blob/main/BLOG.md)

Built for the Meta PyTorch OpenEnv Hackathon Grand Finale 2026. Team Tripod (Yashash, Gajanand, Likith).
