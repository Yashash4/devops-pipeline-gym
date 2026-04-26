---
title: DevOps Pipeline Demo
emoji: 🛠️
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "4.44.0"
app_file: app.py
pinned: false
license: apache-2.0
short_description: Play as the on-call engineer in our incident env
---

# DevOps Pipeline Demo

*Click action buttons. Watch the simulator respond. See what the trained agent has to learn.*

You are the on-call engineer. Five microservices in front of you. Health is masked until you investigate. Each step is role-gated (DEV / SRE / OPS) and the role rotates. Try to clear the incident — or watch yourself break things.

This is the same env our trained Qwen3-1.7B + SFT agent operates in. Every action you take goes through the exact same FastAPI server, hits the exact same deterministic reward grader.

## Links

- **The env (graded artifact):** [yashash045/devops-pipeline-gym](https://huggingface.co/spaces/yashash045/devops-pipeline-gym)
- **The trained adapter:** [yashash045/devops-pipeline-gym-sft-adapter](https://huggingface.co/yashash045/devops-pipeline-gym-sft-adapter)
- **Code:** [Yashash4/devops-pipeline-gym](https://github.com/Yashash4/devops-pipeline-gym)
- **Blog:** [BLOG.md](https://huggingface.co/spaces/yashash045/devops-pipeline-gym/blob/main/BLOG.md)

## How to use

1. Pick a task from the dropdown (start with `clean_deploy` if you want easy mode, `judgment_call` for the hardest one)
2. Click **Reset** to spin up a fresh incident
3. Click action buttons grouped by role. Watch the services table + reward chart update live.
4. Try to reach a clean `approve` (terminal +2.0 reward) before max steps run out.

## What you'll notice

- Acting outside your current role costs `-0.15` and the action is dropped (silently). The Gradio UI lets you click any role's button so you can experience this teaching moment.
- Some actions look right but are wrong because root cause hides downstream. That's the env teaching `investigate before act`.
- Reward bookkeeping is live — same reward function the trained agent sees during RL.

Built for the Meta PyTorch OpenEnv Hackathon Grand Finale 2026. Team Tripod (Yashash, Gajanand, Likith).
