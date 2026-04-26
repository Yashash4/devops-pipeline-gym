# Teaching a 1.7B Model to Stop, Look, Then Act

We handed a small open model a deployment console with five microservices, three role hats, and one rule: don't make it worse. Then we trained it not on facts about Kubernetes, but on the *order* in which a human on-call would touch things. This post is about what that took, and what it didn't.

## The Problem

Frontier LLMs know the right words for incident response. Ask Qwen2.5-72B "the auth service is throwing 500s, what now?" and it will tell you about connection pools, migration locks, and circuit breakers — fluently. What it does not reliably do is *check* before changing anything. It will not notice that the database it's about to restart is the upstream cause of the auth symptom. It will not pick rollback over hotfix when the deploy window is closing. The gap between knowing and doing in incident response is sequencing — investigate before acting, identify root cause through cascading symptoms, choose among multiple valid recovery paths. That's a decision skill, not a knowledge skill, and it has to be trained.

## The Environment

Five microservices in a dependency graph: a primary database feeds an auth service, which feeds an API gateway, which feeds a web frontend. A cache service hangs off the database too. Nine actions are split across three roles — DEV edits configs and runs migrations, SRE inspects logs and pipelines, OPS deploys, rolls back, approves and aborts. Acting outside your role costs reward and the action is dropped on the floor; the role rotates between steps the way a real on-call handoff would.

Health is masked. Until you `view_logs` or `view_config` on a service, you cannot see CPU, latency, or error rate, and a degraded service shows up as `unknown`. You can deploy blind. We just charge you for it.

Six tasks ship: a clean deploy, a broken pipeline, a judgment call with three valid resolutions, a cascading failure where root cause hides behind symptoms, a capacity crisis, and a procedurally generated `random_incident` that draws from forty-plus seed combinations so the agent cannot memorize an answer. The reward is six deterministic Python components — health delta, deploy progress, broke-healthy penalty, sub-goal bonuses, investigation decay, and a single role-alignment signal. No LLM judge anywhere in the loop. Same trajectory in, same score out, every time.

## What We Trained

Two stages. Stage one — supervised fine-tuning on 80 expert trajectories, about thirty minutes on a T4 — is the result that moved the headline number. Stage two — GRPO refinement on an L40S — proved the RL pipeline runs end-to-end but added little reward signal at this compute scale (more on that below). Both stages used the same Qwen3-1.7B-bnb-4bit base with QLoRA r=16 / alpha=32 on all attention and MLP modules.

Trained gets `−0.044` on `judgment_call` at seed `5003`, same prompt format, same task, same scoring rubric. We compare against every untrained baseline we could route to: **Qwen2.5-7B-Instruct gets −1.200, Qwen2.5-72B-Instruct gets −1.232 (n=3 averaged), Llama-3.3-70B gets −1.696, DeepSeek-V3.1 (671B MoE) gets −1.580, Mistral-Large-Instruct gets −1.580, GPT-OSS-120B gets −1.201**. Our trained 1.7B beats every one of them by +1.16 to +1.77 reward — including base models 4× to 400× its size. We did not run untrained Qwen3-1.7B as a same-family baseline within budget; the 7B Qwen2.5 row is what the demo notebook actually invokes via HF Router and is the closest-size untrained model we tested. The adapter is at [yashash045/devops-pipeline-gym-sft-adapter](https://huggingface.co/yashash045/devops-pipeline-gym-sft-adapter).

## What Surprised Us

GRPO did not compound on top of SFT inside our compute budget. The training signal is real — final loss landed around 6e-6, KL stayed bounded (~0.0006), grad_norm stays alive (0.0004 to 0.59), the trainer ran cleanly — but mean reward held near +0.04 and `clipped_ratio` stayed at 1.0, meaning every generation hit the completion-length cap rather than emitting a clean stop. Our read: the per-step reward is bounded to roughly ±0.32, most of the policy improvement is concentrated in the terminal +2.0 for a clean `approve`, and over a 12-step horizon too few rollouts touch that terminal bonus to differentiate the group. The gradient is starved, not noisy. The fix is more sampling, denser shaped reward, or a `<EOS>` token in the SFT trajectories — not a better optimizer.

The other surprise: role rotation was harder for the model to internalize than the action semantics themselves. The model learned *what* deploy does long before it learned *when it's allowed* to do it.

## Why It Matters

Deterministic, verifiable RL environments for professional decision-making are the missing rung between toy gridworlds and shipping real agents. We picked DevOps because failures are well-documented, recoveries are well-defined, and graders can be pure functions. But the same approach — partial observability, role-gated actions, multiple valid paths, procedural variation, no judge LLM in the loop — generalizes to any domain where sequencing matters more than knowledge: legal triage, incident command, supply chain rerouting, clinical workup. If you can write the simulator and the grader as code, you can train the decision. Try it: [yashash045/devops-pipeline-gym](https://huggingface.co/spaces/yashash045/devops-pipeline-gym).
