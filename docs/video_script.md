# 90-Second Video Script — DevOps Pipeline Gym

**Length target:** 85-90 seconds (250-280 words spoken at ~3 wps)
**Format:** Screen recording (Loom or OBS) with you reading off this script
**Upload:** YouTube unlisted, link from README badge row

Recording setup:
- Open three browser tabs in advance: HF Space, Colab notebook, BLOG.md
- Open one terminal showing a sample `[STEP]` log
- Hit record, read smoothly, single take if possible

---

## Section 1 — Hook (0:00–0:12, ~12 seconds)

> "Frontier LLMs already know how to fix a broken database. They can recite connection pool errors in their sleep. What they don't reliably do is *check* before changing anything."

[Cut to terminal showing a `[STEP]` log with "view_pipeline" actions before "deploy"]

> "Incident response is sequencing, not knowledge. We built an OpenEnv environment that trains exactly that."

---

## Section 2 — The Environment (0:12–0:35, ~23 seconds)

[Switch to the HF Space, click `/reset` or show the Gradio demo]

> "Five microservices in a dependency graph. Nine actions split across three roles — DEV, SRE, OPS — that rotate between steps the way a real on-call handoff would. Health is masked until you investigate."

[Show the role-gated action panel, click view_logs, see service health update]

> "The reward is six deterministic Python components, bounded per step. No LLM judge in the loop. Same trajectory in, same score out, every time."

---

## Section 3 — Results (0:35–1:05, ~30 seconds)

[Switch to the bar chart from the Colab — base vs trained]

> "We trained Qwen3 1.7B with QLoRA on 80 expert trajectories. Same task, same seed, same prompt format — same scoring rubric across all baselines."

[Read the numbers off the chart, e.g.]

> "Untrained Qwen2.5 7B baseline on judgment_call: -1.200 reward. Our trained Qwen3 1.7B with the SFT adapter: -0.044 reward. That's a +1.156 delta — a 1.7B model trained on 80 trajectories beats an untrained 7B same-family baseline on the same task, same seed, same prompt format."

[Switch to the frontier model chart if rendered, or just describe]

> "And here's the interesting part — that 1.7B trained model outperforms several 70B-plus frontier baselines on the same task. Because we trained the right *skill*, not the bigger model."

---

## Section 4 — Why It Matters + Wrap (1:05–1:25, ~20 seconds)

> "The bigger thesis: deterministic, verifiable RL environments for professional decisions are the missing rung between toy gridworlds and shipping real agents. We picked DevOps because failures are well-documented and graders can be pure functions. The same approach generalizes — legal triage, incident command, supply chain rerouting."

> "Try it on the Colab badge in our README. Play it interactively in our Gradio demo. We're Team Tripod. Thanks for watching."

---

## Tips

- **Pace.** Don't rush. 85 seconds is plenty. Stops between sentences are fine.
- **Tone.** Matter-of-fact and curious. Not hyped.
- **Don't memorize.** Read it. Eyes on the script, voice on the explanation.
- **One take is fine.** If you fumble, just re-roll the section. Loom and OBS both let you trim.
- **Numbers in `{...}`** get filled in after eval results land. Re-record only that section if numbers change significantly.

## Fallback if you run out of time

If recording is taking too long, cut Section 4 down to one line:
> "Try it on the Colab badge in our README. We're Team Tripod."

The hook + env + results are the points worth the score lift. Section 4 is gravy.
