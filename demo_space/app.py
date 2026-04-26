"""DevOps Pipeline Gym — Gradio "play as the on-call engineer" demo.

Standalone version: pure httpx + gradio, no openenv-core dependency,
to sidestep the websockets dep conflict between gradio-client (<13)
and openenv-core (>=15).

Calls the env Space directly via HTTP POST /reset and POST /step.
"""

from __future__ import annotations

import os
import traceback
from typing import Any, Dict, List, Optional

import gradio as gr
import httpx
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_ENV_URL = os.environ.get(
    "ENV_URL", "https://yashash045-devops-pipeline-gym.hf.space"
)

SERVICES = [
    "database-primary",
    "auth-service",
    "api-gateway",
    "cache-service",
    "web-frontend",
]
TASKS = [
    "clean_deploy",
    "broken_pipeline",
    "judgment_call",
    "cascading_failure",
    "capacity_crisis",
    "random_incident",
]
MIGRATIONS = ["001_init_schema", "002_add_indexes", "003_backfill_users"]


# --- Session ---------------------------------------------------------------

class Session:
    """Per-user session: holds an httpx client and the live observation."""

    def __init__(self, env_url: str = DEFAULT_ENV_URL):
        self.env_url = env_url.rstrip("/")
        self.client: Optional[httpx.Client] = None
        self.observation: Dict[str, Any] = {}
        self.rewards: List[float] = []
        self.step_log: List[str] = []

    def _ensure_client(self):
        if self.client is None:
            self.client = httpx.Client(base_url=self.env_url, timeout=60.0)

    def reset(self, task: str) -> Dict[str, Any]:
        self._ensure_client()
        body = {"task": task} if task else {}
        r = self.client.post("/reset", json=body)
        r.raise_for_status()
        data = r.json()
        self.observation = data.get("observation", data)
        self.rewards = []
        self.step_log = [f"[reset] task={task}"]
        return self.observation

    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        self._ensure_client()
        # Try wrapped form first; fall back to top-level if 422
        try:
            r = self.client.post("/step", json={"action": action})
            if r.status_code == 422:
                r = self.client.post("/step", json=action)
            r.raise_for_status()
        except httpx.HTTPStatusError as e:
            self.step_log.append(
                f"[error] {e.response.status_code}: {e.response.text[:200]}"
            )
            return {"observation": self.observation, "reward": 0.0, "done": False}

        data = r.json()
        self.observation = data.get("observation", self.observation)
        reward = float(data.get("reward", 0.0) or 0.0)
        done = bool(data.get("done", False))
        self.rewards.append(reward)
        atype = action.get("action_type", "?")
        role = action.get("role", "?")
        svc = action.get("service_name", "")
        suffix = f" svc={svc}" if svc else ""
        self.step_log.append(
            f"[step {len(self.rewards):2d}] {role:>3} {atype}{suffix}  reward={reward:+.3f}"
            + ("  DONE" if done else "")
        )
        if len(self.step_log) > 21:
            self.step_log = self.step_log[-21:]
        return data


# --- UI helpers ------------------------------------------------------------

def _services_to_rows(obs: Dict[str, Any]) -> List[List[Any]]:
    rows: List[List[Any]] = []
    for s in obs.get("services", []) or []:
        masked = s.get("health") in (None, "unknown", "Unknown")
        cpu = "?" if masked else f"{s.get('cpu_percent', 0):.0f}%"
        mem = "?" if masked else f"{s.get('memory_percent', 0):.0f}%"
        lat = "?" if masked else f"{s.get('request_latency_ms', 0):.0f}ms"
        err = "?" if masked else f"{s.get('error_rate', 0):.2f}/s"
        rows.append([
            s.get("name", "?"),
            s.get("health", "?"),
            s.get("current_version", "?"),
            cpu, mem, lat, err,
        ])
    return rows


def _reward_chart(rewards: List[float]) -> Any:
    fig, ax = plt.subplots(figsize=(6, 3))
    if rewards:
        steps = list(range(1, len(rewards) + 1))
        s = 0.0
        cumulative = []
        for r in rewards:
            s += r
            cumulative.append(s)
        ax.bar(steps, rewards,
               color=["#10b981" if r >= 0 else "#ef4444" for r in rewards],
               edgecolor="black", linewidth=0.3)
        ax.plot(steps, cumulative, color="#1e40af", lw=2, marker="o", label="cumulative")
        ax.axhline(0, color="black", lw=0.5)
        ax.set_xlabel("step")
        ax.set_ylabel("reward")
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_title(f"step rewards | total = {sum(rewards):+.3f}")
    else:
        ax.text(0.5, 0.5, "Click an action button below to start collecting rewards",
                ha="center", va="center", transform=ax.transAxes, color="#94a3b8",
                fontsize=10)
        ax.axis("off")
    fig.tight_layout()
    return fig


def _summary_text(obs: Dict[str, Any]) -> str:
    role = obs.get("current_role", "?")
    task = obs.get("task_description", "")
    goal = obs.get("goal", "")
    last_err = obs.get("last_action_error") or ""
    last_res = obs.get("last_action_result") or ""
    parts = [f"**Current role:** `{role}`"]
    if task:
        parts.append(f"**Task:** {task}")
    if goal:
        parts.append(f"**Goal:** {goal}")
    if last_res:
        parts.append(f"**Last action:** {last_res}")
    if last_err:
        parts.append(f"**Error:** ⚠️ {last_err}")
    return "\n\n".join(parts)


# --- UI builder ------------------------------------------------------------

def build_ui():
    with gr.Blocks(title="DevOps Pipeline Demo",
                   theme=gr.themes.Soft(primary_hue="blue")) as demo:
        gr.Markdown(
            "# 🛠️ DevOps Pipeline Gym — Play as the On-Call Engineer\n"
            "*Five microservices. Three role hats. One rule: don't make it worse.* "
            "Same env our trained Qwen3-1.7B agent operates in.\n\n"
            "**Env Space:** [yashash045/devops-pipeline-gym](https://huggingface.co/spaces/yashash045/devops-pipeline-gym) · "
            "**Trained adapter:** [yashash045/devops-pipeline-gym-sft-adapter](https://huggingface.co/yashash045/devops-pipeline-gym-sft-adapter) · "
            "**BLOG:** [BLOG.md](https://huggingface.co/spaces/yashash045/devops-pipeline-gym/blob/main/BLOG.md)"
        )

        sess_state = gr.State(value=Session())

        with gr.Row():
            env_url = gr.Textbox(
                label="Env URL",
                value=DEFAULT_ENV_URL,
                scale=3,
                info="Connects to the env Space via HTTP",
            )
            task_dd = gr.Dropdown(
                choices=TASKS,
                value="clean_deploy",
                label="Task",
                scale=2,
            )
            reset_btn = gr.Button("🔄 Reset", variant="primary", scale=1)

        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### Services (current view)")
                services_df = gr.Dataframe(
                    headers=["name", "health", "version", "cpu", "memory", "latency", "errors"],
                    datatype=["str"] * 7,
                    value=[],
                    row_count=(5, "fixed"),
                    col_count=(7, "fixed"),
                    interactive=False,
                    wrap=True,
                )
                summary_md = gr.Markdown("Click **Reset** to start an episode.")
            with gr.Column(scale=1):
                gr.Markdown("### Rewards")
                reward_plot = gr.Plot(value=_reward_chart([]))
                step_log = gr.Textbox(
                    label="Step log (latest 20)",
                    value="",
                    lines=10,
                    max_lines=20,
                    interactive=False,
                )

        gr.Markdown("### Action panel")
        gr.Markdown(
            "Buttons are organised by role (DEV / SRE / OPS). "
            "Acting outside the **current_role** above costs `-0.15` and the action is dropped — try it and see."
        )

        with gr.Row():
            svc_dd = gr.Dropdown(choices=SERVICES, value="auth-service", label="Target service")
            version_tb = gr.Textbox(label="Target version (deploy)", value="v2.0")
            cfg_key = gr.Textbox(label="Config key (edit_config)", value="max_connections")
            cfg_val = gr.Textbox(label="Config value (edit_config)", value="100")
            mig_name = gr.Textbox(label="Migration name", value="001_init_schema")
            mig_type = gr.Dropdown(choices=["schema", "data", "rollback"],
                                    value="schema", label="Migration type")

        with gr.Accordion("🔵 DEV actions", open=True):
            with gr.Row():
                b_view_cfg = gr.Button("view_config")
                b_edit_cfg = gr.Button("edit_config")
                b_run_mig = gr.Button("run_migration")

        with gr.Accordion("🟢 SRE actions", open=True):
            with gr.Row():
                b_view_pipe = gr.Button("view_pipeline")
                b_view_logs = gr.Button("view_logs")

        with gr.Accordion("🟠 OPS actions", open=True):
            with gr.Row():
                b_deploy = gr.Button("deploy", variant="primary")
                b_rollback = gr.Button("rollback")
            with gr.Row():
                b_approve = gr.Button("✅ approve (terminal)", variant="primary")
                b_abort = gr.Button("❌ abort (terminal)", variant="stop")

        outs = [services_df, reward_plot, step_log, summary_md]

        def do_reset(sess: Session, env_url_val: str, task: str):
            try:
                if sess is None or sess.env_url != env_url_val.rstrip("/"):
                    sess = Session(env_url_val)
                obs = sess.reset(task)
                return (
                    _services_to_rows(obs),
                    _reward_chart(sess.rewards),
                    "\n".join(sess.step_log),
                    _summary_text(obs),
                )
            except Exception as e:
                tb = traceback.format_exc()
                return (
                    [],
                    _reward_chart([]),
                    f"[reset error] {type(e).__name__}: {e}",
                    f"❌ Reset failed: {type(e).__name__}\n```\n{tb[-500:]}\n```",
                )

        def do_action(sess: Session, action_type: str, role: str,
                      service_name: str, target_version: str,
                      cfg_k: str, cfg_v: str,
                      mig_n: str, mig_t: str):
            if sess is None or sess.client is None:
                return (
                    [],
                    _reward_chart([]),
                    "[error] click Reset first",
                    "❌ Click **Reset** to start an episode before taking actions.",
                )
            action: Dict[str, Any] = {"action_type": action_type, "role": role}
            if action_type in ("deploy", "rollback", "view_logs", "view_config", "edit_config"):
                action["service_name"] = service_name
            if action_type == "deploy":
                action["target_version"] = target_version
            if action_type == "edit_config":
                action["config_edits"] = [{"key": cfg_k, "value": cfg_v}]
            if action_type == "run_migration":
                action["migration_name"] = mig_n
                action["migration_type"] = mig_t

            try:
                sess.step(action)
                return (
                    _services_to_rows(sess.observation),
                    _reward_chart(sess.rewards),
                    "\n".join(sess.step_log),
                    _summary_text(sess.observation),
                )
            except Exception as e:
                tb = traceback.format_exc()
                sess.step_log.append(f"[error] {type(e).__name__}: {str(e)[:120]}")
                return (
                    _services_to_rows(sess.observation),
                    _reward_chart(sess.rewards),
                    "\n".join(sess.step_log),
                    f"❌ Action failed: {type(e).__name__}\n```\n{tb[-500:]}\n```",
                )

        reset_btn.click(do_reset, [sess_state, env_url, task_dd], outs)

        def _bind(btn, atype, role):
            btn.click(
                lambda s, sv, ver, ck, cv, mn, mt: do_action(
                    s, atype, role, sv, ver, ck, cv, mn, mt),
                [sess_state, svc_dd, version_tb, cfg_key, cfg_val, mig_name, mig_type],
                outs)

        _bind(b_view_pipe, "view_pipeline", "sre")
        _bind(b_view_logs, "view_logs", "sre")
        _bind(b_view_cfg, "view_config", "dev")
        _bind(b_edit_cfg, "edit_config", "dev")
        _bind(b_run_mig, "run_migration", "dev")
        _bind(b_deploy, "deploy", "ops")
        _bind(b_rollback, "rollback", "ops")
        _bind(b_approve, "approve", "ops")
        _bind(b_abort, "abort", "ops")

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.queue(default_concurrency_limit=4).launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
        show_error=True,
    )
