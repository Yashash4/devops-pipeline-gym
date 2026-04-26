"""DevOps Pipeline Gym — Gradio "play as the on-call engineer" demo.

WebSocket version: talks to the env Space via openenv's WebSocket protocol
(JSON messages of shape {"type": "reset|step", "data": {...}}). Uses the
sync API of the `websockets` package so we don't need to invade Gradio's
event loop.
"""

from __future__ import annotations

import json
import os
import traceback
from typing import Any, Dict, List, Optional

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# websockets sync client (works with gradio-client 2.x; we pin both in requirements.txt)
from websockets.sync.client import connect as ws_connect

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


def _to_ws_url(env_url: str) -> str:
    """Convert https://...hf.space → wss://...hf.space/ws."""
    url = env_url.rstrip("/")
    if url.startswith("https://"):
        url = "wss://" + url[len("https://"):]
    elif url.startswith("http://"):
        url = "ws://" + url[len("http://"):]
    return url + "/ws"


# --- Session ---------------------------------------------------------------

class Session:
    """Per-user WebSocket session to the env Space.

    openenv-core uses WebSocket as its sticky-session protocol, so plain
    HTTP /step returns 500 on the env Space (no episode state across HTTP
    requests). We open a long-lived WebSocket here, one per user, kept in
    gr.State so it survives across button clicks.
    """

    def __init__(self, env_url: str = DEFAULT_ENV_URL):
        self.env_url = env_url.rstrip("/")
        self.ws = None  # websockets.sync.client.ClientConnection
        self.observation: Dict[str, Any] = {}
        self.rewards: List[float] = []
        self.step_log: List[str] = []

    def _ensure_ws(self):
        if self.ws is None:
            self.ws = ws_connect(
                _to_ws_url(self.env_url),
                max_size=100 * 1024 * 1024,
                open_timeout=20,
            )

    def _send_recv(self, message: Dict[str, Any]) -> Dict[str, Any]:
        self._ensure_ws()
        self.ws.send(json.dumps(message))
        raw = self.ws.recv(timeout=60)
        return json.loads(raw)

    def reset(self, task: str) -> Dict[str, Any]:
        # openenv reset: kwargs become reset() params. We pass task so env can
        # use it (server-side reset reads DEVOPS_TASK env var, which we can't
        # set from here, but newer openenv builds honor data.task too).
        msg = {"type": "reset", "data": {"task": task} if task else {}}
        resp = self._send_recv(msg)
        if resp.get("type") == "error":
            err = resp.get("data", {}).get("message", "unknown")
            raise RuntimeError(f"env error on reset: {err}")
        data = resp.get("data", {})
        self.observation = data.get("observation", {})
        self.rewards = []
        self.step_log = [f"[reset] task={task}"]
        return self.observation

    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        msg = {"type": "step", "data": action}
        try:
            resp = self._send_recv(msg)
        except Exception as e:
            self.step_log.append(f"[ws error] {type(e).__name__}: {str(e)[:120]}")
            # Reconnect on any error
            try:
                self.ws.close()
            except Exception:
                pass
            self.ws = None
            return {"observation": self.observation, "reward": 0.0, "done": False}

        if resp.get("type") == "error":
            err = resp.get("data", {}).get("message", "unknown")
            self.step_log.append(f"[env error] {err[:120]}")
            return {"observation": self.observation, "reward": 0.0, "done": False}

        data = resp.get("data", {})
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
            "# 🛠️ DevOps Pipeline Gym. Play as the On-Call Engineer.\n"
            "Five microservices. Three roles. One rule: do not make it worse. "
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
                info="Connects to the env Space via WebSocket",
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
            "Acting outside the **current_role** above costs `-0.15` and the action is dropped. Try it and see."
        )

        with gr.Row():
            svc_dd = gr.Dropdown(choices=SERVICES, value="auth-service", label="Target service")
            version_tb = gr.Textbox(label="Target version (deploy)", value="v2.0")
            cfg_key = gr.Textbox(label="Config key (edit_config)", value="max_connections")
            cfg_val = gr.Textbox(label="Config value (edit_config)", value="100")
            mig_name = gr.Textbox(label="Migration name", value="001_init_schema")
            mig_type = gr.Dropdown(choices=["schema", "data", "rollback_migration"],
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
                    if sess and sess.ws:
                        try: sess.ws.close()
                        except: pass
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
                    f"❌ Reset failed: {type(e).__name__}\n```\n{tb[-400:]}\n```",
                )

        def do_action(sess: Session, action_type: str, role: str,
                      service_name: str, target_version: str,
                      cfg_k: str, cfg_v: str,
                      mig_n: str, mig_t: str):
            if sess is None or sess.ws is None:
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
                    f"❌ Action failed: {type(e).__name__}\n```\n{tb[-400:]}\n```",
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
