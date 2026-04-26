# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Interactive Gradio demo: "Play as the on-call engineer."

A human operator drives the DevOps Pipeline environment by hand. Mirrors what
the trained policy sees: same observations, same role gating, same reward
signal. Connects to the FastAPI env via ``DevopsPipelineEnv`` (sticky-session
sync client) so judges can compare their own intuition against the +1.156
delta our trained adapter achieves.

Launch:
    python gradio_app.py                    # connects to http://localhost:8000
    ENV_URL=https://...hf.space python gradio_app.py    # remote env
    GRADIO_SERVER_PORT=7860 python gradio_app.py        # custom port

HF Space deployment recommendation
----------------------------------
Two clean options; we recommend (B) for the hackathon:

(A) Single Space, both servers — add a tiny ``start.sh`` that runs
    ``uvicorn server.app:app --port 8000 &`` then
    ``python gradio_app.py --server-port 7860``, expose 7860 in the Dockerfile.
    Pros: one URL. Cons: bloats the env-server image with gradio (~40 MB) and
    risks judges scoring the env image instead of the demo image.

(B) Two Spaces — keep ``yashash045/devops-pipeline-gym`` as the pure env
    (graded by the hackathon harness, no Gradio noise), spin a separate
    ``yashash045/devops-pipeline-demo`` Space whose Dockerfile installs only
    ``gradio + httpx`` and runs ``python gradio_app.py`` with
    ``ENV_URL`` pointed at the env Space. Pros: zero risk to the graded
    artifact, demo can be redeployed independently. Cons: two URLs.

This file is the entry point for both options — the code is identical.
"""

from __future__ import annotations

import os
import threading
import traceback
from typing import Any, Dict, List, Optional

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from devops_pipeline_gym.client import DevopsPipelineEnv
from devops_pipeline_gym.models import (
    ActionType,
    ConfigEdit,
    MigrationType,
    PipelineAction,
    Role,
)

DEFAULT_ENV_URL = os.environ.get("ENV_URL", "http://localhost:8000")
SERVICES = ["database-primary", "auth-service", "api-gateway", "cache-service", "web-frontend"]
TASKS = ["clean_deploy", "broken_pipeline", "judgment_call",
         "cascading_failure", "capacity_crisis", "random_incident"]
MIGRATIONS = ["001_init_schema", "002_add_indexes", "003_backfill_users"]
ROLE_COLOR = {Role.DEV: "#3b82f6", Role.SRE: "#10b981", Role.OPS: "#f59e0b"}


# --- Session ---------------------------------------------------------------

class Session:
    """Per-user wrapper around a sticky DevopsPipelineEnv client.

    Held in ``gr.State`` so concurrent demo users don't share an episode.
    The underlying server is single-tenant (``SUPPORTS_CONCURRENT_SESSIONS = False``)
    but each Session opens its own HTTP client.
    """

    def __init__(self, env_url: str):
        self._lock = threading.Lock()
        self.env_url = env_url
        self._cm = DevopsPipelineEnv(base_url=env_url).sync()
        self.client = self._cm.__enter__()
        self.obs = None
        self.task = ""
        self.rewards: List[float] = []
        self.log: List[str] = []
        self.done = False

    def reset(self, task: str) -> None:
        with self._lock:
            os.environ["DEVOPS_TASK"] = task
            result = self.client.reset()
            self.obs = result.observation
            self.task = task
            self.rewards = []
            self.log = [f"[RESET] task={task} role={self.obs.current_role.value}"]
            self.done = False

    def step(self, action: PipelineAction) -> None:
        with self._lock:
            result = self.client.step(action)
            self.obs = result.observation
            r = float(result.reward or 0.0)
            self.rewards.append(r)
            self.done = bool(result.done)
            err = self.obs.last_action_error or ""
            self.log.append(
                f"[STEP {self.obs.step_number}] {action.action_type.value}"
                + (f" svc={action.service_name}" if action.service_name else "")
                + (f" role={action.role.value}" if action.role else "")
                + f" reward={r:+.2f} done={str(self.done).lower()}"
                + (f" err={err}" if err else "")
            )

    def close(self) -> None:
        try:
            self._cm.__exit__(None, None, None)
        except Exception:
            pass


# --- Rendering helpers -----------------------------------------------------

def _services_table(obs) -> List[List[str]]:
    if obs is None:
        return [["—", "—", "—", "—", "—"]]
    rows = []
    for s in obs.services:
        masked = s.health.value == "unknown"
        rows.append([
            s.name,
            s.health.value,
            "?" if masked else f"{s.request_latency_ms:.0f} ms",
            "?" if masked else f"{s.error_rate:.2f} /s",
            s.current_version,
        ])
    return rows or [["(no services)", "", "", "", ""]]


def _reward_chart(rewards: List[float]):
    fig, ax = plt.subplots(figsize=(5, 3), dpi=110)
    if rewards:
        steps = list(range(1, len(rewards) + 1))
        cum = [sum(rewards[:i + 1]) for i in range(len(rewards))]
        ax.bar(steps, rewards, color="#94a3b8", label="step reward")
        ax.plot(steps, cum, color="#0ea5e9", marker="o", linewidth=2, label="cumulative")
        ax.axhline(0, color="#64748b", linewidth=0.5)
        ax.legend(loc="best", fontsize=8)
        ax.set_xlabel("step"); ax.set_ylabel("reward")
    else:
        ax.text(0.5, 0.5, "no steps yet", ha="center", va="center",
                transform=ax.transAxes, color="#94a3b8")
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    return fig


def _status_md(sess: Optional[Session]) -> str:
    if sess is None or sess.obs is None:
        return "_Click **Reset** to start an episode._"
    o = sess.obs
    cum = sum(sess.rewards)
    color = ROLE_COLOR.get(o.current_role, "#64748b")
    role_badge = (f"<span style='background:{color};color:white;padding:2px 8px;"
                  f"border-radius:4px;font-weight:600'>{o.current_role.value.upper()}</span>")
    return (
        f"**Task:** `{sess.task}`  ·  "
        f"**Step** {o.step_number} / {o.max_steps}  ·  "
        f"**Role to act:** {role_badge}  ·  "
        f"**Cumulative reward:** `{cum:+.3f}`  ·  "
        f"**Done:** `{str(sess.done).lower()}`\n\n"
        f"**Goal:** {o.goal}\n\n"
        f"**Summary:** {o.summary or '_no alerts_'}"
    )


# --- Action wiring ---------------------------------------------------------

def _safe_step(sess: Session, action: PipelineAction):
    try:
        sess.step(action)
    except Exception as e:
        sess.log.append(f"[ERROR] {type(e).__name__}: {e}")
        traceback.print_exc()


def _ensure_session(sess: Optional[Session], env_url: str) -> Session:
    if sess is None or sess.env_url != env_url:
        if sess is not None:
            sess.close()
        sess = Session(env_url)
    return sess


def _refresh(sess: Session):
    return (sess, _status_md(sess), _services_table(sess.obs),
            _reward_chart(sess.rewards), "\n".join(sess.log[-20:]))


def do_reset(sess, env_url, task):
    sess = _ensure_session(sess, env_url)
    sess.reset(task)
    return _refresh(sess)


def do_action(sess, action_type: str, role: Role, service: str,
              version: str, cfg_key: str, cfg_val: str,
              migration: str, mig_type: str):
    if sess is None or sess.obs is None:
        return _refresh(_ensure_session(sess, DEFAULT_ENV_URL))
    if sess.done:
        sess.log.append("[INFO] Episode is done — click Reset to play again.")
        return _refresh(sess)
    kwargs: Dict[str, Any] = {"action_type": ActionType(action_type), "role": role}
    if action_type in ("view_logs", "view_config", "edit_config", "deploy", "rollback"):
        kwargs["service_name"] = service or "api-gateway"
    if action_type == "deploy":
        kwargs["target_version"] = version or "v1.0.1"
    if action_type == "edit_config":
        kwargs["config_edits"] = [ConfigEdit(key=cfg_key or "database.pool_size",
                                             value=cfg_val or "20")]
    if action_type == "run_migration":
        kwargs["migration_name"] = migration or MIGRATIONS[0]
        kwargs["migration_type"] = MigrationType(mig_type or "schema")
    if action_type in ("approve", "abort"):
        kwargs["reason"] = f"human-driven {action_type}"
    _safe_step(sess, PipelineAction(**kwargs))
    return _refresh(sess)


# --- UI --------------------------------------------------------------------

def build_ui() -> gr.Blocks:
    with gr.Blocks(title="DevOps Pipeline Gym — Play as the Agent",
                   theme=gr.themes.Soft()) as demo:
        gr.Markdown("# DevOps Pipeline Gym — Play as the On-Call Engineer\n"
                    "Drive the same env our trained policy sees. "
                    "Investigate before acting. Mind your role.")
        sess_state = gr.State(value=None)

        with gr.Row():
            env_url = gr.Textbox(value=DEFAULT_ENV_URL, label="Env URL", scale=3)
            task_dd = gr.Dropdown(choices=TASKS, value="broken_pipeline",
                                  label="Task", scale=2)
            reset_btn = gr.Button("Reset", variant="primary", scale=1)

        status_md = gr.Markdown()

        with gr.Row():
            with gr.Column(scale=3):
                services_df = gr.Dataframe(
                    headers=["service", "health", "latency", "error_rate", "version"],
                    datatype=["str"] * 5, interactive=False, label="Service health")

                gr.Markdown("### Action panel — buttons enforce role gating server-side")
                svc_dd = gr.Dropdown(choices=SERVICES, value="api-gateway", label="service")
                with gr.Accordion("DEV actions (config, migrations)", open=True):
                    with gr.Row():
                        b_view_cfg = gr.Button("view_config")
                        b_edit_cfg = gr.Button("edit_config")
                        b_run_mig = gr.Button("run_migration")
                    with gr.Row():
                        cfg_key = gr.Textbox(value="database.pool_size", label="config key")
                        cfg_val = gr.Textbox(value="20", label="value")
                        mig_name = gr.Dropdown(choices=MIGRATIONS, value=MIGRATIONS[0],
                                               label="migration")
                        mig_type = gr.Dropdown(choices=["schema", "data", "rollback_migration"],
                                               value="schema", label="type")
                with gr.Accordion("SRE actions (observation)", open=True):
                    with gr.Row():
                        b_view_pipe = gr.Button("view_pipeline")
                        b_view_logs = gr.Button("view_logs")
                with gr.Accordion("OPS actions (deploy / rollback)", open=True):
                    with gr.Row():
                        version_tb = gr.Textbox(value="v1.0.1", label="target_version")
                        b_deploy = gr.Button("deploy", variant="secondary")
                        b_rollback = gr.Button("rollback", variant="secondary")

                gr.Markdown("### Terminal actions (end the episode)")
                with gr.Row():
                    b_approve = gr.Button("approve", variant="primary")
                    b_abort = gr.Button("abort", variant="stop")

            with gr.Column(scale=2):
                reward_plot = gr.Plot(label="Reward history")
                step_log = gr.Textbox(label="Step log (last 20)", lines=14, interactive=False)

        outs = [sess_state, status_md, services_df, reward_plot, step_log]

        reset_btn.click(do_reset, [sess_state, env_url, task_dd], outs)

        def _bind(btn, atype, role):
            btn.click(
                lambda s, sv, ver, ck, cv, mn, mt: do_action(
                    s, atype, role, sv, ver, ck, cv, mn, mt),
                [sess_state, svc_dd, version_tb, cfg_key, cfg_val, mig_name, mig_type],
                outs)

        _bind(b_view_pipe, "view_pipeline", Role.SRE)
        _bind(b_view_logs, "view_logs", Role.SRE)
        _bind(b_view_cfg, "view_config", Role.DEV)
        _bind(b_edit_cfg, "edit_config", Role.DEV)
        _bind(b_run_mig, "run_migration", Role.DEV)
        _bind(b_deploy, "deploy", Role.OPS)
        _bind(b_rollback, "rollback", Role.OPS)
        _bind(b_approve, "approve", Role.OPS)
        _bind(b_abort, "abort", Role.OPS)

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.queue(default_concurrency_limit=4).launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
        show_error=True,
    )
