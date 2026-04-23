# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FastAPI application for the DevOps Pipeline Environment."""

from openenv.core.env_server.http_server import create_app

from devops_pipeline_gym.models import PipelineAction, PipelineObservation
from server.pipeline_environment import PipelineEnvironment

app = create_app(
    PipelineEnvironment,
    PipelineAction,
    PipelineObservation,
    env_name="devops_pipeline_gym",
    max_concurrent_envs=1,
)

# Store active env on app.state so /grader can access it without class singletons.
# PipelineEnvironment.reset() calls _register_callback if set.
app.state.active_env = None
PipelineEnvironment._register_callback = lambda env: setattr(app.state, "active_env", env)


# --- Additional Required Endpoints -------------------------------------------

@app.get("/tasks")
def get_tasks():
    """Returns list of tasks and the action schema."""
    return {
        "tasks": [
            {
                "name": "clean_deploy",
                "difficulty": "easy",
                "description": "Deploy 2 services with all tests passing. No complications.",
                "max_steps": 15,
            },
            {
                "name": "broken_pipeline",
                "difficulty": "medium",
                "description": "Diagnose test failures, fix config errors, run migrations.",
                "max_steps": 20,
            },
            {
                "name": "judgment_call",
                "difficulty": "hard",
                "description": "Production incident with cascading failures. Hotfix breaks downstream service. 12-step time limit with degrading health.",
                "max_steps": 12,
            },
            {
                "name": "cascading_failure",
                "difficulty": "medium-hard",
                "description": "Root cause analysis across dependency chain. cache-service down, dragging api-gateway and web-frontend. Fix root cause first.",
                "max_steps": 15,
            },
            {
                "name": "capacity_crisis",
                "difficulty": "medium-hard",
                "description": "Peak traffic 4x normal. database-primary connection pool nearly full. Stabilize before tipping points trigger cascading collapse.",
                "max_steps": 15,
            },
            {
                "name": "random_incident",
                "difficulty": "variable",
                "description": "Procedurally generated incident. Service, failure type, and severity are randomized from seed. Infinite variation for curriculum learning.",
                "max_steps": 15,
            },
        ],
        "action_schema": PipelineAction.model_json_schema(),
    }


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/baseline")
async def run_baseline():
    """Return pre-recorded baseline scores. Does NOT run inference.py."""
    return {
        "scores": {
            "clean_deploy": 0.700,
            "broken_pipeline": 0.482,
            "judgment_call": 0.184,
            "cascading_failure": 0.280,
            "capacity_crisis": 0.250,
            "random_incident": 0.350,
        },
        "model": "Qwen/Qwen2.5-72B-Instruct",
        "note": "Baselines re-calibrated after environment tuning for clean_deploy (v2). Recorded 2026-04-08.",
    }


@app.post("/grader")
async def run_grader(task_name: str = ""):
    """Score from active session's episode history."""
    from server.graders import grade_task as _grade_task

    env = getattr(app.state, "active_env", None)
    if env is None or env.get_engine() is None:
        return {"task": task_name, "score": 0.001, "error": "No active session. Call /reset first."}
    if not env.get_episode_history():
        return {"task": env.get_task_name(), "score": 0.001, "error": "No steps taken. Call /step first."}
    active_task = env.get_task_name()
    if task_name and task_name != active_task:
        return {"task": task_name, "score": 0.001, "error": f"Task mismatch: requested '{task_name}' but active task is '{active_task}'."}
    if not task_name:
        task_name = active_task
    score = _grade_task(
        env.get_task_name(),
        env.get_episode_history(),
        env.get_engine(),
    )
    return {"task": env.get_task_name(), "score": score}


def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
