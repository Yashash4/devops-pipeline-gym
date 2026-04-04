# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FastAPI application for the DevOps Pipeline Environment."""

from openenv.core.env_server.http_server import create_app

from devops_pipeline_env.models import PipelineAction, PipelineObservation
from server.pipeline_environment import PipelineEnvironment

app = create_app(
    PipelineEnvironment,
    PipelineAction,
    PipelineObservation,
    env_name="devops_pipeline_env",
    max_concurrent_envs=1,
)


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
        ],
        "action_schema": PipelineAction.model_json_schema(),
    }


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/baseline")
async def run_baseline():
    """Return pre-recorded baseline scores. Does NOT run inference.py."""
    return {
        "scores": {
            "clean_deploy": 0.585,
            "broken_pipeline": 0.482,
            "judgment_call": 0.184,
            "cascading_failure": 0.280,
        },
        "model": "Qwen/Qwen2.5-72B-Instruct",
        "note": "Recorded from inference.py run on 2026-04-04.",
    }


@app.post("/grader")
async def run_grader(task_name: str = "clean_deploy"):
    """Score from active session's episode history."""
    from server.graders import grade_task as _grade_task

    env = PipelineEnvironment._last_instance
    if env is None or env.get_engine() is None:
        return {"task": task_name, "score": 0.0, "error": "No active session. Call /reset first."}
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
