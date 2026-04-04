"""Inference script for the DevOps Pipeline Environment."""

import asyncio
import json
import os
import textwrap

from openai import OpenAI

from devops_pipeline_env import DevopsPipelineEnv, PipelineAction
from devops_pipeline_env.models import ActionType

# --- Env Vars (EXACT hackathon requirements) ----------------------------------
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
if not API_KEY:
    raise ValueError("HF_TOKEN or API_KEY environment variable is required")

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
IMAGE_NAME = os.getenv("IMAGE_NAME")

BENCHMARK = "devops_pipeline"
TASKS = ["clean_deploy", "broken_pipeline", "judgment_call", "cascading_failure"]
MAX_STEPS_PER_TASK = {"clean_deploy": 15, "broken_pipeline": 20, "judgment_call": 12, "cascading_failure": 15}
MAX_TOTAL_REWARD = {"clean_deploy": 0.70, "broken_pipeline": 0.85, "judgment_call": 0.65, "cascading_failure": 0.80}
TEMPERATURE = 0.7
MAX_TOKENS = 300
SUCCESS_SCORE_THRESHOLD = 0.1


# --- Log Functions (EXACT hackathon format) -----------------------------------
def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step, action, reward, done, error):
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success, steps, score, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# --- System Prompt ------------------------------------------------------------
SYSTEM_PROMPT = textwrap.dedent("""
You are a DevOps engineer managing a CI/CD deployment pipeline.
You receive observations as JSON containing:
- task_description: what you need to accomplish
- goal: specific success criteria
- services: current state of each service
- pipeline: pipeline stage and test results
- available_actions: what action_types you can use right now
- last_action_result: what happened after your last action

STRATEGY: Start by investigating degraded services with view_logs before taking action. Check configs with view_config if logs suggest misconfiguration. Fix root causes before deploying.

You must respond with a SINGLE valid JSON object matching the PipelineAction schema.

Example responses:
{"action_type": "view_pipeline"}
{"action_type": "view_logs", "service_name": "api-gateway"}
{"action_type": "deploy", "service_name": "api-gateway", "target_version": "v2.3.1"}
{"action_type": "edit_config", "service_name": "cache-service", "config_edits": [{"key": "redis.host", "value": "redis-prod.internal:6379"}]}
{"action_type": "approve", "reason": "All services deployed and healthy"}

Respond with ONLY the JSON object. No explanation, no markdown.
""").strip()

RETRY_PROMPT = 'Respond with ONLY a JSON action. Example: {"action_type": "view_pipeline"}'


def build_user_message(obs, investigated):
    """Build user message for the current step's observation."""
    current_obs = json.dumps(obs.model_dump(), indent=2, default=str)

    inv_block = ""
    if investigated:
        inv_block = "\n\nINVESTIGATED: " + ", ".join(sorted(investigated))

    return f"CURRENT STATE:\n{current_obs}{inv_block}\n\nWhat is your next action?"


def build_messages(system_prompt, conversation, current_user_msg):
    """Build multi-turn messages list with system prompt + last 6 turns + current."""
    messages = [{"role": "system", "content": system_prompt}]
    # Keep last 6 turns (12 messages = 6 user + 6 assistant)
    recent = conversation[-(6 * 2):]
    messages.extend(recent)
    messages.append({"role": "user", "content": current_user_msg})
    return messages


def parse_llm_action(text):
    """Parse LLM response into PipelineAction. Fallback to view_pipeline on failure."""
    try:
        text = text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        data = json.loads(text)
        return PipelineAction(**data)
    except Exception:
        return PipelineAction(action_type=ActionType.VIEW_PIPELINE)


async def run_task(client, env, task_name):
    rewards = []
    steps_taken = 0
    score = 0.0
    success = False
    max_steps = MAX_STEPS_PER_TASK.get(task_name, 20)
    max_reward = MAX_TOTAL_REWARD.get(task_name, 1.0)
    conversation = []  # Multi-turn: list of {"role": ..., "content": ...}
    investigated = set()

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        os.environ["DEVOPS_TASK"] = task_name
        result = await env.reset()
        obs = result.observation

        for step in range(1, max_steps + 1):
            if result.done:
                break

            user_msg = build_user_message(obs, investigated)
            messages = build_messages(SYSTEM_PROMPT, conversation, user_msg)
            try:
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                    stream=False,
                )
                action_text = (completion.choices[0].message.content or "").strip()
                action = parse_llm_action(action_text)

                # Retry once if parse fell back to default
                if action.action_type == ActionType.VIEW_PIPELINE and "view_pipeline" not in action_text.lower():
                    retry_msgs = build_messages(RETRY_PROMPT, conversation, user_msg)
                    retry_completion = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=retry_msgs,
                        temperature=0.3,
                        max_tokens=150,
                        stream=False,
                    )
                    retry_text = (retry_completion.choices[0].message.content or "").strip()
                    retry_action = parse_llm_action(retry_text)
                    if retry_action.action_type != ActionType.VIEW_PIPELINE or "view_pipeline" in retry_text.lower():
                        action = retry_action
                        action_text = retry_text
            except Exception as e:
                print(f"[DEBUG] LLM call failed: {e}", flush=True)
                action = PipelineAction(action_type=ActionType.VIEW_PIPELINE)
                action_text = '{"action_type": "view_pipeline"}'

            # Track investigated services
            if action.action_type in (ActionType.VIEW_LOGS, ActionType.VIEW_CONFIG) and action.service_name:
                investigated.add(f"{action.action_type.value}:{action.service_name}")

            # Append this turn to conversation history
            conversation.append({"role": "user", "content": user_msg})
            conversation.append({"role": "assistant", "content": action_text})

            result = await env.step(action)
            obs = result.observation

            reward = result.reward or 0.0
            done = result.done
            error = obs.last_action_error

            rewards.append(reward)
            steps_taken = step

            action_str = json.dumps(action.model_dump(exclude_none=True), default=str)
            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            if done:
                break

        score = sum(rewards) / max_reward if max_reward > 0 else 0.0
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Task {task_name} error: {e}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


async def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    if IMAGE_NAME:
        env = await DevopsPipelineEnv.from_docker_image(IMAGE_NAME)
    else:
        env = DevopsPipelineEnv(
            base_url=os.getenv("ENV_BASE_URL", "http://localhost:8000")
        )

    try:
        for task in TASKS:
            await run_task(client, env, task)
    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
