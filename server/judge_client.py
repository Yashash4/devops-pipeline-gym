"""LLM Judge for GRPO training: scores episode trajectory quality.

Primary: Groq llama-3.3-70b-versatile (fast, free tier).
Fallback: Ollama Cloud gpt-oss:120b if Groq fails or is rate-limited.
Failure mode: `score_episode` returns 0.0 on all-provider failure so
training continues without a judge bonus. Never raises.

Called at most ONCE per episode (episode-end), never per-step — keeps
the wall-time cost bounded (~400 calls per 50-step GRPO run with
num_generations=8).
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)


class GroqJudgeClient:
    """Scores SRE incident-response trajectories with Groq + Ollama fallback.

    Return value of ``score_episode`` is clipped to [-1.0, +1.0]; higher
    is better. Training multiplies this by ``judge_weight`` (default 1.0)
    before adding to the episode reward.
    """

    GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
    OLLAMA_URL_PATH = "/api/chat"

    def __init__(
        self,
        groq_api_key: Optional[str] = None,
        ollama_api_key: Optional[str] = None,
        ollama_host: Optional[str] = None,
        model: str = "llama-3.3-70b-versatile",
        ollama_model: str = "gpt-oss:120b",
        timeout: float = 10.0,
        max_retries: int = 2,
    ) -> None:
        self.groq_key = groq_api_key or os.environ.get("GROQ_API_KEY")
        self.ollama_key = ollama_api_key or os.environ.get("OLLAMA_API_KEY")
        self.ollama_host = (
            ollama_host
            or os.environ.get("OLLAMA_HOST")
            or "https://ollama.com"
        )
        self.model = model
        self.ollama_model = ollama_model
        self.timeout = timeout
        self.max_retries = max_retries

    # ─── Prompt construction ────────────────────────────────────────────────

    def _build_judge_prompt(
        self,
        task_name: str,
        action_history: List[Dict[str, Any]],
        final_observation: Dict[str, Any],
        episode_reward: float,
        done: bool,
    ) -> str:
        actions_summary = []
        for i, step in enumerate(action_history[:15], 1):
            action = step.get("action", {}) if isinstance(step, dict) else {}
            reward = step.get("reward", 0.0) if isinstance(step, dict) else 0.0
            action_type = action.get("action_type", "unknown")
            service = action.get("service_name") or ""
            actions_summary.append(
                f"Step {i}: {action_type}({service}) -> reward {float(reward):+.3f}"
            )
        actions_text = "\n".join(actions_summary) if actions_summary else "(no actions)"

        final_summary = "unknown state"
        if isinstance(final_observation, dict):
            final_summary = str(final_observation.get("summary") or final_summary)

        return (
            "You are a Senior SRE reviewing an agent's incident response.\n\n"
            f"TASK: {task_name}\n\n"
            f"AGENT TRAJECTORY ({len(action_history)} actions):\n"
            f"{actions_text}\n\n"
            f"FINAL STATE: {final_summary}\n"
            f"EPISODE TERMINATED: {done}\n"
            f"RULE-BASED REWARD EARNED: {float(episode_reward):+.2f}\n\n"
            "Evaluate this trajectory on SRE workflow quality (NOT just reward earned):\n"
            "1. Did the agent investigate (view_logs, view_pipeline) before acting?\n"
            "2. Was the diagnosis correct given symptoms?\n"
            "3. Were actions appropriate for the failure type?\n"
            "4. Did they avoid breaking healthy services?\n"
            "5. Was the sequence efficient (not wasted steps)?\n\n"
            "Score from -1.0 (harmful/nonsensical) to +1.0 (textbook senior SRE work).\n"
            'Return ONLY a JSON object on a single line, no other text:\n'
            '{"score": <float between -1.0 and 1.0>, "reasoning": "<one sentence>"}'
        )

    # ─── Provider calls ─────────────────────────────────────────────────────

    def _call_groq(self, prompt: str) -> Optional[float]:
        if not self.groq_key:
            return None
        try:
            resp = httpx.post(
                self.GROQ_URL,
                headers={
                    "Authorization": f"Bearer {self.groq_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 200,
                    "temperature": 0.3,
                },
                timeout=self.timeout,
            )
            if resp.status_code != 200:
                logger.warning("Groq %d: %s", resp.status_code, resp.text[:150])
                return None
            data = resp.json()
            content = data["choices"][0]["message"]["content"].strip()
            return self._parse_score(content)
        except Exception as e:
            logger.warning("Groq call failed: %s", str(e)[:100])
            return None

    def _call_ollama(self, prompt: str) -> Optional[float]:
        if not self.ollama_key or not self.ollama_host:
            return None
        try:
            resp = httpx.post(
                f"{self.ollama_host.rstrip('/')}{self.OLLAMA_URL_PATH}",
                headers={
                    "Authorization": f"Bearer {self.ollama_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.ollama_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "options": {"temperature": 0.3, "num_predict": 200},
                },
                timeout=self.timeout * 2,  # Ollama is slower than Groq
            )
            if resp.status_code != 200:
                return None
            data = resp.json()
            content = data.get("message", {}).get("content", "").strip()
            return self._parse_score(content)
        except Exception as e:
            logger.warning("Ollama judge fallback failed: %s", str(e)[:100])
            return None

    # ─── Parsing ────────────────────────────────────────────────────────────

    @staticmethod
    def _clip(x: float) -> float:
        return max(-1.0, min(1.0, x))

    def _parse_score(self, content: str) -> Optional[float]:
        """Extract score from a judge response, tolerant of surrounding text."""
        if not content:
            return None

        # Strip optional markdown code fences.
        text = content.strip()
        if text.startswith("```"):
            text = text.split("```", 2)[1] if "```" in text[3:] else text[3:]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip("` \n")

        # Try a direct JSON parse first.
        try:
            data = json.loads(text)
            return self._clip(float(data.get("score", 0.0)))
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

        # Regex fallback — catches "prose... {"score": 0.6, ...}" cases.
        m = re.search(r'"score"\s*:\s*(-?\d+(?:\.\d+)?)', text)
        if m:
            try:
                return self._clip(float(m.group(1)))
            except ValueError:
                pass
        return None

    # ─── Public API ─────────────────────────────────────────────────────────

    def score_episode(
        self,
        task_name: str,
        action_history: List[Dict[str, Any]],
        final_observation: Dict[str, Any],
        episode_reward: float,
        done: bool,
    ) -> float:
        """Return a judge score in [-1, +1], or 0.0 if all providers fail."""
        prompt = self._build_judge_prompt(
            task_name, action_history, final_observation, episode_reward, done
        )

        # Primary: Groq, with light retry + 1s backoff between attempts.
        for attempt in range(self.max_retries):
            score = self._call_groq(prompt)
            if score is not None:
                return score
            if attempt < self.max_retries - 1:
                time.sleep(1.0)

        # Fallback: Ollama Cloud.
        logger.info("Groq failed after retries; trying Ollama fallback")
        score = self._call_ollama(prompt)
        if score is not None:
            return score

        # Graceful degradation: return neutral 0.0 — training continues.
        logger.warning("All judge providers failed; returning 0.0 (no judge bonus)")
        return 0.0
