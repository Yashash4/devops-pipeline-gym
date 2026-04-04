# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DevOps Pipeline Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from devops_pipeline_env.models import PipelineAction, PipelineObservation


class DevopsPipelineEnv(
    EnvClient[PipelineAction, PipelineObservation, State]
):
    """
    Client for the DevOps Pipeline Environment.

    Example:
        >>> with DevopsPipelineEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     result = client.step(PipelineAction(action_type="view_pipeline"))
    """

    def _step_payload(self, action: PipelineAction) -> Dict:
        return action.model_dump(exclude_none=True)

    def _parse_result(self, payload: Dict) -> StepResult[PipelineObservation]:
        obs_data = payload.get("observation", {})
        observation = PipelineObservation(**obs_data)

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
