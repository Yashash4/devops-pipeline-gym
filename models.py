# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the DevOps Pipeline Environment.

CI/CD deployment pipeline where an AI agent manages microservice deployments.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import BaseModel, Field


# --- Enums -------------------------------------------------------------------

class ActionType(str, Enum):
    VIEW_PIPELINE = "view_pipeline"
    VIEW_LOGS = "view_logs"
    VIEW_CONFIG = "view_config"
    EDIT_CONFIG = "edit_config"
    RUN_MIGRATION = "run_migration"
    DEPLOY = "deploy"
    ROLLBACK = "rollback"
    APPROVE = "approve"
    ABORT = "abort"


class ServiceHealth(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    DOWN = "down"
    UNKNOWN = "unknown"


class PipelineStage(str, Enum):
    IDLE = "idle"
    BUILD = "build"
    TEST = "test"
    STAGING = "staging"
    APPROVAL = "approval"
    DEPLOYING = "deploying"
    DEPLOYED = "deployed"
    ROLLED_BACK = "rolled_back"
    FAILED = "failed"


class MigrationType(str, Enum):
    SCHEMA = "schema"
    DATA = "data"
    ROLLBACK_MIGRATION = "rollback_migration"


class Role(str, Enum):
    """Action mode for role-gated dispatch."""
    DEV = "dev"
    SRE = "sre"
    OPS = "ops"


# Authoritative role→action_type mapping. Defined once so roles.py, rewards.py,
# and any validator use the same source of truth.
ROLE_ACTIONS: Dict[Role, List[ActionType]] = {
    Role.DEV: [
        ActionType.VIEW_CONFIG,
        ActionType.EDIT_CONFIG,
        ActionType.RUN_MIGRATION,
    ],
    Role.SRE: [
        ActionType.VIEW_LOGS,
        ActionType.VIEW_PIPELINE,
    ],
    Role.OPS: [
        ActionType.DEPLOY,
        ActionType.ROLLBACK,
        ActionType.APPROVE,
        ActionType.ABORT,
    ],
}


# --- Sub-models (plain BaseModel) --------------------------------------------

class ConfigEdit(BaseModel):
    key: str = Field(description="Dot-notation config path, e.g. 'database.pool_size'")
    value: str = Field(description="New value as string.")


class ServiceStatus(BaseModel):
    name: str
    health: ServiceHealth
    current_version: str
    cpu_percent: float = Field(description="CPU usage 0-100")
    memory_percent: float = Field(description="Memory usage 0-100")
    error_rate: float = Field(description="Errors per second")
    request_latency_ms: float = Field(description="p95 latency in milliseconds")
    active_connections: int
    last_deploy_timestamp: str = Field(description="ISO 8601 timestamp")
    recovery_status: str = Field(default="stable", description="Recovery state: 'stable' or 'stabilizing (N steps remaining)'")


class PipelineStatus(BaseModel):
    stage: PipelineStage
    triggered_by: str
    started_at: str = Field(description="ISO 8601 timestamp")
    commit_sha: str
    build_logs_snippet: Optional[str] = Field(
        default=None,
        description="Last N lines of build output.",
    )
    test_pass_count: Optional[int] = None
    test_fail_count: Optional[int] = None
    approval_required: bool = False
    blocked_reason: Optional[str] = None


class MigrationStatus(BaseModel):
    pending_migrations: List[str]
    last_applied: Optional[str] = None
    migration_errors: Optional[List[str]] = None


class AlertInfo(BaseModel):
    severity: str = Field(description="One of: critical, warning, info")
    message: str
    service_name: str
    timestamp: str


class RoleHistoryEntry(BaseModel):
    """One entry per role-driven action. Stored on observation."""
    step: int
    role: Role
    action_type: ActionType


# --- Action (extends OpenEnv Action) ----------------------------------------

class PipelineAction(Action):
    """Action for the DevOps Pipeline environment."""

    action_type: ActionType
    service_name: Optional[str] = Field(
        default=None,
        description="Target service. Required for view_logs, view_config, edit_config, deploy, rollback.",
    )
    target_version: Optional[str] = Field(
        default=None,
        description="Version tag to deploy. Required for deploy.",
    )
    config_edits: Optional[List[ConfigEdit]] = Field(
        default=None,
        description="List of config changes. Required for edit_config.",
    )
    migration_type: Optional[MigrationType] = Field(
        default=None,
        description="Type of migration. Required for run_migration.",
    )
    migration_name: Optional[str] = Field(
        default=None,
        description="Migration identifier. Required for run_migration.",
    )
    reason: Optional[str] = Field(
        default=None,
        description="Justification for approve/abort/rollback.",
    )
    role: Role = Field(
        default=Role.SRE,
        description="Which role is taking this action. Defaults to SRE for "
                    "backward compatibility with role-agnostic actions.",
    )


# --- Observation (extends OpenEnv Observation) --------------------------------

class PipelineObservation(Observation):
    """Everything the agent sees after each step."""

    task_description: str = Field(
        default="",
        description="Natural language description of what the agent must accomplish.",
    )
    goal: str = Field(
        default="",
        description="Specific success criteria for the current task.",
    )
    step_number: int = 0
    max_steps: int = 15
    services: List[ServiceStatus] = Field(default_factory=list)
    pipeline: Optional[PipelineStatus] = None
    migrations: Optional[MigrationStatus] = None
    active_alerts: List[AlertInfo] = Field(default_factory=list)
    available_actions: List[str] = Field(
        default_factory=list,
        description="List of valid action_type values in current state.",
    )
    last_action_result: Optional[str] = Field(
        default=None,
        description="Human-readable outcome of the previous action.",
    )
    last_action_error: Optional[str] = Field(
        default=None,
        description="Error message if previous action failed, else null.",
    )
    config_snapshot: Optional[Dict[str, str]] = Field(
        default=None,
        description="Current config key-value pairs when viewing/editing config.",
    )
    summary: Optional[str] = Field(
        default=None,
        description="Quick status summary highlighting degraded/down services.",
    )
    current_role: Role = Field(
        default=Role.SRE,
        description="Which role is expected to act next.",
    )
    role_history: List[RoleHistoryEntry] = Field(
        default_factory=list,
        description="Append-only log of role-driven actions, one entry per accepted step.",
    )
