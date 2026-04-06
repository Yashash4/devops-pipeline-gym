# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Simulation engine for the DevOps Pipeline Environment."""

import random

from devops_pipeline_env.models import (
    ActionType,
    AlertInfo,
    ConfigEdit,
    MigrationStatus,
    PipelineAction,
    PipelineStage,
    PipelineStatus,
    ServiceHealth,
    ServiceStatus,
)


class ServiceState:
    """State machine for a single microservice."""

    def __init__(self, name, version, health, config, dependencies,
                 latency_ms, error_rate, cpu, memory, rng=None):
        self.name = name
        self.current_version = version
        self.target_version = None
        self.health = health
        self.config = dict(config)
        self.dependencies = list(dependencies) if dependencies else []
        self.latency_ms = latency_ms
        self.error_rate = error_rate
        self.cpu_percent = cpu
        self.memory_percent = memory
        self.active_connections = 100
        self.staging_deployed = False
        self.staging_verified = False
        self.prod_deployed = False
        self.last_deploy_timestamp = "2026-04-01T00:00:00Z"
        self.logs = []
        self._rng = rng or random.Random(0)
        # Staged health recovery: 0 = fully recovered, >0 = still recovering
        self._recovery_steps_remaining = 0
        self._recovery_target_latency = 0.0
        self._recovery_target_error_rate = 0.0

    def deploy_to_staging(self, version, scenario):
        """Deploy version to staging. Returns result text."""
        self.staging_deployed = True
        self.target_version = version

        # 8% chance of transient staging failure on first attempt
        # Skip for clean_deploy (easy task) and during incidents (health already degraded/down)
        transient_roll = self._rng.random()  # always consume RNG for determinism
        is_clean_deploy = hasattr(self, '_task_name') and self._task_name == "clean_deploy"
        if not is_clean_deploy and not self.staging_verified and self.health == ServiceHealth.HEALTHY and transient_roll < 0.08:
            self.staging_deployed = True  # deployed but not verified
            self.logs.append(
                f"[DEPLOY] Deployed {self.name} {version} to staging. "
                f"Transient failure: health check timed out. Retry should succeed."
            )
            return (
                f"Deployed {self.name} {version} to staging. "
                f"WARNING: Transient health check timeout. Try deploying again."
            )

        if scenario.check_config_error(self.name, self.config):
            self.health = ServiceHealth.DEGRADED
            lat_mult = self._rng.uniform(0.8, 1.2)
            err_mult = self._rng.uniform(0.9, 1.1)
            self.error_rate = round(12.0 * err_mult, 2)
            self.latency_ms = round(300.0 * lat_mult, 1)
            self.logs.append(
                f"[DEPLOY] Deployed {self.name} {version} to staging. "
                f"WARNING: Health check DEGRADED. Error rate elevated "
                f"({self.error_rate:.1f}/s, latency {self.latency_ms:.0f}ms)."
            )
            return (
                f"Deployed {self.name} {version} to staging. "
                f"WARNING: Health check degraded. Error rate elevated."
            )
        self.health = ServiceHealth.HEALTHY
        self.staging_verified = True
        lat_mult = self._rng.uniform(0.8, 1.2)
        self.error_rate = round(0.1 * self._rng.uniform(0.9, 1.1), 3)
        self.latency_ms = round(45.0 * lat_mult, 1)
        self.logs.append(
            f"[DEPLOY] Deployed {self.name} {version} to staging. Health check: PASSED."
        )
        return f"Deployed {self.name} {version} to STAGING. Staging verified. Deploy same service+version again to PROMOTE TO PRODUCTION."

    def deploy_to_production(self, version):
        """Promote to production."""
        if not self.staging_verified:
            self.health = ServiceHealth.DEGRADED
            lat_mult = self._rng.uniform(0.8, 1.2)
            err_mult = self._rng.uniform(0.9, 1.1)
            self.error_rate = round(25.0 * err_mult, 2)
            self.latency_ms = round(500.0 * lat_mult, 1)
            self.logs.append(
                f"[DEPLOY] Deployed {self.name} {version} to production "
                f"WITHOUT staging verification. High risk."
            )
            return (
                f"Deployed {self.name} {version} to production "
                f"WITHOUT staging verification. High risk."
            )
        self.prod_deployed = True
        self.current_version = version
        # Staged recovery: takes 1-3 steps to fully stabilize
        recovery_steps = self._rng.randint(1, 3)
        self._recovery_steps_remaining = recovery_steps
        base_latency = 45.0 * self._rng.uniform(0.8, 1.2)
        base_error_rate = 0.1 * self._rng.uniform(0.9, 1.1)

        # Non-linear deploy quality: same seed = same outcome
        quality_roll = self._rng.random()
        deploy_note = ""
        if quality_roll < 0.7:
            # Clean deploy — recovers to near-perfect
            pass  # base values are already good
        elif quality_roll < 0.9:
            # Minor issues — recovers to good but not perfect
            base_latency *= 1.5
            base_error_rate *= 3.0
            deploy_note = " Minor post-deploy issues detected."
            self.logs.append(
                f"[DEPLOY] {self.name}: Minor post-deploy issues detected. "
                f"Performance slightly below optimal."
            )
        else:
            # Unstable deploy — recovers poorly
            base_latency *= 2.5
            base_error_rate *= 8.0
            self.error_rate += 1.5
            deploy_note = " Post-deploy instability detected."
            self.logs.append(
                f"[DEPLOY] {self.name}: Post-deploy instability detected. "
                f"Elevated error rate."
            )

        self._recovery_target_latency = round(base_latency, 1)
        self._recovery_target_error_rate = round(base_error_rate, 3)
        # Start at slightly elevated values during recovery
        self.health = ServiceHealth.HEALTHY
        self.latency_ms = round(base_latency * (1.0 + 0.3 * recovery_steps), 1)
        self.error_rate = round(base_error_rate * (1.0 + 0.5 * recovery_steps), 3)
        # Trade-off: deploy causes temporary CPU/latency spike (warmup load)
        # Clean deploy tasks get reduced spikes — they should be clean
        if hasattr(self, '_task_name') and self._task_name == "clean_deploy":
            self.cpu_percent = min(self.cpu_percent + 3, 99)
            self.latency_ms += round(30 * self._rng.uniform(0.8, 1.2), 1)
        else:
            self.cpu_percent = min(self.cpu_percent + 15, 99)
            self.latency_ms += round(200 * self._rng.uniform(0.8, 1.2), 1)
        self.last_deploy_timestamp = "2026-04-01T12:00:00Z"
        self.logs.append(
            f"[DEPLOY] Promoted {self.name} {version} to production. Health: HEALTHY. "
            f"Stabilizing over ~{recovery_steps} step(s). CPU/latency spike from warmup."
        )
        return (
            f"Promoted {self.name} {version} to production. Health: HEALTHY. "
            f"Deployed successfully. Service under warmup load — temporary CPU/latency spike expected."
            f"{deploy_note}"
        )

    def tick_recovery(self):
        """Called each step to progress staged health recovery."""
        if self._recovery_steps_remaining > 0:
            self._recovery_steps_remaining -= 1
            if self._recovery_steps_remaining == 0:
                # Fully recovered
                self.latency_ms = self._recovery_target_latency
                self.error_rate = self._recovery_target_error_rate
            else:
                # Interpolate toward target
                progress = 1.0 - (self._recovery_steps_remaining / (self._recovery_steps_remaining + 1))
                self.latency_ms = round(
                    self.latency_ms + (self._recovery_target_latency - self.latency_ms) * progress, 1
                )
                self.error_rate = round(
                    self.error_rate + (self._recovery_target_error_rate - self.error_rate) * progress, 3
                )

    def rollback(self):
        """Rollback to previous version."""
        self.health = ServiceHealth.HEALTHY
        lat_mult = self._rng.uniform(0.8, 1.2)
        err_mult = self._rng.uniform(0.9, 1.1)
        self.error_rate = round(0.5 * err_mult, 3)
        self.latency_ms = round(50.0 * lat_mult * 0.7, 1)
        self.staging_deployed = False
        self.staging_verified = False
        self.prod_deployed = True  # still in prod, just rolled back
        self._recovery_steps_remaining = 0
        # Trade-off: 25% chance rollback reintroduces a known bug
        regression = False
        if self._rng.random() < 0.25:
            self.error_rate = round(self.error_rate + 3.0, 2)
            regression = True
            self.logs.append(
                f"[ROLLBACK] Rolled back {self.name} to {self.current_version}. "
                f"Warning: rollback may have reintroduced known issue from previous version"
            )
        else:
            self.logs.append(
                f"[ROLLBACK] Rolled back {self.name} to {self.current_version}. Service healthy."
            )
        result = f"Rolled back {self.name} to {self.current_version}. Rolled back. Monitoring for regression..."
        if regression:
            result += f" WARNING: Error rate elevated ({self.error_rate:.1f}/s) — possible regression."
        return result

    def set_config(self, key, value):
        """Edit a config value."""
        old = self.config.get(key, "<not set>")
        self.config[key] = value
        # Trade-off: config change causes brief restart spike
        self.latency_ms += round(100 * self._rng.uniform(0.8, 1.2), 1)
        self.cpu_percent = min(self.cpu_percent + 5, 99)
        self.logs.append(f"[CONFIG] {self.name}: {key} changed from '{old}' to '{value}'. Service restarting.")
        return f"Config {self.name}: {key} changed from '{old}' to '{value}'. Config updated. Service restarting — brief latency spike."

    def get_config_snapshot(self):
        return dict(self.config)

    def get_logs(self):
        return list(self.logs)

    def _get_health_pct(self):
        """Get numeric health percentage for this service."""
        h = 100.0
        if self.health == ServiceHealth.DOWN:
            h = 0.0
        elif self.health == ServiceHealth.DEGRADED:
            h = 50.0
        h -= min(self.error_rate * 2, 30)
        if self.latency_ms > 200:
            h -= min((self.latency_ms - 200) / 10, 30)
        return max(h, 0.0)

    def to_status(self):
        return ServiceStatus(
            name=self.name,
            health=self.health,
            current_version=self.current_version,
            cpu_percent=self.cpu_percent,
            memory_percent=self.memory_percent,
            error_rate=self.error_rate,
            request_latency_ms=self.latency_ms,
            active_connections=self.active_connections,
            last_deploy_timestamp=self.last_deploy_timestamp,
        )


class PipelineEngine:
    """Manages all services, pipeline state, migrations, alerts."""

    def __init__(self, scenario, seed):
        self.scenario = scenario
        self._rng = random.Random(seed)
        self.services = {}
        self.pipeline_stage = PipelineStage.IDLE
        self.migrations_pending = []
        self.migrations_applied = []
        self.migration_errors = []
        self.alerts = []
        self.commit_sha = "abc123"
        self.triggered_by = "deploy-bot"
        self.started_at = "2026-04-01T10:00:00Z"
        self.test_pass = 0
        self.test_fail = 0
        self.build_logs = ""
        self._time_pressure = False  # Set by scenario if needed

        # Initialize from scenario
        scenario.setup(self)

        # Inject the shared RNG and task name into all services created by the scenario
        for svc in self.services.values():
            svc._rng = self._rng
            svc._task_name = scenario.task_name

    def execute(self, action):
        """Execute an action. Returns human-readable result string."""
        # Tick health recovery for all services
        for svc in self.services.values():
            svc.tick_recovery()

        # Time pressure: degrade api-gateway health each step
        if self._time_pressure:
            self._apply_time_pressure()

        # Cascading failures: unhealthy services degrade their dependents
        self._tick_cascading_effects()

        if action.action_type == ActionType.VIEW_PIPELINE:
            result = self._view_pipeline()
        elif action.action_type == ActionType.VIEW_LOGS:
            result = self._view_logs(action.service_name)
        elif action.action_type == ActionType.VIEW_CONFIG:
            result = self._view_config(action.service_name)
        elif action.action_type == ActionType.EDIT_CONFIG:
            result = self._edit_config(action.service_name, action.config_edits)
        elif action.action_type == ActionType.RUN_MIGRATION:
            result = self._run_migration(action.migration_name, action.migration_type)
        elif action.action_type == ActionType.DEPLOY:
            result = self._deploy(action.service_name, action.target_version)
        elif action.action_type == ActionType.ROLLBACK:
            result = self._rollback(action.service_name)
        elif action.action_type == ActionType.APPROVE:
            result = self._approve(action.reason)
        elif action.action_type == ActionType.ABORT:
            result = self._abort(action.reason)
        else:
            result = "Unknown action."

        # Cross-metric compounding: metrics affect each other
        self._tick_metric_compounding()

        # Non-linear tipping points — cliff effects
        self._tick_tipping_points()

        return result

    # --- Cross-metric compounding ---------------------------------------------

    def _tick_metric_compounding(self):
        """Metrics compound on each other — creates realistic spirals and recovery."""
        if self.scenario.task_name == "clean_deploy":
            return
        for name, svc in self.services.items():
            # Degradation spirals (moderate — should not kill episodes in <5 steps)
            if svc.error_rate > 15.0:
                svc.cpu_percent = min(svc.cpu_percent + 3, 99)
            if svc.cpu_percent > 90:
                svc.latency_ms = round(min(svc.latency_ms + 100, 5000), 1)
            if svc.latency_ms > 3000:
                svc.error_rate = round(min(svc.error_rate + 1.0, 50.0), 2)

            # Natural recovery (when metrics are good, they help each other)
            if svc.error_rate < 2.0:
                svc.cpu_percent = max(svc.cpu_percent - 3, 10)
            if svc.cpu_percent < 50:
                svc.latency_ms = round(max(svc.latency_ms - 50, 20), 1)
            if svc.latency_ms < 200 and svc.error_rate < 1.0:
                svc.error_rate = round(max(svc.error_rate - 0.5, 0.0), 2)

    # --- Non-linear tipping points -------------------------------------------

    def _tick_tipping_points(self):
        """Non-linear tipping points — systems cliff instead of degrading linearly."""
        if self.scenario.task_name == "clean_deploy":
            return
        for name, svc in self.services.items():
            # CPU cliff: above 85% = exponential error growth
            if svc.cpu_percent > 85:
                overflow = svc.cpu_percent - 85
                svc.error_rate = round(min(svc.error_rate + overflow * 0.2, 50.0), 2)

            # Latency cliff: above 2000ms = rapid collapse
            if svc.latency_ms > 2000:
                svc.error_rate = round(min(svc.error_rate + 3.0, 50.0), 2)

            # Health cliff: below 30% health = accelerating death spiral
            base = 50.0 if svc.health == ServiceHealth.DEGRADED else (
                100.0 if svc.health == ServiceHealth.HEALTHY else 0.0
            )
            err_penalty = min(svc.error_rate * 2, 30)
            lat_penalty = min(max(0, svc.latency_ms - 200) / 10, 30)
            health_pct = max(0, base - err_penalty - lat_penalty)
            if health_pct < 30:
                svc.error_rate = round(min(svc.error_rate * 1.3, 50.0), 2)

            # Latency → CPU feedback (high latency = retries = more CPU)
            if svc.latency_ms > 1500:
                svc.cpu_percent = min(svc.cpu_percent + 3, 99)

    # --- Cascading failures ---------------------------------------------------

    def _get_dependents(self, service_name):
        """Find all services that list service_name in their dependencies."""
        return [
            svc for svc in self.services.values()
            if service_name in svc.dependencies
        ]

    def _tick_cascading_effects(self):
        """Unhealthy services degrade their dependents each step."""
        for svc in self.services.values():
            health_pct = svc._get_health_pct()
            if health_pct >= 50.0:
                continue  # healthy enough, no cascade

            dependents = self._get_dependents(svc.name)
            for dep in dependents:
                if dep.health == ServiceHealth.DOWN:
                    continue  # already down, can't get worse from cascade

                # Determine cascade severity
                if health_pct < 20.0:
                    # Source is effectively down — moderate cascade
                    err_increase = 1.5
                    lat_increase = 30.0
                else:
                    # Source is degraded — lighter cascade
                    err_increase = 0.5
                    lat_increase = 10.0

                old_err = dep.error_rate
                dep.error_rate = round(min(dep.error_rate + err_increase, 45.0), 2)
                dep.latency_ms = round(min(dep.latency_ms + lat_increase, 4500.0), 1)

                # If error rate gets high enough, mark as degraded
                if dep.error_rate > 5.0 and dep.health == ServiceHealth.HEALTHY:
                    dep.health = ServiceHealth.DEGRADED

                # Floor: cascading alone can't push health below 5%
                # (prevent instant death spirals)
                dep_health = dep._get_health_pct()
                if dep_health < 5.0:
                    dep.error_rate = round(max(old_err, dep.error_rate - err_increase + 1.0), 2)

                # Add cascade alert (only if not already alerted this step)
                cascade_alert_key = f"cascade:{svc.name}->{dep.name}"
                existing = [a for a in self.alerts if cascade_alert_key in a.message]
                if not existing:
                    self.alerts.append(AlertInfo(
                        severity="warning",
                        message=(
                            f"Cascading: {svc.name} (health {health_pct:.0f}%) is degrading "
                            f"{dep.name} — error_rate +{err_increase}/s, latency +{lat_increase:.0f}ms "
                            f"[{cascade_alert_key}]"
                        ),
                        service_name=dep.name,
                        timestamp="2026-04-01T12:00:00Z",
                    ))

                dep.logs.append(
                    f"[CASCADE] Upstream {svc.name} unhealthy (health {health_pct:.0f}%) — "
                    f"{dep.name} error_rate now {dep.error_rate:.1f}/s, "
                    f"latency {dep.latency_ms:.0f}ms"
                )

        # Recovery propagation: healthy services help their dependents recover
        for name, svc in self.services.items():
            if svc.health == ServiceHealth.HEALTHY and svc.error_rate < 2.0:
                dependents = self._get_dependents(name)
                for dep in dependents:
                    if dep.health == ServiceHealth.DEGRADED:
                        dep.error_rate = round(dep.error_rate * 0.9, 2)
                        dep.latency_ms = round(dep.latency_ms * 0.9, 1)

    # --- Action handlers ------------------------------------------------------

    def _view_pipeline(self):
        services_summary = "\n".join(
            f"  {s.name}: {s.health.value} | v{s.current_version} -> "
            f"v{s.target_version or 'N/A'} | "
            f"latency={s.latency_ms:.0f}ms | errors={s.error_rate:.1f}/s"
            for s in self.services.values()
        )
        return (
            f"Pipeline Stage: {self.pipeline_stage.value}\n"
            f"Commit: {self.commit_sha}\n"
            f"Tests: {self.test_pass} passed, {self.test_fail} failed\n"
            f"Pending Migrations: {len(self.migrations_pending)}\n"
            f"Services:\n{services_summary}"
        )

    def _view_logs(self, service_name):
        svc = self.services.get(service_name)
        if not svc:
            return f"No service named '{service_name}'"
        logs = svc.get_logs()
        if not logs:
            return f"No logs available for {service_name}."
        return f"Logs for {service_name}:\n" + "\n".join(logs[-20:])

    def _view_config(self, service_name):
        svc = self.services.get(service_name)
        if not svc:
            return f"No service named '{service_name}'"
        config = svc.get_config_snapshot()
        lines = [f"  {k} = {v}" for k, v in config.items()]
        return f"Config for {service_name}:\n" + "\n".join(lines)

    def _edit_config(self, service_name, edits):
        svc = self.services.get(service_name)
        if not svc:
            return f"No service named '{service_name}'"
        results = []
        for edit in edits:
            result = svc.set_config(edit.key, edit.value)
            results.append(result)
        # If the config error is now fixed and service was degraded, reset staging
        # so the agent can re-deploy through staging with the corrected config
        if svc.health == ServiceHealth.DEGRADED and not self.scenario.check_config_error(service_name, svc.config):
            svc.staging_deployed = False
            svc.staging_verified = False
            svc.health = ServiceHealth.HEALTHY
            svc.error_rate = round(0.1 * self._rng.uniform(0.9, 1.1), 3)
            svc.latency_ms = round(50.0 * self._rng.uniform(0.8, 1.2), 1)
            results.append(f"Config fix detected for {service_name}. Service health restored. Ready for re-deploy.")
        return "\n".join(results)

    def _run_migration(self, migration_name, migration_type):
        if migration_name not in self.migrations_pending:
            return (
                f"Migration '{migration_name}' not found in pending: "
                f"{self.migrations_pending}"
            )
        success = self.scenario.run_migration(self, migration_name)
        if success:
            self.migrations_pending.remove(migration_name)
            self.migrations_applied.append(migration_name)
            return f"Migration '{migration_name}' applied successfully."
        else:
            error = f"Migration '{migration_name}' FAILED."
            self.migration_errors.append(error)
            return error

    def _deploy(self, service_name, target_version):
        svc = self.services.get(service_name)
        if not svc:
            return f"No service named '{service_name}'"

        # Check migration dependencies
        if self.migrations_pending and self.scenario.migration_blocks_deploy(service_name):
            return (
                f"BLOCKED: Pending migrations must be applied before deploying "
                f"{service_name}. Pending: {self.migrations_pending}"
            )

        # Check if any dependency is unhealthy — 50% chance of deploy failure
        for dep_name in svc.dependencies:
            dep_svc = self.services.get(dep_name)
            if dep_svc and dep_svc._get_health_pct() < 50.0:
                if self._rng.random() < 0.5:
                    svc.logs.append(
                        f"[DEPLOY] Deploy {svc.name} {target_version} FAILED — "
                        f"dependency {dep_name} is unhealthy "
                        f"(health {dep_svc._get_health_pct():.0f}%). Retry may succeed."
                    )
                    return (
                        f"DEPLOY UNSTABLE: Dependency {dep_name} is unhealthy "
                        f"(health {dep_svc._get_health_pct():.0f}%). "
                        f"Deploy of {service_name} failed. Retry may succeed."
                    )

        # Determine target environment
        if not svc.staging_deployed:
            self.pipeline_stage = PipelineStage.STAGING
            return svc.deploy_to_staging(target_version, self.scenario)
        else:
            self.pipeline_stage = PipelineStage.DEPLOYING
            result = svc.deploy_to_production(target_version)
            # Notify scenario of deploy (for cascading effects)
            if hasattr(self.scenario, 'on_prod_deploy'):
                extra = self.scenario.on_prod_deploy(self, service_name, target_version)
                if extra:
                    result += "\n" + extra
            # Check if all target services deployed
            if all(s.prod_deployed for s in self.services.values() if s.target_version):
                self.pipeline_stage = PipelineStage.DEPLOYED
            return result

    def _rollback(self, service_name):
        svc = self.services.get(service_name)
        if not svc:
            return f"No service named '{service_name}'"
        self.pipeline_stage = PipelineStage.ROLLED_BACK

        # Check if dependents rely on current version's APIs
        old_version = svc.current_version
        dependents = self._get_dependents(service_name)
        result = svc.rollback()

        # Warn about dependent services and increase their error rates
        for dep in dependents:
            dep.error_rate = round(dep.error_rate + 5.0, 2)
            if dep.health == ServiceHealth.HEALTHY and dep.error_rate > 3.0:
                dep.health = ServiceHealth.DEGRADED
            self.alerts.append(AlertInfo(
                severity="warning",
                message=(
                    f"Rollback impact: {dep.name} depends on {service_name} "
                    f"{old_version}. Rollback may break {dep.name}. "
                    f"Error rate increased to {dep.error_rate:.1f}/s."
                ),
                service_name=dep.name,
                timestamp="2026-04-01T12:00:00Z",
            ))
            dep.logs.append(
                f"[ROLLBACK-IMPACT] {service_name} rolled back from {old_version} — "
                f"{dep.name} error_rate increased to {dep.error_rate:.1f}/s. "
                f"Dependency on {old_version} APIs may be broken."
            )

        if hasattr(self.scenario, 'on_rollback'):
            self.scenario.on_rollback(self, service_name)
        return result

    def _approve(self, reason):
        self.pipeline_stage = PipelineStage.DEPLOYED
        return f"Deployment APPROVED. Reason: {reason or 'No reason given.'}"

    def _abort(self, reason):
        self.pipeline_stage = PipelineStage.FAILED
        return f"Deployment ABORTED. Reason: {reason or 'No reason given.'}"

    # --- State queries --------------------------------------------------------

    def snapshot(self):
        """Capture current state for reward calculation."""
        return {
            "services": {
                name: {
                    "health": s.health.value,
                    "error_rate": s.error_rate,
                    "latency_ms": s.latency_ms,
                    "prod_deployed": s.prod_deployed,
                    "staging_verified": s.staging_verified,
                }
                for name, s in self.services.items()
            },
            "system_health": self.get_system_health(),
            "pipeline_stage": self.pipeline_stage.value,
        }

    def get_system_health(self):
        """Aggregate health 0-100."""
        if not self.services:
            return 100.0
        total = 0.0
        for svc in self.services.values():
            total += svc._get_health_pct()
        return total / len(self.services)

    def get_service_statuses(self):
        return [s.to_status() for s in self.services.values()]

    def get_pipeline_status(self):
        return PipelineStatus(
            stage=self.pipeline_stage,
            triggered_by=self.triggered_by,
            started_at=self.started_at,
            commit_sha=self.commit_sha,
            build_logs_snippet=self.build_logs if self.build_logs else None,
            test_pass_count=self.test_pass,
            test_fail_count=self.test_fail,
        )

    def get_migration_status(self):
        return MigrationStatus(
            pending_migrations=list(self.migrations_pending),
            last_applied=self.migrations_applied[-1] if self.migrations_applied else None,
            migration_errors=self.migration_errors if self.migration_errors else None,
        )

    def get_alerts(self):
        return list(self.alerts)

    def get_service_names(self):
        return list(self.services.keys())

    def has_services(self):
        return len(self.services) > 0

    def has_pending_migrations(self):
        return len(self.migrations_pending) > 0

    def _apply_time_pressure(self):
        """During incidents, degraded services get worse each step."""
        task = self.scenario.task_name

        if task == "judgment_call":
            api_gw = self.services.get("api-gateway")
            if api_gw and api_gw.health == ServiceHealth.DEGRADED:
                degrade_lat = 80 * self._rng.uniform(0.8, 1.2)
                degrade_err = 0.8 * self._rng.uniform(0.9, 1.1)
                api_gw.latency_ms = round(min(api_gw.latency_ms + degrade_lat, 5000), 1)
                api_gw.error_rate = round(min(api_gw.error_rate + degrade_err, 50.0), 2)
                api_gw.cpu_percent = min(api_gw.cpu_percent + 1, 99)
                api_gw.logs.append(
                    f"[DEGRADING] api-gateway latency now {api_gw.latency_ms:.0f}ms, "
                    f"errors {api_gw.error_rate:.1f}/s — situation worsening"
                )

        elif task == "broken_pipeline":
            # Cache-service degrades if config error persists
            cache = self.services.get("cache-service")
            if cache and self.scenario.check_config_error("cache-service", cache.config):
                health_drop = 3.0 * self._rng.uniform(0.8, 1.2)
                cache.error_rate = round(min(cache.error_rate + health_drop * 0.5, 25.0), 2)
                cache.latency_ms = round(min(cache.latency_ms + 30.0 * self._rng.uniform(0.8, 1.2), 2000.0), 1)
                if cache.error_rate > 3.0 and cache.health == ServiceHealth.HEALTHY:
                    cache.health = ServiceHealth.DEGRADED
                cache.logs.append(
                    f"[DEGRADING] cache-service using staging Redis — "
                    f"error_rate now {cache.error_rate:.1f}/s, "
                    f"latency {cache.latency_ms:.0f}ms"
                )

            # Api-gateway latency increases if migration not applied
            api_gw = self.services.get("api-gateway")
            if api_gw and "add_index_users_email" in self.migrations_pending:
                lat_increase = 50.0 * self._rng.uniform(0.8, 1.2)
                api_gw.latency_ms = round(min(api_gw.latency_ms + lat_increase, 2000.0), 1)
                api_gw.logs.append(
                    f"[DEGRADING] api-gateway missing index — "
                    f"user query latency now {api_gw.latency_ms:.0f}ms"
                )

        elif task == "capacity_crisis":
            db = self.services.get("database-primary")
            api_gw = self.services.get("api-gateway")
            # Time pressure only while connection pool bottleneck persists
            if db and self.scenario.check_config_error("database-primary", db.config):
                db.cpu_percent = min(db.cpu_percent + 2, 99)
                db.latency_ms = round(db.latency_ms + 15, 1)
            # api-gateway degrades only while db bottleneck persists
            if (api_gw and api_gw.health == ServiceHealth.DEGRADED
                    and db and self.scenario.check_config_error("database-primary", db.config)):
                api_gw.latency_ms = round(min(api_gw.latency_ms + 30, 5000), 1)
                api_gw.error_rate = round(min(api_gw.error_rate + 0.5, 50.0), 2)

        elif task == "random_incident":
            failing = getattr(self.scenario, '_failing_service', None)
            if failing:
                svc = self.services.get(failing)
                if svc and svc.health == ServiceHealth.DEGRADED:
                    svc.error_rate = round(min(svc.error_rate + 0.5, 50.0), 2)
                    svc.latency_ms = round(min(svc.latency_ms + 30, 5000), 1)
