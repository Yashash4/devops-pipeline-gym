# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Scenario definitions for the DevOps Pipeline Environment."""

from devops_pipeline_env.models import AlertInfo, ServiceHealth


class Scenario:
    """Base scenario configuration."""

    def __init__(
        self,
        task_name,
        task_description,
        goal,
        max_steps,
    ):
        self.task_name = task_name
        self.task_description = task_description
        self.goal = goal
        self.max_steps = max_steps

    def setup(self, engine):
        """Initialize engine state for this scenario. Override per task."""
        raise NotImplementedError

    def run_migration(self, engine, migration_name):
        """Execute a migration. Returns True on success."""
        return True

    def migration_blocks_deploy(self, service_name):
        """Does a pending migration block deploying this service?"""
        return False

    def check_config_error(self, service_name, config):
        """Check if a service has a config error. Returns True if error exists."""
        return False


class CleanDeployScenario(Scenario):
    """Task 1: Easy — deploy 2 services, everything works."""

    def __init__(self):
        super().__init__(
            task_name="clean_deploy",
            task_description=(
                "You are managing a CI/CD pipeline. Two services need to be "
                "deployed to production: api-gateway v2.3.1 and web-frontend "
                "v1.9.0. All tests pass. No migrations required."
            ),
            goal=(
                "Deploy both services to production with zero downtime. "
                "Final system health should remain above 95%."
            ),
            max_steps=15,
        )

    def setup(self, engine):
        from server.pipeline_engine import ServiceState

        engine.services["database-primary"] = ServiceState(
            name="database-primary",
            version="v5.2.0",
            health=ServiceHealth.HEALTHY,
            config={
                "max_connections": "50",
                "replication_lag_ms": "0",
                "shared_buffers": "4GB",
                "wal_level": "replica",
            },
            dependencies=[],
            latency_ms=20.0,
            error_rate=0.3,
            cpu=30.0,
            memory=60.0,
        )
        engine.services["database-primary"].prod_deployed = True
        engine.services["database-primary"].logs = [
            "2026-04-01T09:58:10.001Z INFO  [database-primary] PostgreSQL 15.4 started. max_connections=50. shared_buffers=4GB.",
            "2026-04-01T09:58:10.112Z INFO  [database-primary] Replication lag: 0ms. WAL level: replica. All replicas in sync.",
            "2026-04-01T09:58:10.334Z INFO  [database-primary] Connection pool: 12/50 active. Query throughput: 1.2k/s.",
        ]

        engine.services["auth-service"] = ServiceState(
            name="auth-service",
            version="v3.1.0",
            health=ServiceHealth.HEALTHY,
            config={
                "token_ttl_seconds": "3600",
                "jwt_algorithm": "RS256",
                "rate_limit_per_minute": "1000",
                "cert_expiry": "2026-12-01",
            },
            dependencies=["database-primary"],
            latency_ms=12.0,
            error_rate=0.1,
            cpu=25.0,
            memory=40.0,
        )
        engine.services["auth-service"].prod_deployed = True
        engine.services["auth-service"].logs = [
            "2026-04-01T09:58:10.001Z INFO  [auth-service] OAuth2 provider started. Algorithm: RS256. TTL: 3600s.",
            "2026-04-01T09:58:10.112Z INFO  [auth-service] Certificate valid until 2026-12-01. Rate limit: 1000/min.",
        ]

        engine.services["api-gateway"] = ServiceState(
            name="api-gateway",
            version="v2.3.0",
            health=ServiceHealth.HEALTHY,
            config={
                "database.pool_size": "20",
                "cache.ttl": "300",
                "log.level": "info",
            },
            dependencies=["database-primary", "auth-service"],
            latency_ms=45.0,
            error_rate=0.1,
            cpu=35.0,
            memory=42.0,
        )
        engine.services["api-gateway"].target_version = "v2.3.1"
        engine.services["api-gateway"].logs = [
            "2026-04-01T09:58:12.441Z INFO  [api-gateway] com.pipeline.core.Bootstrap — Service api-gateway started on port 8080 (pid=1423)",
            "2026-04-01T09:58:12.887Z INFO  [api-gateway] com.pipeline.db.ConnectionPool — Database connection pool initialized: 20 connections to postgres-primary.internal:5432/pipeline_db",
            "2026-04-01T09:58:13.102Z INFO  [api-gateway] com.pipeline.health.HealthCheck — Liveness probe passed. Readiness probe passed. Registered with service mesh.",
            "2026-04-01T09:58:14.330Z WARN  [api-gateway] com.pipeline.api.DeprecationFilter — Endpoint /api/v1/legacy is deprecated and scheduled for removal in v3.0. 12 requests in last 24h.",
            "2026-04-01T09:58:15.001Z DEBUG [api-gateway] com.pipeline.gc.GCMonitor — G1 GC pause: 8ms (Young), heap: 412MB/1024MB. Within acceptable thresholds.",
            "2026-04-01T10:00:00.000Z INFO  [api-gateway] com.pipeline.deploy.Coordinator — Build artifact api-gateway-2.3.1.jar verified. SHA256: a4f8c2e9...b31d. Ready for deployment.",
        ]

        engine.services["web-frontend"] = ServiceState(
            name="web-frontend",
            version="v1.8.0",
            health=ServiceHealth.HEALTHY,
            config={
                "api.endpoint": "https://api.internal:8080",
                "cdn.enabled": "true",
                "log.level": "info",
            },
            dependencies=["api-gateway", "auth-service"],
            latency_ms=30.0,
            error_rate=0.05,
            cpu=28.0,
            memory=35.0,
        )
        engine.services["web-frontend"].target_version = "v1.9.0"
        engine.services["web-frontend"].logs = [
            "2026-04-01T09:58:11.201Z INFO  [web-frontend] server.js:42 — Express server listening on 0.0.0.0:3000 (worker pid=891)",
            "2026-04-01T09:58:12.005Z INFO  [web-frontend] cdn/warmup.js:118 — CDN cache warmed: 847 assets prefetched (2.3MB total, threshold 5MB — OK)",
            "2026-04-01T09:58:12.340Z INFO  [web-frontend] health/probe.js:25 — Health check endpoint /healthz responding. Upstream api-gateway: reachable.",
            "2026-04-01T09:58:13.887Z DEBUG [web-frontend] middleware/metrics.js:67 — Request histogram initialized. p50=12ms, p95=28ms, p99=45ms over last 5min window.",
            "2026-04-01T10:00:00.000Z INFO  [web-frontend] deploy/artifact.js:31 — Build bundle web-frontend-1.9.0.tar.gz verified. 1.8MB (gzip). Asset hash: e7c3f1...a92b.",
        ]

        engine.commit_sha = "a1b2c3d"
        engine.triggered_by = "deploy-bot"
        engine.test_pass = 145
        engine.test_fail = 0
        engine.build_logs = "Build succeeded. 145/145 tests passed."


class BrokenPipelineScenario(Scenario):
    """Task 2: Medium — diagnose failures, fix config, run migration."""

    def __init__(self):
        super().__init__(
            task_name="broken_pipeline",
            task_description=(
                "Deploy 3 services. There are test failures to investigate, "
                "a pending database migration, and a config issue to diagnose. "
                "Not all failures are blocking — you must determine which are "
                "real bugs and which are test environment issues."
            ),
            goal=(
                "Deploy all 3 services to production. Fix any config errors. "
                "Run required migrations. Achieve system health above 90%."
            ),
            max_steps=20,
        )

    def setup(self, engine):
        from server.pipeline_engine import ServiceState

        engine.services["database-primary"] = ServiceState(
            name="database-primary",
            version="v5.2.0",
            health=ServiceHealth.HEALTHY,
            config={
                "max_connections": "50",
                "replication_lag_ms": "0",
                "shared_buffers": "4GB",
                "wal_level": "replica",
            },
            dependencies=[],
            latency_ms=30.0,
            error_rate=0.5,
            cpu=40.0,
            memory=60.0,
        )
        engine.services["database-primary"].prod_deployed = True
        engine.services["database-primary"].logs = [
            "2026-04-01T10:00:00.001Z INFO  [database-primary] PostgreSQL 15.4 started. max_connections=50. shared_buffers=4GB.",
            "2026-04-01T10:00:00.112Z INFO  [database-primary] Replication lag: 0ms. WAL level: replica. All replicas in sync.",
            "2026-04-01T10:00:00.334Z INFO  [database-primary] Connection pool: 15/50 active. Query throughput: 1.4k/s.",
        ]

        engine.services["auth-service"] = ServiceState(
            name="auth-service",
            version="v3.1.0",
            health=ServiceHealth.HEALTHY,
            config={
                "token_ttl_seconds": "3600",
                "jwt_algorithm": "RS256",
                "rate_limit_per_minute": "1000",
                "cert_expiry": "2026-12-01",
            },
            dependencies=["database-primary"],
            latency_ms=15.0,
            error_rate=0.2,
            cpu=28.0,
            memory=40.0,
        )
        engine.services["auth-service"].prod_deployed = True
        engine.services["auth-service"].logs = [
            "2026-04-01T10:00:00.501Z INFO  [auth-service] OAuth2 provider started. Algorithm: RS256. TTL: 3600s.",
            "2026-04-01T10:00:00.612Z INFO  [auth-service] Certificate valid until 2026-12-01. Rate limit: 1000/min.",
        ]

        engine.services["api-gateway"] = ServiceState(
            name="api-gateway",
            version="v2.3.0",
            health=ServiceHealth.HEALTHY,
            config={
                "database.pool_size": "20",
                "cache.ttl": "300",
                "log.level": "info",
            },
            dependencies=["database-primary", "auth-service"],
            latency_ms=50.0,
            error_rate=0.2,
            cpu=40.0,
            memory=45.0,
        )
        engine.services["api-gateway"].target_version = "v2.3.1"
        engine.services["api-gateway"].logs = [
            "2026-04-01T10:00:01.112Z INFO  [api-gateway] com.pipeline.core.Bootstrap — Service api-gateway started on port 8080 (pid=2201)",
            "2026-04-01T10:00:02.334Z INFO  [api-gateway] com.pipeline.db.ConnectionPool — Pool initialized: 20/20 connections available to postgres-primary.internal:5432",
            "2026-04-01T10:00:03.001Z DEBUG [api-gateway] com.pipeline.gc.GCMonitor — G1 GC pause: 11ms (Young), heap: 389MB/1024MB. Within acceptable thresholds.",
            "2026-04-01T10:00:04.220Z WARN  [api-gateway] com.pipeline.api.DeprecationFilter — Endpoint /api/v1/users is deprecated since v2.2.0. Removal scheduled for v3.0. 3 requests in last 24h from legacy-client-ios.",
            # --- Test failures ---
            "2026-04-01T10:05:00.441Z ERROR [test-runner] tests.integration.PaymentServiceTest#test_payment_timeout",
            "    java.util.concurrent.TimeoutException: Timed out after 30000ms waiting for payment-service response",
            "        at com.pipeline.client.HttpClient.execute(HttpClient.java:142)",
            "        at com.pipeline.payment.PaymentGateway.processPayment(PaymentGateway.java:87)",
            "        at tests.integration.PaymentServiceTest.test_payment_timeout(PaymentServiceTest.java:45)",
            "    --- Flaky test history: FAIL (run 1), PASS (run 2), PASS (run 3) — intermittent, likely CI network latency to payment-service-staging.internal ---",
            "2026-04-01T10:05:01.223Z ERROR [test-runner] tests.integration.PaymentServiceTest#test_payment_retry",
            "    java.util.concurrent.TimeoutException: Timed out after 30000ms waiting for payment-service response",
            "        at com.pipeline.client.HttpClient.execute(HttpClient.java:142)",
            "        at com.pipeline.payment.PaymentGateway.retryPayment(PaymentGateway.java:103)",
            "        at tests.integration.PaymentServiceTest.test_payment_retry(PaymentServiceTest.java:62)",
            "    --- Flaky test history: FAIL (run 1), FAIL (run 2), PASS (run 3) — intermittent, same root cause as test_payment_timeout ---",
            "2026-04-01T10:05:02.887Z ERROR [test-runner] tests.regression.V1EndpointTest#test_deprecated_endpoint_v1",
            "    java.lang.AssertionError: expected status 200, got 404",
            "    GET /api/v1/users returned HTTP 404 Not Found",
            "        at tests.regression.V1EndpointTest.test_deprecated_endpoint_v1(V1EndpointTest.java:28)",
            "    --- Note: /api/v1/users was removed in v2.3.0 per deprecation schedule. Test is outdated. See JIRA-4521. ---",
            "2026-04-01T10:05:03.100Z INFO  [test-runner] Test summary: 142 passed, 3 failed (2 intermittent, 1 outdated regression test)",
            # --- Migration warning ---
            "2026-04-01T10:06:00.012Z WARN  [api-gateway] com.pipeline.db.SchemaValidator — Schema version mismatch: expected v15, found v14. Migration add_index_users_email is pending.",
            "2026-04-01T10:06:00.334Z ERROR [api-gateway] com.pipeline.db.QueryExecutor — Query on users table without email index: SELECT * FROM users WHERE email = ? — sequential scan on 2.4M rows, p95 latency 890ms",
        ]

        engine.services["web-frontend"] = ServiceState(
            name="web-frontend",
            version="v1.8.0",
            health=ServiceHealth.HEALTHY,
            config={
                "api.endpoint": "https://api.internal:8080",
                "cdn.enabled": "true",
                "log.level": "info",
            },
            dependencies=["api-gateway", "auth-service"],
            latency_ms=30.0,
            error_rate=0.05,
            cpu=28.0,
            memory=35.0,
        )
        engine.services["web-frontend"].target_version = "v1.9.0"
        engine.services["web-frontend"].logs = [
            "2026-04-01T10:00:01.887Z INFO  [web-frontend] server.js:42 — Express server listening on 0.0.0.0:3000 (worker pid=1102)",
            "2026-04-01T10:00:02.101Z INFO  [web-frontend] health/probe.js:25 — Health check: OK. Upstream api-gateway reachable at api.internal:8080.",
            "2026-04-01T10:00:03.445Z DEBUG [web-frontend] middleware/metrics.js:67 — Request histogram: p50=11ms, p95=26ms, p99=41ms. Baseline nominal.",
            "2026-04-01T10:00:04.112Z INFO  [web-frontend] cdn/warmup.js:118 — Asset bundle size: 1.9MB (gzip). Threshold: 5MB — OK.",
        ]

        engine.services["cache-service"] = ServiceState(
            name="cache-service",
            version="v1.2.0",
            health=ServiceHealth.HEALTHY,
            config={
                "redis.host": "redis-staging.internal:6379",
                "redis.max_connections": "50",
                "redis.timeout": "5000",
            },
            dependencies=["database-primary"],
            latency_ms=10.0,
            error_rate=0.0,
            cpu=15.0,
            memory=20.0,
        )
        engine.services["cache-service"].target_version = "v1.2.1"
        engine.services["cache-service"].logs = [
            "2026-04-01T10:00:01.334Z INFO  [cache-service] com.pipeline.cache.RedisConnector — Connecting to Redis at redis-staging.internal:6379...",
            "2026-04-01T10:00:01.887Z INFO  [cache-service] com.pipeline.cache.RedisConnector — Redis connection established. Pool: 10/50 connections. Latency: 2ms.",
            "2026-04-01T10:00:02.112Z DEBUG [cache-service] com.pipeline.gc.GCMonitor — GC pause: 3ms (Young), heap: 128MB/512MB. Within acceptable thresholds.",
            "2026-04-01T10:00:03.445Z INFO  [cache-service] com.pipeline.cache.EvictionPolicy — LRU eviction policy active. Cache hit rate: 94.2% (last 1h). Capacity: 340MB/512MB.",
            "2026-04-01T10:00:04.001Z WARN  [cache-service] com.pipeline.cache.RedisConnector — DNS resolution for redis-staging.internal took 45ms (threshold: 10ms). Possible DNS cache miss.",
        ]

        engine.commit_sha = "d4e5f6g"
        engine.triggered_by = "ci-pipeline"
        engine.test_pass = 142
        engine.test_fail = 3
        engine.build_logs = (
            "Build succeeded. 142/145 tests passed, 3 failed.\n"
            "FAIL: tests.integration.PaymentServiceTest#test_payment_timeout (TimeoutException after 30000ms)\n"
            "FAIL: tests.integration.PaymentServiceTest#test_payment_retry (TimeoutException after 30000ms)\n"
            "FAIL: tests.regression.V1EndpointTest#test_deprecated_endpoint_v1 (HTTP 404, expected 200)"
        )

        engine.migrations_pending = ["add_index_users_email"]
        engine._time_pressure = True

    def migration_blocks_deploy(self, service_name):
        return service_name == "api-gateway"

    def run_migration(self, engine, migration_name):
        if migration_name == "add_index_users_email":
            return True
        return False

    def check_config_error(self, service_name, config):
        if service_name == "cache-service":
            return config.get("redis.host") == "redis-staging.internal:6379"
        return False


class JudgmentCallScenario(Scenario):
    """Task 3: Hard — production incident, ambiguous best path.

    Complexity that makes this genuinely hard:
    - Time pressure: only 12 steps, health degrades each step
    - Hotfix trap: v2.3.2 fixes the query but breaks web-frontend auth
      (refactored auth middleware changes API contract)
    - Agent must either: rollback (safe but loses features), deploy hotfix
      AND fix web-frontend config, or find another path
    - Multiple services affected, cascading consequences
    """

    def __init__(self):
        super().__init__(
            task_name="judgment_call",
            task_description=(
                "PRODUCTION INCIDENT. api-gateway is severely degraded "
                "(1500ms latency, 12 errors/sec). A hotfix v2.3.2 is available "
                "but only smoke-tested — it includes an auth middleware refactor "
                "that may break web-frontend. Revenue is bleeding at $500/min. "
                "You have limited time to resolve this. Health degrades every step. "
                "Note: the hotfix v2.3.2 switches authentication from HS256 to RS256. "
                "If deploying the hotfix, you may need to update web-frontend's "
                "auth configuration for compatibility."
            ),
            goal=(
                "Restore api-gateway to healthy state (latency <100ms, "
                "error rate <1/s). Keep web-frontend healthy. "
                "Resolve the incident. Consider all side effects of your chosen approach."
            ),
            max_steps=12,
        )
        self._hotfix_deployed = False

    def setup(self, engine):
        from server.pipeline_engine import ServiceState

        engine.services["database-primary"] = ServiceState(
            name="database-primary",
            version="v5.2.0",
            health=ServiceHealth.HEALTHY,
            config={
                "max_connections": "50",
                "replication_lag_ms": "0",
                "shared_buffers": "4GB",
                "wal_level": "replica",
            },
            dependencies=[],
            latency_ms=120.0,
            error_rate=1.5,
            cpu=72.0,
            memory=60.0,
        )
        engine.services["database-primary"].prod_deployed = True
        engine.services["database-primary"].logs = [
            "2026-04-01T11:44:50.001Z INFO  [database-primary] PostgreSQL 15.4 started. max_connections=50. shared_buffers=4GB.",
            "2026-04-01T11:44:50.112Z INFO  [database-primary] Replication lag: 0ms. WAL level: replica. All replicas in sync.",
            "2026-04-01T11:44:50.334Z WARN  [database-primary] Connection pool utilization rising: 38/50 active. CPU at 72%.",
            "2026-04-01T11:44:51.001Z INFO  [database-primary] Slow query log: SELECT * FROM sessions WHERE expires_at < NOW() -- 850ms.",
        ]

        engine.services["auth-service"] = ServiceState(
            name="auth-service",
            version="v3.1.0",
            health=ServiceHealth.HEALTHY,
            config={
                "token_ttl_seconds": "3600",
                "jwt_algorithm": "RS256",
                "rate_limit_per_minute": "1000",
                "cert_expiry": "2026-12-01",
            },
            dependencies=["database-primary"],
            latency_ms=20.0,
            error_rate=0.3,
            cpu=35.0,
            memory=40.0,
        )
        engine.services["auth-service"].prod_deployed = True
        engine.services["auth-service"].logs = [
            "2026-04-01T11:44:50.001Z INFO  [auth-service] OAuth2 provider running. Algorithm: RS256. TTL: 3600s.",
            "2026-04-01T11:44:51.334Z WARN  [auth-service] Hotfix v2.3.2 switches api-gateway auth middleware from HS256 to RS256. Downstream services must update token validation.",
        ]

        engine.services["api-gateway"] = ServiceState(
            name="api-gateway",
            version="v2.3.1",
            health=ServiceHealth.DEGRADED,
            config={
                "database.pool_size": "20",
                "database.query_timeout": "30000",
                "cache.ttl": "300",
                "log.level": "warn",
            },
            dependencies=["database-primary", "auth-service"],
            latency_ms=1500.0,
            error_rate=12.0,
            cpu=85.0,
            memory=78.0,
        )
        engine.services["api-gateway"].target_version = "v2.3.2"
        engine.services["api-gateway"].prod_deployed = True
        engine.services["api-gateway"].logs = [
            "2026-04-01T11:44:58.112Z ERROR [api-gateway] com.pipeline.db.QueryExecutor — Slow query: SELECT u.id, u.email FROM users u WHERE u.email LIKE '%@example.com' — 12.3s, 2,412,847 rows scanned (Sequential Scan, no index on users.email)",
            "2026-04-01T11:45:00.887Z ERROR [api-gateway] com.pipeline.server.RequestHandler — Request timeout: GET /api/v2/users — 5001ms (limit: 5000ms) at RequestHandler.handle(RequestHandler.java:89)",
            "2026-04-01T11:45:01.445Z WARN  [api-gateway] com.pipeline.db.ConnectionPool — Connection pool exhausted: 20/20 connections busy. 3 requests queued. Oldest connection held for 11.2s.",
            "2026-04-01T11:45:02.112Z ERROR [api-gateway] com.pipeline.server.ErrorHandler — HTTP 503 Service Unavailable returned to client (request_id=req-8f2a-4c1b). Upstream timeout.",
            "2026-04-01T11:45:03.001Z DEBUG [api-gateway] com.pipeline.gc.GCMonitor — G1 GC pause: 142ms (Full GC), heap: 980MB/1024MB. CRITICAL: approaching OOM.",
            "2026-04-01T11:45:30.200Z INFO  [api-gateway] com.pipeline.deploy.ArtifactRegistry — Hotfix v2.3.2 available. Commits: a8f2c1e add_email_index (CREATE INDEX on users.email), b3d9e7f refactor_auth (HS256->RS256 JWT), c1a4b8d update_error_msgs (RFC 7807)",
            "2026-04-01T11:45:30.887Z INFO  [api-gateway] com.pipeline.test.SmokeRunner — v2.3.2 smoke tests: 5/5 passed (health, auth, query, cache, metrics). Full regression suite NOT executed (est. 45min).",
            "2026-04-01T11:45:31.112Z WARN  [api-gateway] com.pipeline.deploy.CompatChecker — BREAKING CHANGE in v2.3.2: refactor_auth changes JWT from HS256 (/auth/validate) to RS256 (/auth/v2/validate). Downstream services with api.auth_version=v1 will get 401 Unauthorized. Affected: web-frontend.",
            "2026-04-01T11:46:00.334Z INFO  [api-gateway] com.pipeline.deploy.RollbackManager — Rollback target: v2.3.0 (tag: release-2.3.0, last deployed 2026-03-28). WARNING: v2.3.0 does not include /api/v2/user-preferences endpoint added in v2.3.1.",
            "2026-04-01T11:46:01.001Z WARN  [api-gateway] com.pipeline.metrics.AlertManager — Revenue impact: 2,847 failed requests in 5min * $0.87 avg = ~$500/min estimated loss.",
        ]

        engine.services["web-frontend"] = ServiceState(
            name="web-frontend",
            version="v1.9.0",
            health=ServiceHealth.HEALTHY,
            config={
                "api.endpoint": "https://api.internal:8080",
                "api.auth_version": "v1",
                "cdn.enabled": "true",
                "log.level": "info",
            },
            dependencies=["api-gateway", "auth-service"],
            latency_ms=30.0,
            error_rate=0.5,
            cpu=28.0,
            memory=35.0,
        )
        engine.services["web-frontend"].prod_deployed = True
        engine.services["web-frontend"].logs = [
            "2026-04-01T11:44:59.887Z WARN  [web-frontend] middleware/upstream.js:34 — Upstream api-gateway latency spike: p95=1847ms (baseline: 45ms). Circuit breaker at 80% threshold.",
            "2026-04-01T11:45:05.112Z ERROR [web-frontend] services/api-client.js:89 — API call failed:",
            "    GET https://api.internal:8080/api/v2/users?email=user@test.com",
            "    Error: ETIMEDOUT after 5000ms",
            "    at TcpSocket.connect (net.js:1141:16)",
            "    at ApiClient.request (services/api-client.js:67:12)",
            "    at UserController.getProfile (controllers/user.js:43:28)",
            "2026-04-01T11:45:06.001Z DEBUG [web-frontend] middleware/metrics.js:67 — Request histogram degraded: p50=340ms, p95=4200ms, p99=5001ms (timeout). Baseline was p50=12ms.",
            "2026-04-01T11:45:07.445Z WARN  [web-frontend] config/auth.js:15 — Current auth config: api.auth_version=v1 (HS256 via /auth/validate). If api-gateway upgrades to v2.3.2, this must change to v2 (RS256 via /auth/v2/validate).",
            "2026-04-01T11:46:00.112Z INFO  [web-frontend] deploy/staging.js:28 — v1.9.1 deployed to staging 20 minutes ago. Depends on api-gateway /api/v2/user-preferences endpoint (added in v2.3.1).",
        ]

        engine.commit_sha = "h7i8j9k"
        engine.triggered_by = "incident-response"
        engine.test_pass = 5
        engine.test_fail = 0
        engine.build_logs = (
            "Hotfix build. Smoke tests only: 5/5 passed.\n"
            "WARNING: Full test suite was NOT executed.\n"
            "Commits in v2.3.2: add_email_index, refactor_auth, update_error_msgs\n"
            "NOTICE: refactor_auth changes JWT validation endpoint — downstream "
            "services using api.auth_version=v1 will get 401 errors"
        )

        engine.alerts = [
            AlertInfo(
                severity="critical",
                message="api-gateway p95 latency >1000ms for 15 minutes. SLO violation: 99.9% availability target breached.",
                service_name="api-gateway",
                timestamp="2026-04-01T11:45:00Z",
            ),
            AlertInfo(
                severity="warning",
                message="Revenue impact estimated $500/min based on failed request rate (2847 req/5min * $0.87 avg)",
                service_name="api-gateway",
                timestamp="2026-04-01T11:46:00Z",
            ),
            AlertInfo(
                severity="warning",
                message="api-gateway health degrading — latency increasing ~200ms/min. Connection pool saturation at 100%.",
                service_name="api-gateway",
                timestamp="2026-04-01T11:47:00Z",
            ),
        ]

        # Enable time-pressure: health degrades each step
        engine._time_pressure = True

    def on_prod_deploy(self, engine, service_name, version):
        """When api-gateway v2.3.2 hits prod, web-frontend breaks due to auth refactor."""
        if service_name == "api-gateway" and version == "v2.3.2":
            self._hotfix_deployed = True
            web_fe = engine.services.get("web-frontend")
            if web_fe and web_fe.config.get("api.auth_version") == "v1":
                web_fe.health = ServiceHealth.DEGRADED
                web_fe.error_rate = 20.0
                web_fe.latency_ms = 800.0
                web_fe.logs.append(
                    "2026-04-01T11:50:01.887Z ERROR [web-frontend] services/auth-client.js:44 — Authentication failed:\n"
                    "    POST https://api.internal:8080/auth/validate\n"
                    "    HTTP 401 Unauthorized: {\"error\": \"invalid_algorithm\", \"expected\": \"RS256\", \"received\": \"HS256\"}\n"
                    "    at AuthClient.validateToken (services/auth-client.js:38:16)\n"
                    "    at SessionMiddleware.authenticate (middleware/session.js:22:30)\n"
                    "    Config: api.auth_version=v1 — must be updated to v2 for compatibility with api-gateway v2.3.2 RS256 JWT validation."
                )
                return (
                    "WARNING: web-frontend is now returning 401 errors. "
                    "api-gateway v2.3.2 auth middleware is incompatible with "
                    "web-frontend api.auth_version=v1."
                )
        return None

    def on_rollback(self, engine, service_name):
        """Rollback api-gateway to v2.3.0 loses v2.3.1 API changes web-frontend depends on."""
        if service_name == "api-gateway":
            web_fe = engine.services.get("web-frontend")
            if web_fe:
                web_fe.health = ServiceHealth.DEGRADED
                web_fe.error_rate = 8.0
                web_fe.latency_ms = 400.0
                web_fe.logs.append(
                    "2026-04-01T11:52:00.112Z ERROR [web-frontend] controllers/user.js:58 — Upstream endpoint missing:\n"
                    "    GET https://api.internal:8080/api/v2/user-preferences\n"
                    "    HTTP 404 Not Found\n"
                    "    at UserController.getPreferences (controllers/user.js:52:16)\n"
                    "    at Router.dispatch (node_modules/express/lib/router/index.js:281:12)\n"
                    "    Note: /api/v2/user-preferences was added in v2.3.1 but api-gateway has been rolled back to v2.3.0.\n"
                    "    Impact: User settings page returns 500 error. 404 errors on ~12% of page loads."
                )
            # Also stop time pressure since api-gateway is fixed
            engine._time_pressure = False

    def check_config_error(self, service_name, config):
        """After hotfix v2.3.2 deploys, web-frontend needs api.auth_version=v2."""
        if service_name == "web-frontend" and self._hotfix_deployed:
            return config.get("api.auth_version") == "v1"
        return False


class CascadingFailureScenario(Scenario):
    """Task 4: Medium-Hard — root cause analysis across a dependency chain.

    cache-service is the root cause (down, config error).
    api-gateway depends on cache-service → degrading from cascade.
    web-frontend depends on api-gateway → starting to degrade.
    Agent must fix the root cause FIRST, or fixes to downstream services
    will be undone by continued cascading.
    """

    def __init__(self):
        super().__init__(
            task_name="cascading_failure",
            task_description=(
                "CASCADING FAILURE. cache-service is down and dragging api-gateway "
                "and web-frontend down with it. You must identify the root cause "
                "and fix services in the correct order — fixing downstream services "
                "first won't help while the root cause persists."
            ),
            goal=(
                "Restore all services to healthy. Fix the root cause (cache-service) "
                "first, then recover downstream services. Deploy all services to "
                "production. System health above 90%."
            ),
            max_steps=15,
        )

    def setup(self, engine):
        from server.pipeline_engine import ServiceState

        engine.services["database-primary"] = ServiceState(
            name="database-primary",
            version="v5.2.0",
            health=ServiceHealth.HEALTHY,
            config={
                "max_connections": "50",
                "replication_lag_ms": "0",
                "shared_buffers": "4GB",
                "wal_level": "replica",
            },
            dependencies=[],
            latency_ms=20.0,
            error_rate=0.3,
            cpu=35.0,
            memory=60.0,
        )
        engine.services["database-primary"].prod_deployed = True
        engine.services["database-primary"].logs = [
            "2026-04-01T12:00:00.001Z INFO  [database-primary] PostgreSQL 15.4 started. max_connections=50. shared_buffers=4GB.",
            "2026-04-01T12:00:00.112Z INFO  [database-primary] Replication lag: 0ms. WAL level: replica. All replicas in sync.",
            "2026-04-01T12:00:00.334Z INFO  [database-primary] Connection pool: 14/50 active. Query throughput: 1.1k/s.",
        ]

        # ROOT CAUSE: cache-service is degraded with a config error
        engine.services["cache-service"] = ServiceState(
            name="cache-service",
            version="v1.2.0",
            health=ServiceHealth.DEGRADED,
            config={
                "redis.host": "redis-prod.internal:6379",
                "redis.max_connections": "5",
                "redis.timeout": "5000",
            },
            dependencies=["database-primary"],
            latency_ms=2000.0,
            error_rate=25.0,
            cpu=8.0,
            memory=95.0,
        )
        engine.services["cache-service"].target_version = "v1.2.1"
        engine.services["cache-service"].prod_deployed = True
        engine.services["cache-service"].logs = [
            "2026-04-01T12:00:01.112Z INFO  [cache-service] com.pipeline.cache.RedisConnector — Connecting to Redis at redis-prod.internal:6379...",
            "2026-04-01T12:00:02.887Z ERROR [cache-service] com.pipeline.cache.RedisConnector — Connection pool exhausted: 5/5 connections in use. Cannot allocate new connection.",
            "    redis.exceptions.ConnectionError: max_connections (5) reached — pool is full",
            "    at com.pipeline.cache.RedisConnector.acquire(RedisConnector.java:89)",
            "    at com.pipeline.cache.CacheManager.get(CacheManager.java:45)",
            "2026-04-01T12:00:03.445Z ERROR [cache-service] com.pipeline.cache.CacheManager — Cache read timeout: GET user:session:8f2a4c1b — waited 5000ms, pool has 0 available connections. All 5 slots occupied by long-running queries.",
            "2026-04-01T12:00:04.001Z WARN  [cache-service] com.pipeline.cache.EvictionPolicy — LRU eviction stalled. Cache hit rate dropped to 12.3% (was 94.2%). Memory: 490MB/512MB — near capacity.",
            "2026-04-01T12:00:05.112Z WARN  [cache-service] com.pipeline.health.HealthCheck — Readiness probe DEGRADED. Connection pool saturated. Response time 2000ms (threshold: 500ms).",
            "2026-04-01T12:00:05.334Z DEBUG [cache-service] com.pipeline.gc.GCMonitor — GC pause: 89ms (Full GC), heap: 480MB/512MB. Memory pressure from connection pool backlog.",
            "2026-04-01T12:00:06.001Z INFO  [cache-service] com.pipeline.ops.ConfigAdvisor — Recommended: increase redis.max_connections from 5 to at least 50 for production workload. Current setting is development default.",
        ]

        engine.services["auth-service"] = ServiceState(
            name="auth-service",
            version="v3.1.0",
            health=ServiceHealth.HEALTHY,
            config={
                "token_ttl_seconds": "3600",
                "jwt_algorithm": "RS256",
                "rate_limit_per_minute": "1000",
                "cert_expiry": "2026-12-01",
            },
            dependencies=["database-primary"],
            latency_ms=12.0,
            error_rate=0.1,
            cpu=25.0,
            memory=40.0,
        )
        engine.services["auth-service"].prod_deployed = True
        engine.services["auth-service"].logs = [
            "2026-04-01T12:00:00.501Z INFO  [auth-service] OAuth2 provider started. Algorithm: RS256. TTL: 3600s.",
            "2026-04-01T12:00:00.612Z INFO  [auth-service] Certificate valid until 2026-12-01. Rate limit: 1000/min.",
        ]

        # DOWNSTREAM 1: api-gateway depends on cache-service, degrading
        engine.services["api-gateway"] = ServiceState(
            name="api-gateway",
            version="v2.3.0",
            health=ServiceHealth.DEGRADED,
            config={
                "database.pool_size": "20",
                "cache.ttl": "300",
                "cache.backend": "redis",
                "log.level": "info",
            },
            dependencies=["cache-service", "database-primary", "auth-service"],
            latency_ms=300.0,
            error_rate=5.0,
            cpu=60.0,
            memory=55.0,
        )
        engine.services["api-gateway"].target_version = "v2.3.1"
        engine.services["api-gateway"].prod_deployed = True
        engine.services["api-gateway"].logs = [
            "2026-04-01T12:00:10.112Z INFO  [api-gateway] com.pipeline.core.Bootstrap — Service api-gateway running on port 8080 (pid=3301)",
            "2026-04-01T12:00:11.445Z ERROR [api-gateway] com.pipeline.cache.CacheClient — Cache lookup failed: GET user:profile:12345 — upstream cache-service returned 503 Service Unavailable",
            "    at com.pipeline.cache.CacheClient.get(CacheClient.java:67)",
            "    at com.pipeline.api.UserController.getProfile(UserController.java:43)",
            "2026-04-01T12:00:12.001Z WARN  [api-gateway] com.pipeline.api.FallbackHandler — Falling back to database for cache misses. DB query latency 420ms (normal: 15ms with cache). Cache-service dependency unhealthy.",
            "2026-04-01T12:00:13.334Z ERROR [api-gateway] com.pipeline.db.ConnectionPool — Pool saturation: 18/20 connections busy. Cache fallback is overloading database.",
            "2026-04-01T12:00:14.001Z DEBUG [api-gateway] com.pipeline.gc.GCMonitor — G1 GC pause: 45ms (Young), heap: 650MB/1024MB. Elevated from cache fallback memory overhead.",
            "2026-04-01T12:00:15.112Z WARN  [api-gateway] com.pipeline.metrics.HealthReporter — Service health DEGRADED. Root cause: upstream cache-service is DOWN. Latency elevated due to database fallback.",
        ]

        # DOWNSTREAM 2: web-frontend depends on api-gateway, starting to feel it
        engine.services["web-frontend"] = ServiceState(
            name="web-frontend",
            version="v1.8.0",
            health=ServiceHealth.HEALTHY,
            config={
                "api.endpoint": "https://api.internal:8080",
                "cdn.enabled": "true",
                "log.level": "info",
            },
            dependencies=["api-gateway", "auth-service"],
            latency_ms=80.0,
            error_rate=0.5,
            cpu=30.0,
            memory=35.0,
        )
        engine.services["web-frontend"].target_version = "v1.9.0"
        engine.services["web-frontend"].prod_deployed = True
        engine.services["web-frontend"].logs = [
            "2026-04-01T12:00:20.112Z INFO  [web-frontend] server.js:42 — Express server running on 0.0.0.0:3000 (worker pid=1501)",
            "2026-04-01T12:00:21.445Z WARN  [web-frontend] middleware/upstream.js:34 — Upstream api-gateway latency elevated: p95=420ms (baseline: 45ms). Not yet at circuit breaker threshold.",
            "2026-04-01T12:00:22.001Z ERROR [web-frontend] services/api-client.js:89 — Intermittent failures from api-gateway: 3 timeouts in last minute. Error rate 1.2/s (baseline: 0.05/s).",
            "2026-04-01T12:00:23.334Z DEBUG [web-frontend] middleware/metrics.js:67 — Request histogram: p50=85ms, p95=380ms, p99=1200ms. Baseline was p50=12ms.",
            "2026-04-01T12:00:24.001Z INFO  [web-frontend] cdn/warmup.js:118 — Asset bundle size: 2.1MB (gzip). Threshold: 5MB — OK.",
        ]

        engine.commit_sha = "k9l0m1n"
        engine.triggered_by = "monitoring-alert"
        engine.test_pass = 145
        engine.test_fail = 0
        engine.build_logs = "Build succeeded. 145/145 tests passed."

        engine.alerts = [
            AlertInfo(
                severity="critical",
                message="cache-service DEGRADED — connection pool saturated (5/5 connections). All dependent services affected.",
                service_name="cache-service",
                timestamp="2026-04-01T12:00:05Z",
            ),
            AlertInfo(
                severity="warning",
                message="api-gateway DEGRADED — falling back to database due to cache-service outage. Latency 10x baseline.",
                service_name="api-gateway",
                timestamp="2026-04-01T12:00:15Z",
            ),
            AlertInfo(
                severity="info",
                message="web-frontend experiencing elevated error rates from api-gateway degradation. User-facing impact beginning.",
                service_name="web-frontend",
                timestamp="2026-04-01T12:00:22Z",
            ),
        ]

    def check_config_error(self, service_name, config):
        """cache-service has max_connections=5 (dev default, should be 50 for prod)."""
        if service_name == "cache-service":
            return config.get("redis.max_connections") == "5"
        return False


class CapacityCrisisScenario(Scenario):
    """Task 5: Medium-Hard — capacity crisis, prevent collapse under 4x traffic.

    database-primary connection pool nearly full (47/50). api-gateway degraded.
    Agent must increase capacity at the bottleneck before tipping points trigger
    cascading failures. Inaction = death spiral.
    """

    def __init__(self):
        super().__init__(
            task_name="capacity_crisis",
            task_description=(
                "CAPACITY CRISIS. Peak traffic is 4x normal. database-primary "
                "connection pool is nearly full (47/50). api-gateway is degraded "
                "under load. Systems are straining but nothing has failed yet. "
                "You must stabilize the system before services start collapsing. "
                "Every step of inaction risks tipping points."
            ),
            goal=(
                "Prevent system collapse. Keep all services above critical health. "
                "Address the root bottleneck (database connection capacity). "
                "Stabilize api-gateway."
            ),
            max_steps=15,
        )

    def setup(self, engine):
        from server.pipeline_engine import ServiceState

        engine.services["database-primary"] = ServiceState(
            name="database-primary",
            version="v5.2.0",
            health=ServiceHealth.HEALTHY,
            config={
                "max_connections": "50",
                "replication_lag_ms": "0",
                "shared_buffers": "4GB",
                "wal_level": "replica",
            },
            dependencies=[],
            latency_ms=180.0,
            error_rate=3.2,
            cpu=60.0,
            memory=70.0,
        )
        engine.services["database-primary"].prod_deployed = True
        engine.services["database-primary"].logs = [
            "2026-04-01T16:30:01.001Z WARN  [database-primary] Connection pool utilization: 94% (47/50 active). Queue depth: 23 pending requests.",
            "2026-04-01T16:30:02.112Z WARN  [database-primary] Slow query detected: SELECT * FROM orders WHERE user_id IN (SELECT...) -- 1.2s execution time. Sequential scan on orders table (2.4M rows).",
            "2026-04-01T16:30:03.334Z INFO  [database-primary] Autovacuum running on users table. Last vacuum: 6 hours ago.",
        ]

        engine.services["auth-service"] = ServiceState(
            name="auth-service",
            version="v3.1.0",
            health=ServiceHealth.HEALTHY,
            config={
                "token_ttl_seconds": "3600",
                "jwt_algorithm": "RS256",
                "rate_limit_per_minute": "1000",
                "cert_expiry": "2026-12-01",
            },
            dependencies=["database-primary"],
            latency_ms=50.0,
            error_rate=0.8,
            cpu=45.0,
            memory=40.0,
        )
        engine.services["auth-service"].prod_deployed = True
        engine.services["auth-service"].logs = [
            "2026-04-01T16:30:00.501Z INFO  [auth-service] OAuth2 provider running. Algorithm: RS256. TTL: 3600s.",
            "2026-04-01T16:30:01.112Z INFO  [auth-service] Rate limiter: 850/1000 requests per minute. Headroom: 15%.",
        ]

        engine.services["api-gateway"] = ServiceState(
            name="api-gateway",
            version="v2.3.1",
            health=ServiceHealth.DEGRADED,
            config={
                "database.pool_size": "20",
                "cache.ttl": "300",
                "log.level": "warn",
            },
            dependencies=["database-primary", "auth-service"],
            latency_ms=450.0,
            error_rate=5.1,
            cpu=82.0,
            memory=65.0,
        )
        engine.services["api-gateway"].prod_deployed = True
        engine.services["api-gateway"].logs = [
            "2026-04-01T16:30:05.001Z WARN  [api-gateway] Request queue depth: 156. Average response time: 450ms (threshold: 200ms).",
            "2026-04-01T16:30:06.112Z ERROR [api-gateway] Upstream timeout: database-primary did not respond within 3000ms. Retry 2/3.",
            "2026-04-01T16:30:07.334Z INFO  [api-gateway] Rate limiter: 82% of capacity. Non-critical endpoints: /api/v1/recommendations, /api/v1/analytics",
        ]

        engine.services["cache-service"] = ServiceState(
            name="cache-service",
            version="v1.2.0",
            health=ServiceHealth.HEALTHY,
            config={
                "redis.host": "redis-prod.internal:6379",
                "redis.max_connections": "50",
                "redis.timeout": "5000",
                "eviction_policy": "lru",
                "max_memory": "2GB",
            },
            dependencies=["database-primary"],
            latency_ms=80.0,
            error_rate=1.2,
            cpu=55.0,
            memory=50.0,
        )
        engine.services["cache-service"].prod_deployed = True
        engine.services["cache-service"].logs = [
            "2026-04-01T16:30:01.334Z INFO  [cache-service] Redis connection pool: 35/50 connections. Hit rate: 78.3%.",
            "2026-04-01T16:30:02.112Z WARN  [cache-service] Cache miss rate elevated: 21.7% (baseline: 6%). More queries falling through to database-primary.",
            "2026-04-01T16:30:03.001Z DEBUG [cache-service] GC pause: 12ms (Young), heap: 380MB/512MB. Within acceptable thresholds.",
        ]

        engine.services["web-frontend"] = ServiceState(
            name="web-frontend",
            version="v1.9.0",
            health=ServiceHealth.HEALTHY,
            config={
                "api.endpoint": "https://api.internal:8080",
                "cdn.enabled": "true",
                "log.level": "info",
            },
            dependencies=["api-gateway", "auth-service"],
            latency_ms=200.0,
            error_rate=2.0,
            cpu=60.0,
            memory=45.0,
        )
        engine.services["web-frontend"].prod_deployed = True
        engine.services["web-frontend"].logs = [
            "2026-04-01T16:30:10.112Z WARN  [web-frontend] Upstream api-gateway latency elevated: p95=520ms (baseline: 45ms).",
            "2026-04-01T16:30:11.445Z INFO  [web-frontend] Request histogram: p50=180ms, p95=450ms, p99=1200ms. Baseline was p50=12ms.",
            "2026-04-01T16:30:12.001Z WARN  [web-frontend] User-facing error rate: 2.0/s. Support tickets increasing.",
        ]

        engine.commit_sha = "p3q4r5s"
        engine.triggered_by = "traffic-spike-alert"
        engine.test_pass = 145
        engine.test_fail = 0
        engine.build_logs = "Build succeeded. 145/145 tests passed."

        engine.alerts = [
            AlertInfo(
                severity="critical",
                message="database-primary connection pool at 94% (47/50). Queue depth 23. Risk of connection exhaustion.",
                service_name="database-primary",
                timestamp="2026-04-01T16:30:01Z",
            ),
            AlertInfo(
                severity="warning",
                message="api-gateway DEGRADED — upstream database timeouts causing 450ms avg response time.",
                service_name="api-gateway",
                timestamp="2026-04-01T16:30:05Z",
            ),
            AlertInfo(
                severity="info",
                message="Traffic spike detected: 4x normal load. All services under increased pressure.",
                service_name="web-frontend",
                timestamp="2026-04-01T16:30:10Z",
            ),
        ]

        engine._time_pressure = True

    def check_config_error(self, service_name, config):
        """database-primary has max_connections=50 (too low for 4x traffic)."""
        if service_name == "database-primary":
            return int(config.get("max_connections", "50")) < 75
        return False


SCENARIOS = {
    "clean_deploy": CleanDeployScenario,
    "broken_pipeline": BrokenPipelineScenario,
    "judgment_call": JudgmentCallScenario,
    "cascading_failure": CascadingFailureScenario,
    "capacity_crisis": CapacityCrisisScenario,
}


def load_scenario(task_name, seed):
    """Load and return a scenario instance for the given task."""
    scenario_cls = SCENARIOS.get(task_name)
    if scenario_cls is None:
        raise ValueError(f"Unknown task: {task_name}. Available: {list(SCENARIOS.keys())}")
    return scenario_cls()
