"""Render a replay JSON (from export_replay.py) into per-step PNG frames.

Phase J.6 (win-1st add): the side-by-side video in Phase O is built from
two replay JSONs (baseline + trained, same task+seed) → two sequences of
PNG frames → optionally GIFs → composited in a video editor.

Hardcoded service layout (no networkx; matplotlib + optional PIL only):

    web-frontend                 (top, y=7)
         ↑
    api-gateway                  (middle, y=5)
       ↑     ↖
    auth-service     cache-service   (mid, y=3, side-by-side)
        ↑              ↑
       database-primary           (bottom, y=1)

Color buckets per service health (string-valued from the env's
ServiceHealth enum):
    healthy   → green   (#4CAF50)
    degraded  → yellow  (#FFC107)
    down      → red     (#F44336)
    unknown   → gray    (#9E9E9E)

Numeric health (0..100) is also accepted and bucketed with the same
thresholds as eval_baseline._extract_system_health uses for recovery:
    >= 80 → green, >= 40 → yellow, < 40 → red.

CLI:
    python training/render_replay.py \\
        --replay-json outputs/replay_trained_judgment.json \\
        --output-dir outputs/frames_trained_judgment/ \\
        --make-gif
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

logger = logging.getLogger("render_replay")

# Hardcoded layout — (x, y) center for each service node.
_LAYOUT: Dict[str, Tuple[float, float]] = {
    "web-frontend":     (5.0, 7.0),
    "api-gateway":      (5.0, 5.0),
    "auth-service":     (3.0, 3.0),
    "cache-service":    (7.0, 3.0),
    "database-primary": (5.0, 1.0),
}

# Dependency arrows: (from_service, to_service) — drawn from-tail → to-head.
# Direction follows data-flow upward: db feeds auth & cache; auth feeds api;
# cache feeds api; api feeds web-frontend.
_ARROWS: List[Tuple[str, str]] = [
    ("database-primary", "auth-service"),
    ("database-primary", "cache-service"),
    ("auth-service",     "api-gateway"),
    ("cache-service",    "api-gateway"),
    ("api-gateway",      "web-frontend"),
]

_NODE_W, _NODE_H = 1.8, 0.9

_COLOR_HEALTHY  = "#4CAF50"
_COLOR_DEGRADED = "#FFC107"
_COLOR_DOWN     = "#F44336"
_COLOR_UNKNOWN  = "#9E9E9E"
_COLOR_TEXT     = "#1a1a1a"
_COLOR_ARROW    = "#666666"


def _bucket_color(health) -> str:
    """Map a service's health value (string OR numeric) to a hex color."""
    if isinstance(health, (int, float)):
        if health >= 80: return _COLOR_HEALTHY
        if health >= 40: return _COLOR_DEGRADED
        if health > 0:   return _COLOR_DOWN
        return _COLOR_UNKNOWN
    s = str(health).lower() if health is not None else "unknown"
    return {
        "healthy":  _COLOR_HEALTHY,
        "degraded": _COLOR_DEGRADED,
        "down":     _COLOR_DOWN,
        "unknown":  _COLOR_UNKNOWN,
    }.get(s, _COLOR_UNKNOWN)


def _format_action(action: Dict[str, Any]) -> str:
    """One-line human description of a step's action."""
    parts = [action.get("action_type", "?")]
    if action.get("service_name"):
        parts.append(action["service_name"])
    if action.get("target_version"):
        parts.append(f"v={action['target_version']}")
    if action.get("migration_name"):
        parts.append(f"mig={action['migration_name']}")
    if action.get("role"):
        parts.append(f"role={action['role']}")
    return " ".join(parts)


def render_step(
    step_data: Dict[str, Any],
    replay_meta: Dict[str, Any],
    output_path: Path,
    figsize: Tuple[float, float] = (12.0, 6.0),
) -> None:
    """Draw one step as a PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 9)
    ax.set_aspect("equal")
    ax.axis("off")

    # ─── Title strip (top of canvas) ───────────────────────────────────────────
    label = "trained" if replay_meta.get("trained") else "baseline"
    title = (
        f"{label.upper()}  |  task: {replay_meta.get('task', '?')}  "
        f"seed: {replay_meta.get('seed', '?')}"
    )
    ax.text(0.0, 8.6, title, fontsize=11, fontweight="bold", color=_COLOR_TEXT)
    ax.text(
        10.0, 8.6,
        f"system_health: {step_data.get('system_health', 0):.0f}/100",
        fontsize=11, fontweight="bold", color=_COLOR_TEXT, ha="right",
    )

    # ─── Arrows (drawn first so node patches render on top) ───────────────────
    for src, dst in _ARROWS:
        if src not in _LAYOUT or dst not in _LAYOUT:
            continue
        x0, y0 = _LAYOUT[src]
        x1, y1 = _LAYOUT[dst]
        # Pull arrowhead just inside the destination node so it doesn't
        # disappear under the patch.
        dy = y1 - y0
        y1_adj = y1 - (_NODE_H / 2 + 0.05) * (1 if dy > 0 else -1)
        ax.add_patch(FancyArrowPatch(
            (x0, y0 + _NODE_H / 2 + 0.05),
            (x1, y1_adj),
            arrowstyle="-|>", mutation_scale=12,
            color=_COLOR_ARROW, linewidth=1.4, zorder=1,
        ))

    # ─── Service nodes ─────────────────────────────────────────────────────────
    services = {s.get("name"): s for s in step_data.get("services", [])}
    for name, (cx, cy) in _LAYOUT.items():
        svc = services.get(name)
        color = _bucket_color(svc.get("health") if svc else "unknown")
        ax.add_patch(FancyBboxPatch(
            (cx - _NODE_W / 2, cy - _NODE_H / 2),
            _NODE_W, _NODE_H,
            boxstyle="round,pad=0.05,rounding_size=0.18",
            facecolor=color, edgecolor=_COLOR_TEXT, linewidth=1.0, zorder=2,
        ))
        # Two-line label inside node: name + (health, cpu)
        ax.text(cx, cy + 0.18, name, fontsize=8.5, fontweight="bold",
                ha="center", va="center", color="white", zorder=3)
        if svc:
            health_label = (
                str(svc.get("health"))
                if isinstance(svc.get("health"), str)
                else f"h:{svc.get('health', 0):.0f}"
            )
            cpu = svc.get("cpu_percent", 0.0)
            ax.text(
                cx, cy - 0.22,
                f"{health_label} | cpu:{cpu:.0f}%",
                fontsize=7.5, ha="center", va="center", color="white", zorder=3,
            )
        else:
            ax.text(cx, cy - 0.22, "(no data)", fontsize=7,
                    ha="center", va="center", color="white", zorder=3)

    # ─── Action annotation strip (bottom) ──────────────────────────────────────
    step_n = step_data.get("step", "?")
    action_str = _format_action(step_data.get("action", {}))
    reward = step_data.get("reward", 0.0)
    sign = "+" if reward >= 0 else ""
    annotation = f"Step {step_n}: {action_str}   →   reward {sign}{reward:.2f}"
    if step_data.get("done"):
        annotation += "   [DONE]"
    ax.text(0.0, 0.05, annotation, fontsize=10, color=_COLOR_TEXT,
            fontfamily="monospace")

    err = step_data.get("last_action_error")
    if err:
        ax.text(0.0, -0.30, f"error: {err[:90]}",
                fontsize=8, color=_COLOR_DOWN, fontfamily="monospace")

    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def assemble_gif(frame_paths: List[Path], output_path: Path,
                 frame_duration_ms: int = 1500) -> bool:
    """Combine PNG frames into a GIF via PIL. Return True on success."""
    try:
        from PIL import Image
    except ImportError:
        logger.warning("Pillow not installed — skipping GIF assembly. "
                       "PNGs are still in the output dir.")
        return False
    if not frame_paths:
        return False
    frames = [Image.open(str(p)).convert("RGB") for p in frame_paths]
    frames[0].save(
        str(output_path),
        save_all=True,
        append_images=frames[1:],
        duration=frame_duration_ms,
        loop=0,
        optimize=True,
    )
    return True


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="Render replay JSON into per-step PNG frames (and optional GIF).")
    p.add_argument("--replay-json", required=True, help="Replay JSON from export_replay.py")
    p.add_argument("--output-dir", required=True, help="Directory for output frames (created if missing)")
    p.add_argument("--make-gif", action="store_true",
                   help="Also assemble frames into <stem>.gif via Pillow")
    p.add_argument("--frame-duration", type=int, default=1500,
                   help="Per-frame duration in ms for the assembled GIF (default 1500)")
    args = p.parse_args()

    replay = json.loads(Path(args.replay_json).read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    task = replay.get("task", "task")
    seed = replay.get("seed", "0")
    label = "trained" if replay.get("trained") else "baseline"
    base = f"replay_{label}_{task}_{seed}"

    frame_paths: List[Path] = []
    for step_data in replay.get("steps", []):
        step_n = int(step_data.get("step", 0))
        out = output_dir / f"{base}_step_{step_n:02d}.png"
        render_step(step_data, replay, out)
        frame_paths.append(out)
    logger.info("Wrote %d PNG frames into %s", len(frame_paths), output_dir)

    if args.make_gif:
        gif_path = output_dir / f"{base}.gif"
        ok = assemble_gif(frame_paths, gif_path, frame_duration_ms=args.frame_duration)
        if ok:
            logger.info("Wrote %s", gif_path)
        else:
            logger.info("GIF assembly skipped or failed; PNG frames remain.")


if __name__ == "__main__":
    main()
