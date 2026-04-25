"""Strip `handoff_notes` from all assistant action JSONs in SFT trajectories.

Phase B (models.py) removed `handoff_notes` from `PipelineAction`. The SFT
dataset still contained it in 289 of 390 assistant messages. Because the
parent `Action` class from openenv-core sets `extra="forbid"`, every call
to `PipelineAction(**action_data)` made by the GRPO reward / inference /
integration test code raises `ValidationError: extra_forbidden` when the
LLM emits an action with `handoff_notes` — which it WILL after SFT,
because the training data taught it that pattern.

This script normalises the dataset: each assistant message's content is a
single JSON object; we json.loads it, pop the offending key, and json.dumps
it back. Behavioural content (action sequences, roles, services, reasons)
is untouched.

A backup is written to data/sft_trajectories.jsonl.pre_strip.bak so the
operation is reversible.

Run from the project root:
    python scripts/strip_handoff_notes.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

INPUT_PATH = Path("data/sft_trajectories.jsonl")
BACKUP_PATH = Path("data/sft_trajectories.jsonl.pre_strip.bak")


def _strip_in_action_json(content: str) -> tuple[str, bool]:
    """Try to parse `content` as a JSON action object. If it has handoff_notes,
    drop it and return the re-serialised object. Otherwise return content
    unchanged.

    Returns (new_content, changed_bool).
    """
    try:
        obj = json.loads(content)
    except json.JSONDecodeError:
        return content, False
    if not isinstance(obj, dict):
        return content, False
    if "handoff_notes" not in obj:
        return content, False
    del obj["handoff_notes"]
    # Re-serialise compactly (matches existing single-line style); preserve
    # non-ascii (e.g. em-dashes in `reason` fields) verbatim.
    return json.dumps(obj, ensure_ascii=False), True


def main() -> int:
    if not INPUT_PATH.exists():
        print(f"ERROR: {INPUT_PATH} not found — run from project root", file=sys.stderr)
        return 2

    raw_lines = INPUT_PATH.read_text(encoding="utf-8").splitlines(keepends=True)

    BACKUP_PATH.write_text("".join(raw_lines), encoding="utf-8")
    print(f"Backup written: {BACKUP_PATH}")

    output_lines: list[str] = []
    total_records = 0
    total_asst_msgs = 0
    modified_asst_msgs = 0
    total_changes = 0
    bad_record_lines = 0

    for line in raw_lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            output_lines.append(line)
            continue
        try:
            rec = json.loads(stripped)
        except json.JSONDecodeError as e:
            print(f"WARN: bad JSON line, preserving as-is: {e}")
            output_lines.append(line)
            bad_record_lines += 1
            continue

        total_records += 1
        for msg in rec.get("messages", []):
            if msg.get("role") != "assistant":
                continue
            total_asst_msgs += 1
            new_content, changed = _strip_in_action_json(msg.get("content", ""))
            if changed:
                msg["content"] = new_content
                modified_asst_msgs += 1
                total_changes += 1

        # Preserve trailing newline shape: existing file uses LF after each record.
        output_lines.append(json.dumps(rec, ensure_ascii=False) + "\n")

    INPUT_PATH.write_text("".join(output_lines), encoding="utf-8")

    print(f"Total records processed: {total_records}")
    print(f"Total assistant messages: {total_asst_msgs}")
    print(f"Assistant messages modified: {modified_asst_msgs}")
    print(f"Total handoff_notes occurrences removed: {total_changes}")
    if bad_record_lines:
        print(f"WARN: {bad_record_lines} non-JSON non-comment lines preserved verbatim")
    return 0


if __name__ == "__main__":
    sys.exit(main())
