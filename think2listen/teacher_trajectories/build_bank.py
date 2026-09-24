#!/usr/bin/env python3
"""Collect verified rollout records into the public teacher-bank schema."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _record(row: dict[str, Any], *, supervise_reasoning: bool) -> dict[str, Any] | None:
    metadata = dict(row.get("metadata") or {})
    merged = {**metadata, **row}
    correct = bool(row.get("correct")) or (
        str(row.get("predicted_letter") or "") == str(merged.get("correct_letter") or "")
    )
    if row.get("status") not in {"scored", "success"} or not correct:
        return None

    correct_letter = str(merged["correct_letter"])
    actions = []
    queries = []
    for action in row.get("actions") or []:
        kind = action.get("kind")
        if kind == "ask":
            query = str(action.get("query") or "").strip()
            observation = str(action.get("observation") or action.get("actor_observation") or "").strip()
            if not query or not observation:
                raise ValueError(f"Incomplete ASK in sample {merged.get('sample_id')}")
            queries.append(query)
            actions.append({
                "kind": "ask",
                "canonical_action": f"<ask>{query}</ask>",
                "query": query,
                "observation": observation,
                "raw_teacher_text": str(action.get("text") or action.get("actor_text") or ""),
            })
        elif kind == "answer":
            answer = str(action.get("answer") or "")
            if answer != correct_letter:
                raise ValueError(f"Wrong terminal answer in sample {merged.get('sample_id')}")
            actions.append({
                "kind": "answer",
                "canonical_action": f"<answer>{answer}</answer>",
                "answer": answer,
                "raw_teacher_text": str(action.get("text") or action.get("actor_text") or ""),
            })
    if not actions or actions[-1]["kind"] != "answer":
        raise ValueError(f"Missing terminal answer in sample {merged.get('sample_id')}")

    sample_id = str(merged["sample_id"])
    key = json.dumps([sample_id, queries, correct_letter], ensure_ascii=False)
    ask_count = len(queries)
    return {
        "trajectory_id": hashlib.sha256(key.encode()).hexdigest()[:24],
        "sample_id": sample_id,
        "data_split": "train",
        "teacher_source": str(row.get("teacher_source") or "verified_rollout"),
        "question": merged["question"],
        "choices": merged["choices"],
        "initial_description": merged["initial_description"],
        "correct_letter": correct_letter,
        "ask_count": ask_count,
        "ask_bucket": "direct" if ask_count == 0 else "one_ask" if ask_count == 1 else "multi_ask",
        "actions": actions,
        "audio_prompt_version": merged.get("audio_protocol_version", "targeted-audio-v1"),
        "prompt_alignment": "public_v1_exact",
        "same_actor_prompt_as_rl": True,
        "training_target_policy": "raw_teacher_actions" if supervise_reasoning else "canonical_actions_only",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--supervise-reasoning", action="store_true")
    args = parser.parse_args()

    by_id: dict[str, dict[str, Any]] = {}
    for path in args.input:
        for row in _read_jsonl(path):
            record = _record(row, supervise_reasoning=args.supervise_reasoning)
            if record is not None:
                by_id.setdefault(record["trajectory_id"], record)
    rows = sorted(by_id.values(), key=lambda item: (item["sample_id"], item["trajectory_id"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"trajectories": len(rows), "output": str(args.output)}))


if __name__ == "__main__":
    main()
