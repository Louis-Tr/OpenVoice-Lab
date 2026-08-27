from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from .config import PipelineConfig
from .io_utils import (
    atomic_write_json,
    atomic_write_jsonl,
    atomic_write_text,
    read_jsonl,
)
from .text import audit_model_text


VALID_DECISIONS = {"accept", "reject", "edit"}


def load_decisions(path: Path) -> tuple[dict[str, dict[str, Any]], int]:
    latest: dict[str, dict[str, Any]] = {}
    action_count = 0
    if not path.exists():
        return latest, action_count
    for action in read_jsonl(path):
        sample_id = action.get("sample_id")
        decision = action.get("decision")
        if not isinstance(sample_id, str) or decision not in VALID_DECISIONS:
            raise ValueError(f"Invalid preserved review action: {action}")
        if decision == "edit":
            edited = action.get("edited_model_input_text", "")
            if (
                not isinstance(edited, str)
                or not edited.strip()
                or not audit_model_text(edited)["safe"]
            ):
                raise ValueError(
                    f"Unsafe or empty edited_model_input_text for {sample_id}"
                )
        latest[sample_id] = action
        action_count += 1
    return latest, action_count


def apply_review(
    records: list[dict[str, Any]],
    config: PipelineConfig,
    *,
    accept_unreviewed_for_smoke: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    review_dir = config.path("review")
    review_dir.mkdir(parents=True, exist_ok=True)
    decisions_path = review_dir / str(config.section("review")["decisions_file"])
    latest, action_count = load_decisions(decisions_path)
    queue: list[dict[str, Any]] = []
    decision_counts = {"accept": 0, "reject": 0, "edit": 0}
    for record in records:
        required = bool(record.get("review_flags"))
        record["review_required"] = required
        if not required:
            record["review_status"] = "not_required"
            continue
        decision = latest.get(record["sample_id"])
        if decision is None:
            if accept_unreviewed_for_smoke:
                record["review_status"] = "accepted_unreviewed_for_smoke"
                record["review_policy_override"] = "accept_unreviewed_for_smoke"
            else:
                record["review_status"] = "pending"
                if "MANUAL_REVIEW_PENDING" not in record["exclusion_reasons"]:
                    record["exclusion_reasons"].append("MANUAL_REVIEW_PENDING")
        else:
            record["review_status"] = decision["decision"]
            record["review_latest_action"] = decision
            decision_counts[decision["decision"]] += 1
            if decision["decision"] == "reject":
                if "MANUAL_REJECTED" not in record["exclusion_reasons"]:
                    record["exclusion_reasons"].append("MANUAL_REJECTED")
            elif decision["decision"] == "edit":
                record["review_edited_model_input_text"] = decision[
                    "edited_model_input_text"
                ].strip()
                record["model_input_text"] = record["review_edited_model_input_text"]
        record["exclusion_reasons"].sort()
        queue.append(_queue_record(record))

    atomic_write_jsonl(review_dir / "review_queue.jsonl", queue)
    atomic_write_text(review_dir / "review_queue.html", _review_html(queue))
    summary = {
        "queue_size": len(queue),
        "preserved_action_count": action_count,
        "latest_decision_counts": decision_counts,
        "pending_count": sum(
            record.get("review_status") == "pending" for record in records
        ),
        "accepted_unreviewed_for_smoke_count": sum(
            record.get("review_status") == "accepted_unreviewed_for_smoke"
            for record in records
        ),
        "accept_unreviewed_for_smoke": accept_unreviewed_for_smoke,
        "decisions_file": str(decisions_path),
        "actions_fabricated": False,
    }
    atomic_write_json(review_dir / "review_summary.json", summary)
    return records, summary


def _queue_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": record["sample_id"],
        "audio": record.get("clean_audio"),
        "original_transcript": record.get("original_transcript"),
        "normalized_transcript": record.get("normalized_transcript"),
        "model_input_text": record.get("model_input_text"),
        "asr_transcript": record.get("asr_transcript"),
        "asr_wer": record.get("asr_wer"),
        "asr_status": record.get("asr_status"),
        "quality_metrics": record.get("quality_metrics"),
        "source_quality_signals": record.get("source_quality_signals"),
        "review_flags": record.get("review_flags", []),
        "review_status": record.get("review_status"),
        "latest_action": record.get("review_latest_action"),
    }


def _review_html(queue: list[dict[str, Any]]) -> str:
    rows = []
    for item in queue:
        sample_id = html.escape(item["sample_id"])
        rows.append(
            f"<article data-id='{sample_id}'><h2>{sample_id}</h2>"
            f"<audio controls preload='none' src='/audio/{sample_id}'></audio>"
            f"<p><b>Original:</b> {html.escape(item.get('original_transcript') or '')}</p>"
            f"<p><b>Normalized:</b> {html.escape(item.get('normalized_transcript') or '')}</p>"
            f"<p><b>Model input:</b> {html.escape(item.get('model_input_text') or '')}</p>"
            f"<p><b>ASR:</b> {html.escape(item.get('asr_transcript') or 'not run')} "
            f"(WER: {html.escape(str(item.get('asr_wer')))})</p>"
            f"<pre>{html.escape(json.dumps({'quality': item.get('quality_metrics'), 'flags': item.get('review_flags')}, indent=2))}</pre>"
            "<label>Edited model text <input class='edit' size='80'></label> "
            "<button data-decision='accept'>Accept</button> "
            "<button data-decision='reject'>Reject</button> "
            "<button data-decision='edit'>Accept edit</button><output></output></article>"
        )
    body = "\n".join(rows) or "<p>No records currently require manual review.</p>"
    return f"""<!doctype html><meta charset='utf-8'><title>Medical TTS review queue</title>
<style>body{{font:15px system-ui;max-width:1100px;margin:2rem auto}}article{{border:1px solid #bbb;padding:1rem;margin:1rem 0}}pre{{white-space:pre-wrap;background:#f5f5f5;padding:.5rem}}button{{margin:.3rem}}</style>
<h1>Medical TTS manual review queue</h1>
<p>Serve with <code>python -m training.data_pipeline.review_server --config training/config/dataset.yaml</code>. Every submitted action is appended; existing history is never overwritten.</p>
{body}
<script>document.addEventListener('click',async e=>{{if(!e.target.dataset.decision)return;let a=e.target.closest('article'),d=e.target.dataset.decision,p={{sample_id:a.dataset.id,decision:d}};if(d==='edit')p.edited_model_input_text=a.querySelector('.edit').value;let r=await fetch('/decision',{{method:'POST',headers:{{'content-type':'application/json'}},body:JSON.stringify(p)}});a.querySelector('output').textContent=r.ok?' saved':' '+await r.text();}})</script>"""
