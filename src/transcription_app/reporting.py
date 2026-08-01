"""Generate a self-contained HTML evaluation report with SVG charts."""

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Iterable

from .models import ProjectData


def _format_value(value: object) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _bar_chart(items: list[tuple[str, float]], title: str, maximum: float = 1.0) -> str:
    if not items:
        return "<p>No data is available for this chart.</p>"
    width = 760
    left = 180
    chart_width = 520
    row_height = 48
    height = 70 + len(items) * row_height
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)}">']
    parts.append(f'<text x="20" y="30" font-size="20" font-weight="700">{escape(title)}</text>')
    for index, (label, value) in enumerate(items):
        y = 55 + index * row_height
        bar_width = max(0.0, min(chart_width, chart_width * value / maximum if maximum else 0.0))
        parts.append(f'<text x="20" y="{y + 22}" font-size="14">{escape(label)}</text>')
        parts.append(f'<rect x="{left}" y="{y}" width="{chart_width}" height="28" rx="4" fill="#e5e7eb"/>')
        parts.append(f'<rect x="{left}" y="{y}" width="{bar_width:.1f}" height="28" rx="4" fill="#355c7d"/>')
        parts.append(f'<text x="{left + chart_width + 10}" y="{y + 20}" font-size="14">{value:.3f}</text>')
    parts.append("</svg>")
    return "".join(parts)


def export_html_report(project: ProjectData, path: str | Path) -> Path:
    target = Path(path)
    if target.suffix.casefold() != ".html":
        target = target.with_suffix(".html")
    target.parent.mkdir(parents=True, exist_ok=True)
    metrics = [(key, value) for key, value in project.metrics.items() if key != "source_comparison"]
    source_items = [
        (str(item.get("source", "")), float(item.get("wer", 0.0)))
        for item in project.metrics.get("source_comparison", [])
    ]
    model_items = [
        (str(item.get("model", "")), float(item.get("macro_f1", 0.0)))
        for item in project.model_comparison
    ]
    review_count = sum(turn.manual_review for turn in project.turns)
    total_turns = len(project.turns)
    metric_rows = "".join(
        f"<tr><td>{escape(key.replace('_', ' ').title())}</td><td>{escape(_format_value(value))}</td></tr>"
        for key, value in metrics
    )
    html = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Transcription Evaluation Report</title>
<style>
body{{font-family:Segoe UI,Arial,sans-serif;margin:0;background:#f3f4f6;color:#17202a}}main{{max-width:1050px;margin:32px auto;background:white;padding:32px;border-radius:12px;box-shadow:0 4px 20px #0002}}h1,h2{{color:#294c67}}.summary{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:14px}}.card{{background:#eef4f8;padding:18px;border-radius:9px}}.value{{font-size:28px;font-weight:700}}table{{border-collapse:collapse;width:100%;margin:15px 0 30px}}th,td{{padding:10px;border-bottom:1px solid #d9e0e5;text-align:left}}svg{{width:100%;height:auto;border:1px solid #e1e5e8;border-radius:8px;margin:10px 0 28px}}small{{color:#566573}}</style></head>
<body><main>
<h1>Transcription Evaluation Report</h1>
<p><strong>Learner:</strong> {escape(project.metadata.learner_id)} &nbsp; <strong>Session:</strong> {escape(project.metadata.session_number)} &nbsp; <strong>Conversation:</strong> {escape(project.metadata.conversation_type)}</p>
<div class="summary"><div class="card"><div class="value">{total_turns}</div><div>Speech turns</div></div><div class="card"><div class="value">{review_count}</div><div>Manual reviews required</div></div><div class="card"><div class="value">{(review_count/total_turns if total_turns else 0):.1%}</div><div>Review rate</div></div></div>
<h2>Evaluation metrics</h2><table><thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>{metric_rows or '<tr><td colspan="2">Gold Standard data is not available.</td></tr>'}</tbody></table>
<h2>Transcription-source comparison</h2>{_bar_chart(source_items, 'Word Error Rate by Source')}
<h2>Machine-learning comparison</h2>{_bar_chart(model_items, 'Macro F1 by Model')}
<p><small>WER and CER require an aligned Gold Standard transcript. Speaker accuracy is N/A without usable Gold Standard speaker labels. Speech-error preservation is N/A when the Gold Standard contains no detectable hesitation, repetition, self-correction, unclear marker, or Hebrew word. Signal quality fields are calculated for PCM WAV audio; other supported audio formats can still be transcribed and reviewed.</small></p>
</main></body></html>'''
    target.write_text(html, encoding="utf-8")
    return target
