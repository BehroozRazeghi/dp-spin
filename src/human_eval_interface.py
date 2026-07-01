"""Render a static page for manual plan and summary evaluation.

For a private DP-SPIN run, the page should contain only the released semantic plan, its generated summary, and public metadata. 
Including raw protected records or non-private reference artifacts changes the evaluation boundary and must be labeled accordingly.
"""
from __future__ import annotations

import argparse
import html
from pathlib import Path
from .utils import read_json

TEMPLATE = """<!doctype html><html><head><meta charset='utf-8'><title>DP-SPIN Human Evaluation</title>
<style>body{font-family:Arial,sans-serif;max-width:900px;margin:30px auto;line-height:1.45} .card{border:1px solid #ccc;border-radius:8px;padding:16px;margin:20px 0}.plan{background:#f7f7ff}.summary{background:#f8fff8} textarea{width:100%;height:90px} label{display:block;margin-top:8px}</style></head><body>
<h1>DP-SPIN Human Evaluation</h1>
<p>Rate the summary with respect to the private plan. Do not inspect raw records.</p>
<div class='card plan'><h2>Private plan</h2><pre>{plan}</pre></div>
<div class='card summary'><h2>Summary</h2><p>{summary}</p></div>
<form>
<label>Coverage (1-5): <input type='number' min='1' max='5'></label>
<label>Faithfulness to plan (1-5): <input type='number' min='1' max='5'></label>
<label>Usefulness (1-5): <input type='number' min='1' max='5'></label>
<label>Clarity (1-5): <input type='number' min='1' max='5'></label>
<label>Notes:<textarea></textarea></label>
</form>
</body></html>"""


def make_page(plan_path: str, summary_path: str, output: str) -> None:
    """Render a standalone HTML page for manual summary evaluation.
    """
    plan = html.escape(__import__('json').dumps(read_json(plan_path), indent=2, ensure_ascii=False))
    summary = html.escape(Path(summary_path).read_text(encoding='utf-8')).replace('\n', '<br>')
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(TEMPLATE.format(plan=plan, summary=summary), encoding='utf-8')
    print(f"Wrote human evaluation page to {output}")


def main() -> None:
    """Command-line entry point for human-evaluation page generation.
    """
    p = argparse.ArgumentParser(description="Create a static HTML human-evaluation form for one run.")
    p.add_argument('--plan', required=True)
    p.add_argument('--summary', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    make_page(args.plan, args.summary, args.output)

if __name__ == '__main__':
    main()
