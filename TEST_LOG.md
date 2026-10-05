# Test log

All runs used the local `research_pack/` evidence set, ticker `SRVCABLE`, the
Groq provider, and model `openai/gpt-oss-120b`. No external web search was
used.

## Run 1 — baseline

Design: ask for the brief directly from the evidence pack.

Result: the run completed successfully. It produced a concise brief with
citations, reported the conflicting Q1 revenue figures, and included the
receivables, copper-price, GST, and promoter-pledge risks. It did not include
every weak source in the narrative, which was appropriate.

## Run 2 — evidence-aware

Change from Run 1: instruct the model to rank source reliability, reconcile
conflicts, and exclude irrelevant entities before writing.

Result: the run completed successfully. Compared with baseline, it more
explicitly prioritized the official filing and transcript, surfaced the
₹1,248 crore versus ₹1,428 crore conflict in the snapshot, and retained the
promoter-pledge discrepancy as an open risk.

## Run 3 — validated

Change from Run 2: instruct the model to build a claim-to-source checklist,
require citations for every claim, and explicitly surface conflicts.

Result: after removing citation markup from claim text, the model returned
valid structured data and passed citation-ID validation. The rendered output
used one consistent `[Sx]` citation per citation-array entry, with no
duplicate or mixed-format markers.

## Citation-format fix

The system prompt explicitly says that claim `text` must contain no citation
markup and that source IDs belong only in the `citations` arrays. The renderer
remains the single place that formats inline citations.

## Corrected validation run

The validated variant was rerun after this prompt fix and completed
successfully. Its output confirmed the same clean citation behavior.
