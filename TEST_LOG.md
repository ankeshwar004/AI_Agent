# SRVCABLE agent test log

The Gemini key was intentionally not committed. The three design trials below
were run in offline mode, which exercises document loading and prompt
construction without making an API request. With `GEMINI_API_KEY` set, the
same request enters the LangGraph workflow and validates the returned brief.

## Run 1: baseline

Command:

```bash
python research_agent.py --ticker SRVCABLE --input-dir research_pack \
  --variant baseline --dry-run
```

Result: the eight documents loaded successfully, but the baseline design only
asked for direct summarization. It did not make source ranking and conflict
reconciliation explicit enough.

Change: add the `evidence-aware` variant, which tells the model to rank source
reliability and reject irrelevant entities.

## Run 2: evidence-aware

Command:

```bash
python research_agent.py --ticker SRVCABLE --input-dir research_pack \
  --variant evidence-aware --dry-run
```

Result: the prompt included all eight source IDs and the untrusted-document
boundary. This specifically covers the promotional blog's embedded
instruction, the conflicting revenue figures, and the local cable-TV article.

Change: add the `validated` variant while keeping the same structured output
schema.

## Run 3: validated final design

Command:

```bash
python research_agent.py --ticker SRVCABLE --input-dir research_pack \
  --variant validated --dry-run
```

Result: the final request prepared all eight documents and enabled the claim-
to-source checklist instruction. Unit tests confirmed that a grounded brief is
accepted and unknown citations are rejected.

Limitation: these offline runs cannot assess Gemini's prose quality. After
adding the key, the final command should be run with `--output
srvcable_brief.md`.

## Framework migration

The final runner uses a simple LangGraph function to call LangChain's
`ChatGoogleGenerativeAI`. The graph is intentionally small:

```text
START -> generate_brief -> END
```

The model response is now requested with LangChain's
`with_structured_output(ResearchBrief)`. Pydantic validates the response shape,
and the application renders it to Markdown. Runtime error handling is
intentionally omitted for this prototype. The three variants remain available
as process-prompt experiments around the same structured schema.

## Final validation status

The local final validation passed:

- All eight source files load and receive stable source IDs.
- The validated prompt includes every source in an untrusted-content boundary.
- The LangGraph compiles with its `generate_brief` node.
- The Pydantic structured output renders all required Markdown sections.

The actual Gemini-generated SRVCABLE brief remains key-dependent. No brief is
claimed as generated until the command is run with `GEMINI_API_KEY`:

```bash
python research_agent.py --ticker SRVCABLE --input-dir research_pack \
  --variant validated --output srvcable_brief.md
```
