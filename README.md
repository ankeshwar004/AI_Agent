# Super Investing AI research agent

This repository contains a small, evidence-grounded research agent for Indian
stocks. It accepts an NSE ticker and a local folder of Markdown documents, then
asks a Groq-hosted model to produce a one-page Markdown research brief.

## Setup

Python 3.10 or newer is required. Install the LangGraph and LangChain Groq
integration:

```bash
python -m pip install -r requirements.txt
```

Add a Groq API key when you are ready:

```bash
export GROQ_API_KEY="your-key"
```

The key is read only from the environment and is not stored in the repository.

## Run on the supplied test pack

```bash
python research_agent.py \
  --ticker SRVCABLE \
  --input-dir research_pack \
  --variant validated \
  --output srvcable_brief.md
```

The default model is `openai/gpt-oss-120b`, selected for structured-output
support and strong instruction following on Groq. Override it with `--model`
if needed.

A real final brief requires `GROQ_API_KEY`.

## Design

- [document_loader.py](src/document_loader.py) reads the Markdown files and
  gives each document a stable ID such as `[S1]`.
- [research_brief_system.txt](prompts/research_brief_system.txt) is the
  standalone system prompt.
- [research_agent.py](research_agent.py) builds a LangGraph workflow with a
  `generate_brief` function that uses LangChain's
  `ChatGroq.with_structured_output(ResearchBrief)`.
- The `baseline`, `evidence-aware`, and `validated` variants remain available
  for the three prompt-design experiments. They change process guidance, not
  the output schema.

The prompt deliberately handles conflicting figures, irrelevant entities, weak
promotional sources, and prompt injection inside a source document. The
prototype reports API and model errors directly rather than hiding them.

## Test log

See [TEST_LOG.md](TEST_LOG.md) for the three design trials and the changes made
between them.
