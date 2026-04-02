# Local model workflow

## Goal

Use local OpenAI-compatible model endpoints during development for any higher-level post-processing tasks such as:

- section labeling
- table summarization
- metadata extraction
- chunk enrichment
- document classification

## Available local endpoints

- coding: `http://10.10.100.80:30892`
- inference: `http://10.10.100.80:30889`
- midrange: `http://10.10.100.80:30891`
- vlm: `http://10.10.100.80:30893`

## Proposed usage

- deterministic extraction stays parser-first
- local LLMs are used only for optional enrichment
- OCR/text extraction should never depend entirely on an LLM
- ChatGPT audit can be a separate review pass over code, schema, and API design

## Suggested phases

1. Build parser/router/extraction core
2. Add PDF OCR fallback and image cleanup
3. Add optional local-LLM enrichers
4. Run external audit/review against the codebase and design
