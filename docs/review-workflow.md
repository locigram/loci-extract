# Review workflow

## Development plan

1. Build extraction core using local parser/OCR libraries.
2. Use local OpenAI-compatible models for optional enrichment and design/code review loops.
3. Run a final external audit with ChatGPT on:
   - API design
   - extraction schema
   - code quality
   - deployment model
   - security/privacy considerations

## Notes

- Core extraction should remain deterministic and parser-first.
- Local models should help with reasoning, enrichment, and code-review assistance.
- Final ChatGPT audit should be treated as an independent review pass, not the primary implementation engine.
