"""``loci-extract`` CLI — argparse entry point.

Calls into ``loci_extract.core`` for the real work. Verbose output goes
to stderr so stdout stays clean for piping JSON/CSV/Lacerte/TXF downstream.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from loci_extract import __version__
from loci_extract.core import ExtractionOptions, extract_batch, extract_document
from loci_extract.formatters import format_extraction
from loci_extract.schema import Document, Extraction

# Defaults used when env vars are unset. These point at the in-house
# surugpu llama-server that typically hosts Qwen3-VL 32B for the team.
_DEFAULT_MODEL_URL = os.getenv("LOCI_EXTRACT_MODEL_URL", "http://10.10.100.20:9020/v1")
_DEFAULT_MODEL_NAME = os.getenv("LOCI_EXTRACT_MODEL_NAME", "qwen3-vl-32b")
_DEFAULT_VISION_MODEL = os.getenv("LOCI_EXTRACT_VISION_MODEL", _DEFAULT_MODEL_NAME)


def _eprint(*args, **kwargs) -> None:
    print(*args, file=sys.stderr, **kwargs)


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="loci-extract",
        description=(
            "Local-first tax document extractor. Reads W-2s, 1099s, 1098s, SSA-1099, "
            "RRB-1099, and K-1s from PDF and emits JSON/CSV/Lacerte/TXF."
        ),
    )
    p.add_argument("input", help="PDF file, or directory (with --batch)")
    p.add_argument("-o", "--output", help="Output file (default: stdout)")
    p.add_argument(
        "--model",
        default=_DEFAULT_MODEL_URL,
        help=f"OpenAI-compatible LLM base URL (default: ${{LOCI_EXTRACT_MODEL_URL}} or {_DEFAULT_MODEL_URL})",
    )
    p.add_argument(
        "--model-name",
        default=_DEFAULT_MODEL_NAME,
        help=f"Model name passed in API call (default: ${{LOCI_EXTRACT_MODEL_NAME}} or {_DEFAULT_MODEL_NAME})",
    )
    p.add_argument(
        "--vision",
        action="store_true",
        help="Route image pages through the VLM instead of OCR",
    )
    p.add_argument(
        "--vision-model",
        default=_DEFAULT_VISION_MODEL,
        help=f"VLM model name for image pages (default: {_DEFAULT_VISION_MODEL})",
    )
    p.add_argument(
        "--ocr-engine",
        choices=["auto", "tesseract", "easyocr", "paddleocr"],
        default="auto",
        help="OCR engine for image pages (default: auto)",
    )
    p.add_argument(
        "--gpu",
        choices=["auto", "true", "false"],
        default="auto",
        help="Use GPU for OCR when applicable (default: auto)",
    )
    p.add_argument("--dpi", type=int, default=300, help="OCR/vision render DPI (default: 300)")
    p.add_argument("--batch", action="store_true", help="Treat input as a directory of PDFs")
    p.add_argument(
        "--format",
        choices=["json", "csv", "lacerte", "txf"],
        default="json",
        help="Output format (default: json)",
    )
    p.add_argument(
        "--no-redact",
        dest="redact",
        action="store_false",
        help="Disable SSN/TIN redaction (default: redact to last 4)",
    )
    p.set_defaults(redact=True)
    p.add_argument("--temperature", type=float, default=0.0, help="LLM temperature (default: 0.0)")
    p.add_argument("--max-tokens", type=int, default=4096, help="LLM max tokens (default: 4096)")
    p.add_argument("--retry", type=int, default=2, help="Retry count on invalid JSON (default: 2)")
    p.add_argument(
        "--no-fix-orientation",
        dest="fix_orientation",
        action="store_false",
        help="Disable auto-rotation of scanned-sideways pages (default: enabled)",
    )
    p.set_defaults(fix_orientation=True)
    p.add_argument(
        "--chunk-size",
        type=int,
        default=6000,
        dest="chunk_size_tokens",
        help="Max input tokens per LLM chunk for long documents (default: 6000)",
    )
    p.add_argument(
        "--no-verify-totals",
        dest="verify_totals",
        action="store_false",
        help="Skip Python-side totals verification on financial documents",
    )
    p.set_defaults(verify_totals=True)
    p.add_argument(
        "--family",
        choices=["tax", "financial_simple", "financial_multi", "financial_txn", "financial_reserve"],
        default=None,
        help="Force document family (default: auto-detect)",
    )
    p.add_argument(
        "--parallel-chunks",
        type=int,
        default=4,
        dest="max_parallel",
        help="Concurrent LLM calls for chunked financial docs (default: 4; 1 = sequential)",
    )
    p.add_argument("--verbose", action="store_true", help="Pipeline steps to stderr")
    p.add_argument("--version", action="version", version=f"loci-extract {__version__}")
    return p


def _options_from_args(args: argparse.Namespace) -> ExtractionOptions:
    return ExtractionOptions(
        model_url=args.model,
        model_name=args.model_name,
        ocr_engine=args.ocr_engine,
        gpu=args.gpu,
        dpi=args.dpi,
        vision=args.vision,
        vision_model=args.vision_model,
        redact=args.redact,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        retry=args.retry,
        chunk_size_tokens=args.chunk_size_tokens,
        verify_totals=args.verify_totals,
        fix_orientation=args.fix_orientation,
        force_family=args.family,
        max_parallel=args.max_parallel,
    )


def _collect_batch_pdfs(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.pdf") if p.is_file())


def _merge(batch_results) -> Extraction:
    merged_docs: list[Document] = []
    for _path, extraction in batch_results:
        merged_docs.extend(extraction.documents)
    return Extraction(documents=merged_docs)


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    opts = _options_from_args(args)

    progress = _eprint if args.verbose else None
    input_path = Path(args.input)

    try:
        if args.batch:
            if not input_path.is_dir():
                _eprint(f"--batch requires a directory, got {input_path}")
                return 1
            pdfs = _collect_batch_pdfs(input_path)
            if not pdfs:
                _eprint(f"No PDFs found under {input_path}")
                return 1
            batch_results = extract_batch(pdfs, opts, progress_callback=progress)
            extraction = _merge(batch_results)
        else:
            if not input_path.is_file():
                _eprint(f"Input file not found: {input_path}")
                return 1
            extraction = extract_document(input_path, opts, progress_callback=progress)
    except Exception as exc:
        _eprint(f"ERROR: {type(exc).__name__}: {exc}")
        return 1

    try:
        output = format_extraction(extraction, args.format)
    except NotImplementedError as exc:
        _eprint(f"ERROR: {exc}")
        return 1

    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
        if args.verbose:
            _eprint(f"wrote {args.output}")
    else:
        sys.stdout.write(output)
        if not output.endswith("\n"):
            sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
