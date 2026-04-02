from __future__ import annotations

from app.schemas import Chunk, TextSegment


def chunk_text(text: str, *, max_chars: int = 1600, overlap: int = 120) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    paragraphs = [part.strip() for part in text.split("\n\n") if part.strip()]
    if not paragraphs:
        paragraphs = [text]

    chunks: list[str] = []
    buffer = ""
    for paragraph in paragraphs:
        candidate = f"{buffer}\n\n{paragraph}".strip() if buffer else paragraph
        if len(candidate) <= max_chars:
            buffer = candidate
            continue
        if buffer:
            chunks.append(buffer)
            tail = buffer[-overlap:].strip()
            buffer = f"{tail}\n\n{paragraph}".strip() if tail else paragraph
        else:
            chunks.append(paragraph[:max_chars])
            buffer = paragraph[max_chars - overlap :]
    if buffer:
        chunks.append(buffer)
    return chunks


def build_chunks(document_id: str, segments: list[TextSegment]) -> list[Chunk]:
    chunks: list[Chunk] = []
    idx = 1
    for segment in segments:
        for piece in chunk_text(segment.text):
            chunks.append(
                Chunk(
                    chunk_id=f"{document_id}:chunk:{idx}",
                    text=piece,
                    source_refs=[{"type": segment.type, "index": segment.index, "label": segment.label}],
                )
            )
            idx += 1
    return chunks
