from __future__ import annotations

import re
from dataclasses import dataclass

import tiktoken

_TOKENIZER = tiktoken.get_encoding("cl100k_base")
_METADATA_LINE = re.compile(
    r"(?m)^\s*-\s+\*\*(?:title|author|year|source|license):\*\*[^\n]*(?:\n|$)",
    re.IGNORECASE,
)
_HEADING = re.compile(r"(?m)^(#{1,6})[ \t]+([^\n]+?)\s*$")


@dataclass(frozen=True)
class TextChunk:
    content: str
    section: str | None
    source_start: int
    source_end: int
    token_count: int


@dataclass(frozen=True)
class _Section:
    prefix: str
    section: str | None
    body: str
    body_start: int


def _mask_metadata(text: str) -> str:
    return _METADATA_LINE.sub(lambda match: "".join("\n" if char == "\n" else " " for char in match.group()), text)


def _trimmed_span(text: str, start: int) -> tuple[str, int]:
    leading = len(text) - len(text.lstrip())
    content = text.strip()
    return content, start + leading


def _sections(text: str) -> list[_Section]:
    masked = _mask_metadata(text)
    headings = list(_HEADING.finditer(masked))
    sections: list[_Section] = []
    document_heading: str | None = next(
        (match.group(0).strip() for match in headings if len(match.group(1)) == 1),
        None,
    )
    heading_stack: dict[int, str] = {}
    first_body_end = headings[0].start() if headings else len(masked)
    preamble, preamble_start = _trimmed_span(masked[:first_body_end], 0)
    if preamble:
        sections.append(_Section(prefix="", section=None, body=preamble, body_start=preamble_start))

    for index, heading in enumerate(headings):
        level = len(heading.group(1))
        heading_end = headings[index + 1].start() if index + 1 < len(headings) else len(masked)
        body, body_start = _trimmed_span(masked[heading.end() : heading_end], heading.end())
        heading_text = heading.group(0).strip()
        if level == 1:
            if body:
                sections.append(
                    _Section(prefix=heading_text, section=None, body=body, body_start=body_start)
                )
            heading_stack = {1: heading_text}
            continue

        for ancestor_level in tuple(heading_stack):
            if ancestor_level >= level:
                del heading_stack[ancestor_level]
        heading_stack[level] = heading_text
        if not body:
            continue
        prefix_parts = []
        if document_heading:
            prefix_parts.append(document_heading)
        prefix_parts.extend(
            heading_stack[ancestor_level]
            for ancestor_level in sorted(heading_stack)
            if ancestor_level > 1
        )
        section_heading = next(
            (
                re.sub(r"^#{1,6}\s+", "", heading_stack[ancestor_level]).strip()
                for ancestor_level in sorted(heading_stack, reverse=True)
                if ancestor_level > 1
            ),
            None,
        )
        sections.append(
            _Section(
                prefix="\n\n".join(prefix_parts),
                section=section_heading,
                body=body,
                body_start=body_start,
            )
        )
    if not sections and masked.strip():
        body, body_start = _trimmed_span(masked, 0)
        sections.append(_Section(prefix="", section=None, body=body, body_start=body_start))
    return sections


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[TextChunk]:
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("chunk_size must be positive and overlap must be within the chunk size")

    chunks: list[TextChunk] = []
    for section in _sections(text):
        prefix = f"{section.prefix}\n\n" if section.prefix else ""
        prefix_tokens = len(_TOKENIZER.encode(prefix))
        body_tokens = _TOKENIZER.encode(section.body)
        if prefix_tokens >= chunk_size:
            raise ValueError("Section headings exceed the configured chunk size")
        window_size = chunk_size - prefix_tokens
        if len(body_tokens) <= window_size:
            content = prefix + _TOKENIZER.decode(body_tokens)
            chunks.append(
                TextChunk(
                    content=content,
                    section=section.section,
                    source_start=section.body_start,
                    source_end=section.body_start + len(section.body),
                    token_count=len(_TOKENIZER.encode(content)),
                )
            )
            continue

        start = 0
        while start < len(body_tokens):
            end = min(start + window_size, len(body_tokens))
            content = prefix + _TOKENIZER.decode(body_tokens[start:end])
            while len(_TOKENIZER.encode(content)) > chunk_size and end > start:
                end -= 1
                content = prefix + _TOKENIZER.decode(body_tokens[start:end])
            if end == start:
                raise ValueError("Section header leaves no room for content")
            chunks.append(
                TextChunk(
                    content=content,
                    section=section.section,
                    source_start=section.body_start + len(_TOKENIZER.decode(body_tokens[:start])),
                    source_end=section.body_start + len(_TOKENIZER.decode(body_tokens[:end])),
                    token_count=len(_TOKENIZER.encode(content)),
                )
            )
            if end == len(body_tokens):
                break
            start = end - overlap
    return chunks
