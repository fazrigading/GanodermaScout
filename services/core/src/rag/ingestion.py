from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .chunker import chunk_text

_SUPPORTED_EXTENSIONS = {".md", ".markdown", ".html", ".htm", ".pdf", ".txt"}
_METADATA_FIELDS = ("title", "author", "year", "source", "license")
_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")


@dataclass(frozen=True)
class DocumentMetadata:
    document_id: str
    title: str | None
    author: str | None
    year: int | None
    source: str | None
    license: str | None
    source_file: str


@dataclass(frozen=True)
class IngestedChunk:
    chunk_id: str
    document_id: str
    content: str
    metadata: dict[str, Any]
    section: str | None
    citation_anchor: str
    source_start: int
    source_end: int
    token_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _year(value: Any) -> int | None:
    match = _YEAR.search(str(value)) if value is not None else None
    return int(match.group()) if match else None


def _markdown_metadata(text: str) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for key, value in re.findall(
        r"(?im)^\s*-\s+\*\*(Title|Author|Year|Source|License):\*\*\s*(.*?)\s*$", text
    ):
        metadata[key.lower()] = value.strip()
    title_heading = re.search(r"(?m)^#\s+([^\n]+)", text)
    if title_heading:
        heading = title_heading.group(1).strip()
        identifier, separator, title = heading.partition(":")
        if separator and re.fullmatch(r"[A-Z0-9][A-Z0-9_-]*", identifier.strip()):
            metadata["document_id"] = identifier.strip()
            metadata.setdefault("title", title.strip())
        else:
            metadata.setdefault("title", heading)
    return metadata


def _html_metadata(soup: Any) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    title = soup.title.get_text(" ", strip=True) if soup.title else None
    heading = soup.find("h1")
    if heading:
        title = heading.get_text(" ", strip=True)
    if title:
        metadata["title"] = title

    field_names = {
        "author": ("author", "dc.creator"),
        "year": ("datepublished", "date", "dc.date"),
        "source": ("publisher", "dc.publisher", "source"),
        "license": ("license", "dc.rights"),
    }
    for field, names in field_names.items():
        for name in names:
            tag = soup.find("meta", attrs={"name": re.compile(f"^{re.escape(name)}$", re.I)})
            if tag is None:
                tag = soup.find("meta", attrs={"property": re.compile(f"^{re.escape(name)}$", re.I)})
            if tag is not None and tag.get("content"):
                metadata[field] = tag["content"].strip()
                break
    return metadata


def _html_to_markdown(soup: Any) -> str:
    lines = []
    for element in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li"]):
        if element.name == "li" and element.find(["p"]):
            continue
        value = element.get_text(" ", strip=True)
        if not value:
            continue
        if element.name.startswith("h"):
            value = f"{'#' * int(element.name[1])} {value}"
        lines.append(value)
    return "\n\n".join(lines)


def _pdf_metadata(reader: Any) -> dict[str, Any]:
    pdf_metadata = reader.metadata or {}
    metadata = {
        "title": pdf_metadata.get("/Title"),
        "author": pdf_metadata.get("/Author"),
        "year": pdf_metadata.get("/CreationDate") or pdf_metadata.get("/PubDate"),
        "source": pdf_metadata.get("/Publisher") or pdf_metadata.get("/Source"),
        "license": pdf_metadata.get("/License") or pdf_metadata.get("/Rights"),
    }
    return {key: value for key, value in metadata.items() if value}


def _read_document(path: Path) -> tuple[str, dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix in {".md", ".markdown", ".txt"}:
        text = path.read_text(encoding="utf-8")
        return text, _markdown_metadata(text) if suffix != ".txt" else {}
    if suffix in {".html", ".htm"}:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")
        return _html_to_markdown(soup), _html_metadata(soup)
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(path)
        text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
        return text, _pdf_metadata(reader)
    raise ValueError(f"Unsupported document type: {suffix}")


def _load_manifest(corpus_dir: Path) -> dict[str, dict[str, Any]]:
    manifest_path = corpus_dir / "manifest.json"
    if not manifest_path.is_file():
        return {}
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = data.get("documents", [])
    if not isinstance(entries, list):
        raise ValueError("Corpus manifest 'documents' must be a list")
    manifest = {}
    for entry in entries:
        if not isinstance(entry, dict) or not entry.get("file"):
            continue
        manifest[entry["file"]] = entry
    return manifest


def _document_metadata(
    path: Path, corpus_dir: Path, extracted: dict[str, Any], manifest: dict[str, Any]
) -> DocumentMetadata:
    values = {**extracted, **manifest}
    document_id = str(values.get("document_id") or values.get("doc_id") or path.stem)
    year = _year(values.get("year"))
    fields = {field: values.get(field) for field in _METADATA_FIELDS}
    return DocumentMetadata(
        document_id=document_id,
        title=str(fields["title"]).strip() if fields["title"] else None,
        author=str(fields["author"]).strip() if fields["author"] else None,
        year=year,
        source=str(fields["source"]).strip() if fields["source"] else None,
        license=str(fields["license"]).strip() if fields["license"] else None,
        source_file=path.relative_to(corpus_dir).as_posix(),
    )


def _metadata_dict(metadata: DocumentMetadata) -> dict[str, Any]:
    return {
        "document_id": metadata.document_id,
        "title": metadata.title,
        "author": metadata.author,
        "year": metadata.year,
        "source": metadata.source,
        "license": metadata.license,
        "source_file": metadata.source_file,
    }


def ingest_corpus(
    corpus_dir: str | Path,
    *,
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[IngestedChunk]:
    root = Path(corpus_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Corpus directory does not exist: {root}")
    manifest = _load_manifest(root)
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in _SUPPORTED_EXTENSIONS
    )
    chunks = []
    seen_document_ids: set[str] = set()
    for path in files:
        relative_path = path.relative_to(root).as_posix()
        entry = manifest.get(relative_path, manifest.get(path.name, {}))
        text, extracted = _read_document(path)
        metadata = _document_metadata(path, root, extracted, entry)
        if metadata.document_id in seen_document_ids:
            raise ValueError(f"Duplicate document ID in corpus: {metadata.document_id}")
        seen_document_ids.add(metadata.document_id)
        chunks.extend(_build_chunks(text, metadata, chunk_size=chunk_size, overlap=overlap))
    return chunks


def _build_chunks(
    text: str,
    metadata: DocumentMetadata,
    *,
    chunk_size: int,
    overlap: int,
) -> list[IngestedChunk]:
    base_metadata = _metadata_dict(metadata)
    chunks = []
    for index, chunk in enumerate(chunk_text(text, chunk_size=chunk_size, overlap=overlap), start=1):
        chunk_id = f"{metadata.document_id}#p{index}"
        chunks.append(
            IngestedChunk(
                chunk_id=chunk_id,
                document_id=metadata.document_id,
                content=chunk.content,
                metadata={**base_metadata, "section": chunk.section},
                section=chunk.section,
                citation_anchor=chunk_id,
                source_start=chunk.source_start,
                source_end=chunk.source_end,
                token_count=chunk.token_count,
            )
        )
    return chunks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest agronomy documents into JSON passage chunks")
    parser.add_argument("--corpus", default="data/sample_dataset/corpus")
    parser.add_argument("--output", default="-", help="JSON output path, or '-' for stdout")
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--overlap", type=int, default=50)
    args = parser.parse_args(argv)

    chunks = ingest_corpus(args.corpus, chunk_size=args.chunk_size, overlap=args.overlap)
    rendered = json.dumps([chunk.to_dict() for chunk in chunks], ensure_ascii=False, indent=2)
    if args.output == "-":
        print(rendered)
    else:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
