"""Validate bundled sample dataset: images, labels, corpus, golden pairs."""

import json
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
IMG_DIR = ROOT / "data" / "sample_dataset" / "images"
CORPUS_DIR = ROOT / "data" / "sample_dataset" / "corpus"
GOLDEN = ROOT / "data" / "sample_dataset" / "eval_golden.json"

errors: list[str] = []


def check_images() -> None:
    for name in ("primordium_001.png", "mature_001.png"):
        path = IMG_DIR / name
        if not path.exists():
            errors.append(f"missing image {name}")
            continue
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            w, h = im.size
        label = IMG_DIR / (Path(name).stem + ".txt")
        if not label.exists():
            errors.append(f"missing label for {name}")
            continue
        for line in label.read_text().splitlines():
            parts = line.split()
            if len(parts) != 5:
                errors.append(f"bad yolo line in {label.name}: {line}")
                continue
            _, cx, cy, bw, bh = (float(v) for v in parts)
            for v in (cx, cy, bw, bh):
                if not 0.0 <= v <= 1.0:
                    errors.append(f"yolo value out of range in {label.name}: {line}")
            x_min = (cx - bw / 2) * w
            y_min = (cy - bh / 2) * h
            x_max = (cx + bw / 2) * w
            y_max = (cy + bh / 2) * h
            if not (0 <= x_min < x_max <= w and 0 <= y_min < y_max <= h):
                errors.append(f"bbox outside dims in {label.name}: {line}")


def check_corpus() -> None:
    manifest_path = CORPUS_DIR / "manifest.json"
    if not manifest_path.exists():
        errors.append("missing corpus manifest.json")
        return
    manifest = json.loads(manifest_path.read_text())
    for doc in manifest.get("documents", []):
        for field in ("doc_id", "file", "title", "author", "year", "license"):
            if field not in doc:
                errors.append(f"manifest doc missing {field}: {doc}")
        file_path = CORPUS_DIR / doc.get("file", "")
        if not file_path.exists():
            errors.append(f"missing corpus file {doc.get('file')}")


def check_golden() -> None:
    if not GOLDEN.exists():
        errors.append("missing eval_golden.json")
        return
    pairs = json.loads(GOLDEN.read_text()).get("pairs", [])
    if len(pairs) != 10:
        errors.append(f"expected 10 golden pairs, found {len(pairs)}")
    manifest = json.loads((CORPUS_DIR / "manifest.json").read_text())
    valid_spans = {c["chunk_id"] for d in manifest.get("documents", []) for c in d.get("chunks", [])}
    for pair in pairs:
        if not pair.get("question") or not pair.get("answer"):
            errors.append(f"golden pair missing qa text: {pair.get('pair_id')}")
        for span in pair.get("citation_spans", []):
            if span not in valid_spans:
                errors.append(f"unknown citation span {span} in {pair.get('pair_id')}")


def main() -> int:
    check_images()
    check_corpus()
    check_golden()
    if errors:
        print("sample data validation failed:")
        for err in errors:
            print(f" - {err}")
        return 1
    print("sample data validation passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
