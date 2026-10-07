from __future__ import annotations

import re
from typing import Any

from .state import AgentState


NO_SOURCE_ADVISORY = (
    "No source-grounded recommendation is available; consult a plantation agronomist."
)
SAFE_REWRITE = (
    "Unverified advice or chemical dosage was removed; consult a plantation "
    "agronomist or laboratory before acting."
)
SAFE_REFUSAL = (
    "I can't comply with instructions to bypass safety controls. I can help with "
    "source-grounded Ganoderma guidance or inspection history."
)

_INPUT_PATTERNS = (
    (
        "instruction_override",
        re.compile(
            r"\b(?:ignore|disregard|forget|override)\b.{0,60}"
            r"\b(?:system|developer|previous|prior|above|earlier|safety|guardrails?|all)\b"
            r".{0,30}\b(?:instructions?|rules?)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "reveal_hidden_prompt",
        re.compile(
            r"\b(?:reveal|show|print|repeat|display|dump|expose)\b.{0,60}"
            r"\b(?:system|developer|hidden|internal)\s+(?:prompt|instructions?|message)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "safety_bypass",
        re.compile(
            r"\b(?:bypass|disable|ignore|override|remove)\b.{0,50}"
            r"\b(?:safety|guardrails?|policies|policy)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "jailbreak",
        re.compile(
            r"\b(?:jailbreak|DAN(?:\s+mode)?|developer\s+mode|do\s+anything\s+now)\b",
            re.IGNORECASE,
        ),
    ),
)
_CITATION_TAG = re.compile(r"\[Source:\s*([^\]]+)\]")
_DOSE_AMOUNT = re.compile(
    r"(?<!\w)\d+(?:[.,]\d+)?\s*(?:"
    r"micrograms?|mcg|[µμ]g|milligrams?|mg|kilograms?|kg|grams?|g|"
    r"millilit(?:er|re)s?|ml|lit(?:er|re)s?|l|ppm|ppb|%"
    r")(?:\s*(?:/|per)\s*(?:\d+(?:[.,]\d+)?\s*)?(?:"
    r"hectares?|ha|kilograms?|kg|grams?|g|millilit(?:er|re)s?|ml|"
    r"lit(?:er|re)s?|l"
    r"))?(?!\w)",
    re.IGNORECASE,
)
_CHEMICAL_TERM = re.compile(
    r"\b(?:chemical\w*|fungicid\w*|pesticid\w*|herbicid\w*|"
    r"insecticid\w*|acaricid\w*|nematicid\w*|[a-z][a-z0-9-]*azole)\b",
    re.IGNORECASE,
)
_APPLICATION_ACTION = re.compile(
    r"\b(?:apply|applies|applied|applying|spray\w*|mix\w*|dilut\w*|"
    r"treat\w*|dose\w*|administer\w*|add\w*|use\w*)\b",
    re.IGNORECASE,
)


def input_guard_node(state: AgentState) -> dict[str, Any]:
    query = state.get("user_query", "")
    query = query if isinstance(query, str) else ""
    matched_patterns = [
        name for name, pattern in _INPUT_PATTERNS if pattern.search(query)
    ]
    metadata = dict(state.get("verification_metadata", {}))
    metadata.update(
        {
            "prompt_injection_detected": bool(matched_patterns),
            "prompt_injection_patterns": matched_patterns,
            "status": "refused" if matched_patterns else "pending",
        }
    )
    if matched_patterns:
        return {
            "drafted_advisory": SAFE_REFUSAL,
            "verification_status": "refused",
            "verification_metadata": metadata,
        }
    return {
        "verification_status": "pending",
        "verification_metadata": metadata,
    }


def _unsupported_dosages(text: str, source_contents: list[str]) -> list[str]:
    unsupported = []
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", text):
        sentence = sentence.strip()
        if not sentence:
            continue
        for amount in _DOSE_AMOUNT.finditer(sentence):
            chemicals = list(_CHEMICAL_TERM.finditer(sentence))
            actions = list(_APPLICATION_ACTION.finditer(sentence))
            if chemicals:
                related = min(
                    chemicals, key=lambda match: abs(match.start() - amount.start())
                )
                start = min(amount.start(), related.start())
                end = max(amount.end(), related.end())
                dosage_claim = sentence[start:end].strip()
            elif actions:
                related = min(
                    actions, key=lambda match: abs(match.start() - amount.start())
                )
                dosage_claim = sentence[min(amount.start(), related.start()) :].strip()
            else:
                dosage_claim = sentence
            dosage_claim = dosage_claim.rstrip(".,;:")
            if not any(
                dosage_claim.casefold() in content.casefold()
                for content in source_contents
            ):
                unsupported.append(dosage_claim)
    return unsupported


def verification_node(state: AgentState) -> dict[str, Any]:
    passages = list(state.get("retrieved_passages", []))
    source_contents = [str(passage.get("content", "")) for passage in passages]
    allowed_ids = {
        str(passage.get("citation_anchor") or passage.get("chunk_id"))
        for passage in passages
        if passage.get("citation_anchor") or passage.get("chunk_id")
    }
    advisory = str(state.get("drafted_advisory", "") or "").strip()
    if not passages and advisory == NO_SOURCE_ADVISORY:
        metadata = dict(state.get("verification_metadata", {}))
        metadata.update(
            {
                "citation_errors": 0,
                "ungrounded_claims": 0,
                "unsupported_chemical_dosages": 0,
                "status": "approved",
            }
        )
        return {
            "drafted_advisory": advisory,
            "verification_status": "approved",
            "verification_metadata": metadata,
        }

    verified_lines = []
    citation_errors = 0
    ungrounded_claims = 0
    unsupported_dosages = 0
    rewrite_needed = False
    for line in advisory.splitlines():
        if not line.strip():
            continue
        if line.strip() == NO_SOURCE_ADVISORY:
            verified_lines.append(line.strip())
            continue

        tags = list(_CITATION_TAG.finditer(line))
        citation_ids = list(dict.fromkeys(tag.group(1).strip() for tag in tags))
        unknown_ids = [citation_id for citation_id in citation_ids if citation_id not in allowed_ids]
        text = _CITATION_TAG.sub("", line).strip()
        if not text or not citation_ids or unknown_ids:
            citation_errors += len(unknown_ids) or 1
            ungrounded_claims += 1
            rewrite_needed = True
            continue

        unsupported = _unsupported_dosages(text, source_contents)
        if unsupported:
            unsupported_dosages += len(unsupported)
            rewrite_needed = True
            continue

        citations = " ".join(f"[Source: {citation_id}]" for citation_id in citation_ids)
        verified_lines.append(f"{text} {citations}")

    if rewrite_needed:
        verified_lines.append(SAFE_REWRITE)
    if not verified_lines:
        verified_lines.append(SAFE_REWRITE if advisory else NO_SOURCE_ADVISORY)
        rewrite_needed = bool(advisory)

    status = "rewritten" if rewrite_needed else "approved"
    metadata = dict(state.get("verification_metadata", {}))
    metadata.update(
        {
            "citation_errors": citation_errors,
            "ungrounded_claims": ungrounded_claims,
            "unsupported_chemical_dosages": unsupported_dosages,
            "status": status,
        }
    )
    return {
        "drafted_advisory": "\n".join(verified_lines),
        "verification_status": status,
        "verification_metadata": metadata,
    }
