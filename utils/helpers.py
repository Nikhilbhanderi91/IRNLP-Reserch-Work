"""
================================================
utils/helpers.py
Shared utility / helper functions
================================================
"""

import re
import json
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional


# ─── Text helpers ────────────────────────────────────────────────────────────

def normalize_text(text: str) -> str:
    """Remove control characters and normalize unicode whitespace."""
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def truncate(text: str, max_chars: int = 300, suffix: str = "…") -> str:
    """Safely truncate a string to *max_chars* characters."""
    return text if len(text) <= max_chars else text[:max_chars].rstrip() + suffix


def flatten_list(nested: List[Any]) -> List[Any]:
    """Recursively flatten an arbitrarily nested list."""
    result: List[Any] = []
    for item in nested:
        if isinstance(item, list):
            result.extend(flatten_list(item))
        else:
            result.append(item)
    return result


def deduplicate(items: List[str], case_sensitive: bool = False) -> List[str]:
    """Return unique items preserving insertion order."""
    seen = set()
    out = []
    for item in items:
        key = item if case_sensitive else item.lower()
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out


# ─── File helpers ─────────────────────────────────────────────────────────────

def ensure_dir(path: str | Path) -> Path:
    """Create directory (and parents) if it does not exist, return Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(data: Dict | List, path: str | Path, indent: int = 2) -> None:
    """Serialise *data* to a JSON file."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=indent)


def load_json(path: str | Path) -> Any:
    """Load and return JSON from *path*."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ─── Scoring helpers ──────────────────────────────────────────────────────────

def top_n(items: List[Dict], key: str = "score", n: int = 15, ascending: bool = False) -> List[Dict]:
    """Return top-N dicts sorted by *key*."""
    return sorted(items, key=lambda x: x.get(key, 0), reverse=not ascending)[:n]


def merge_keyword_dicts(
    *lists: List[Dict],
    word_key: str = "word",
    score_key: str = "score",
    top: int = 30,
) -> List[Dict]:
    """
    Merge multiple keyword-score lists, averaging scores for duplicates
    and returning the top-N combined list.
    """
    aggregated: Dict[str, List[float]] = {}
    for lst in lists:
        for item in lst:
            w = item.get(word_key, "").lower().strip()
            s = float(item.get(score_key, 0.0))
            if w:
                aggregated.setdefault(w, []).append(s)
    merged = [
        {word_key: w, score_key: sum(scores) / len(scores)}
        for w, scores in aggregated.items()
    ]
    return top_n(merged, key=score_key, n=top)


def safe_get(d: Optional[Dict], *keys: str, default: Any = None) -> Any:
    """Safely navigate a nested dict with *keys*."""
    node = d
    for k in keys:
        if not isinstance(node, dict):
            return default
        node = node.get(k, default)
    return node
