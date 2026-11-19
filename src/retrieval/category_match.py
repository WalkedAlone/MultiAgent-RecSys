"""Category matching for deterministic product filters.

Plain substring matching is too loose for structured intents: ``laptop``
matches ``Laptop Accessories`` and ``phone`` matches ``Headphones``.  This
module matches whole category path segments (with simple singular/plural
folding) and only allows accessory-only segments when the user explicitly
asked for an accessory.

The helper is intentionally dependency-free so the retrieval filter, the
catalog fallback, the product tools and the Critic can share exactly the
same semantics.
"""

from __future__ import annotations

import re
from typing import Any, Mapping

# Stems of words that describe accessories rather than the device itself.
# Kept focused on category-path vocabulary; product titles are not matched
# with this guard because titles such as "charging case" can legitimately
# belong to a real device (e.g. earbuds).
_ACCESSORY_STEMS = {
    "accessory",
    "adapter",
    "bag",
    "cable",
    "case",
    "charger",
    "cover",
    "holder",
    "kit",
    "mount",
    "part",
    "protector",
    "replacement",
    "skin",
    "sleeve",
    "stand",
    "strap",
}

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _stem(token: str) -> str:
    """Fold common English plurals; good enough for category names."""
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith("ses"):
        # lenses -> lens; cases -> case; classes -> class.
        return token[:-2] if token[-3] == "s" else token[:-1]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def category_tokens(text: Any) -> set[str]:
    tokens = _TOKEN_RE.findall(str(text or "").casefold())
    return {_stem(token) for token in tokens}


def _has_accessory(tokens: set[str]) -> bool:
    return bool(tokens & _ACCESSORY_STEMS)


def category_matches(
    category: str | None,
    metadata: Mapping[str, Any],
) -> bool:
    """Return whether ``metadata`` belongs to the requested category.

    Matching rules:
    * a query token set must be contained in one category-path segment or in
      the main category's non-accessory tokens;
    * a path segment that only describes accessories (``Laptop Accessories``)
      is ignored unless the query itself contains an accessory word;
    * accessory queries such as ``laptop bag`` may match across the whole
      category path.

    Empty category means "no category filter" and matches everything.
    """
    query_words = category_tokens(category)
    if not query_words:
        return True

    main_words = category_tokens(metadata.get("main_category"))
    segments = [
        segment
        for segment in str(metadata.get("category_path") or "").split(">")
        if segment.strip()
    ]
    segment_words = [category_tokens(segment) for segment in segments]
    query_has_accessory = _has_accessory(query_words)

    for words in segment_words:
        if not query_words <= words:
            continue
        if not query_has_accessory and _has_accessory(words):
            continue
        return True

    main_core = main_words - _ACCESSORY_STEMS
    if query_words <= main_core:
        return True
    if query_has_accessory and query_words <= main_words:
        return True

    if query_has_accessory:
        all_path_words: set[str] = set(main_words)
        for words in segment_words:
            all_path_words |= words
        if query_words <= all_path_words:
            return True

    return False
