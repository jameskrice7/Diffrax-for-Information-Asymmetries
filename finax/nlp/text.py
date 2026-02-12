"""NLP-style feature tools designed for tabular text columns."""

from __future__ import annotations

import re
from collections import Counter
from typing import Iterable

import pandas as pd

_TOKEN_RE = re.compile(r"[a-zA-Z']+")


def normalize_text(text: str) -> str:
    """Normalize text into a lowercase token-friendly form."""

    return " ".join(_TOKEN_RE.findall((text or "").lower()))


def keyword_intensity(text_series: pd.Series, keywords: Iterable[str]) -> pd.Series:
    """Score each row by normalized keyword occurrence density."""

    keyword_set = {k.lower() for k in keywords}
    scores: list[float] = []
    for value in text_series.fillna(""):
        tokens = normalize_text(value).split()
        if not tokens:
            scores.append(0.0)
            continue
        matches = sum(1 for token in tokens if token in keyword_set)
        scores.append(matches / len(tokens))
    return pd.Series(scores, index=text_series.index, name="keyword_intensity")


def bag_of_words(text_series: pd.Series, *, max_features: int = 100) -> pd.DataFrame:
    """Build a lightweight bag-of-words matrix without external dependencies."""

    tokenized = [normalize_text(value).split() for value in text_series.fillna("")]
    vocab = [w for w, _ in Counter(token for row in tokenized for token in row).most_common(max_features)]

    data = []
    for row in tokenized:
        counts = Counter(row)
        data.append([counts.get(word, 0) for word in vocab])

    return pd.DataFrame(data, index=text_series.index, columns=[f"bow_{w}" for w in vocab])


def text_feature_frame(
    df: pd.DataFrame,
    *,
    text_column: str,
    keywords: Iterable[str],
    max_features: int = 25,
) -> pd.DataFrame:
    """Generate intuitive text-derived features ready for simulation/modeling."""

    text_series = df[text_column].astype(str)
    features = bag_of_words(text_series, max_features=max_features)
    features["keyword_intensity"] = keyword_intensity(text_series, keywords)
    features["text_length"] = text_series.fillna("").str.len()
    return features
