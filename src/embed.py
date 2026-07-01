"""Provide the fixed TF-IDF representation used by semantic assignment.

In a DP-SPIN private run, the fitted vectorizer is part of the public semantic contribution map. 
Its vocabulary and inverse-document-frequency weights must be estimated from atom text and optional public or auxiliary text, not from the protected target split.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


@dataclass
class TfidfEmbedder:
    """Thin wrapper around a fitted TF-IDF vectorizer used for assignment.
    """
    vectorizer: TfidfVectorizer

    @classmethod
    def fit(cls, texts: Sequence[str], min_df: int = 1, max_features: int | None = 50000) -> "TfidfEmbedder":
        """Fit a public TF-IDF representation on supplied public/auxiliary texts.

        ``texts`` may be an iterator.  Do not convert it to a Python list here;
        large public/auxiliary corpora can contain millions of records.
        """
        vec = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=min_df,
            max_features=max_features,
            norm="l2",
        )
        vec.fit(texts)
        return cls(vec)

    def transform(self, texts: Sequence[str]):
        """Transform texts into the fitted TF-IDF space.
        """
        return self.vectorizer.transform(list(texts))


def atom_text(atom: dict) -> str:
    """Construct the textual representation of an atom label and description.
    """
    return f"{atom.get('label', '')}. {atom.get('description', '')}"
