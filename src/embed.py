"""Fixed public text representations for semantic assignment.

Every representation in this module is frozen before protected target records
are processed.  Lexical TF--IDF representations may be fitted only on public
atom text and optional public/disjoint auxiliary text.  The optional sentence-
transformer representation is a fixed pretrained public model and performs no
fitting on either auxiliary or protected target records.

The module deliberately exposes heterogeneous families so external-validity
claims need not rest on a single representation: word TF--IDF, character
TF--IDF, and an optional pretrained sentence encoder.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Literal, Protocol, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

EmbeddingFamily = Literal["tfidf_word", "tfidf_char", "sentence_transformer"]
SUPPORTED_EMBEDDING_FAMILIES: tuple[str, ...] = (
    "tfidf_word",
    "tfidf_char",
    "sentence_transformer",
)


class PublicEmbedder(Protocol):
    """Interface shared by sparse lexical and dense sentence representations."""

    family: str

    def transform(self, texts: Sequence[str]): ...

    def metadata(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class EmbeddingMetadata:
    family: str
    analyzer: str
    ngram_range: tuple[int, int]
    min_df: int
    max_features: int | None
    vocabulary_size: int
    output_dimension: int
    normalization: str
    fitted_on: str
    model_name: str | None = None
    model_revision: str | None = None


@dataclass
class TfidfEmbedder:
    """Thin wrapper around a frozen TF--IDF representation."""

    vectorizer: TfidfVectorizer
    family: str = "tfidf_word"
    fitted_on: str = "public_atom_text_only"

    @classmethod
    def fit(
        cls,
        texts: Iterable[str],
        *,
        family: Literal["tfidf_word", "tfidf_char"] = "tfidf_word",
        min_df: int = 1,
        max_features: int | None = 50000,
        word_ngram_range: tuple[int, int] = (1, 2),
        char_ngram_range: tuple[int, int] = (3, 5),
        fitted_on: str = "public_atom_text_only",
    ) -> "TfidfEmbedder":
        """Fit a public TF--IDF representation.

        ``tfidf_word`` uses lower-cased word unigrams and bigrams with English
        stop-word removal. ``tfidf_char`` uses character n-grams inside word
        boundaries, which is less sensitive to spelling and morphology.  Both
        are deterministic conditional on their public fitting corpus.
        """
        if family not in {"tfidf_word", "tfidf_char"}:
            raise ValueError("TfidfEmbedder supports only tfidf_word/tfidf_char")
        if min_df < 1:
            raise ValueError("min_df must be at least 1")
        if max_features is not None and max_features < 1:
            raise ValueError("max_features must be positive or None")

        if family == "tfidf_word":
            analyzer = "word"
            ngram_range = tuple(map(int, word_ngram_range))
            stop_words: str | None = "english"
            token_pattern = r"(?u)\b\w\w+\b"
        else:
            analyzer = "char_wb"
            ngram_range = tuple(map(int, char_ngram_range))
            stop_words = None
            token_pattern = None

        vec = TfidfVectorizer(
            lowercase=True,
            stop_words=stop_words,
            analyzer=analyzer,
            ngram_range=ngram_range,
            min_df=min_df,
            max_features=max_features,
            norm="l2",
            token_pattern=token_pattern,
            dtype=np.float64,
        )
        corpus = list(texts)
        if not corpus:
            raise ValueError("public embedding fitting corpus must be nonempty")
        vec.fit(corpus)
        return cls(vec, family=family, fitted_on=fitted_on)

    def transform(self, texts: Sequence[str]):
        return self.vectorizer.transform(list(texts))

    def metadata(self) -> dict[str, Any]:
        vec = self.vectorizer
        dimension = int(len(getattr(vec, "vocabulary_", {}) or {}))
        meta = EmbeddingMetadata(
            family=self.family,
            analyzer=str(vec.analyzer),
            ngram_range=tuple(map(int, vec.ngram_range)),
            min_df=int(vec.min_df),
            max_features=None if vec.max_features is None else int(vec.max_features),
            vocabulary_size=dimension,
            output_dimension=dimension,
            normalization=str(vec.norm),
            fitted_on=self.fitted_on,
        )
        return asdict(meta)


@dataclass
class SentenceTransformerEmbedder:
    """Fixed pretrained sentence encoder.

    This optional backend never fits on project records.  Model name, revision,
    device, trust setting, and local-files-only policy are recorded so runs are
    auditable.  Install ``requirements-semantic-embeddings.txt`` to enable it.
    """

    model: Any
    model_name: str
    model_revision: str | None
    batch_size: int = 64
    normalize_embeddings: bool = True
    device: str | None = None
    local_files_only: bool = False
    trust_remote_code: bool = False
    family: str = "sentence_transformer"

    @classmethod
    def load(
        cls,
        *,
        model_name: str,
        model_revision: str | None = None,
        batch_size: int = 64,
        normalize_embeddings: bool = True,
        device: str | None = None,
        local_files_only: bool = False,
        trust_remote_code: bool = False,
    ) -> "SentenceTransformerEmbedder":
        if not str(model_name).strip():
            raise ValueError("sentence-transformer model_name must be nonempty")
        if batch_size < 1:
            raise ValueError("sentence-transformer batch_size must be positive")
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "embedding_family='sentence_transformer' requires the optional "
                "sentence-transformers dependency. Install "
                "requirements-semantic-embeddings.txt."
            ) from exc

        kwargs: dict[str, Any] = {
            "trust_remote_code": bool(trust_remote_code),
            "local_files_only": bool(local_files_only),
        }
        if model_revision:
            kwargs["revision"] = str(model_revision)
        if device:
            kwargs["device"] = str(device)
        try:
            model = SentenceTransformer(str(model_name), **kwargs)
        except TypeError:  # Compatibility with older sentence-transformers.
            kwargs.pop("local_files_only", None)
            model = SentenceTransformer(str(model_name), **kwargs)
        return cls(
            model=model,
            model_name=str(model_name),
            model_revision=None if model_revision is None else str(model_revision),
            batch_size=int(batch_size),
            normalize_embeddings=bool(normalize_embeddings),
            device=None if device is None else str(device),
            local_files_only=bool(local_files_only),
            trust_remote_code=bool(trust_remote_code),
        )

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        encoded = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
        )
        arr = np.asarray(encoded, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError("sentence-transformer encoder returned a non-matrix output")
        return arr

    def metadata(self) -> dict[str, Any]:
        dimension = int(self.model.get_sentence_embedding_dimension())
        return {
            "family": self.family,
            "analyzer": "fixed_pretrained_sentence_encoder",
            "ngram_range": None,
            "min_df": None,
            "max_features": None,
            "vocabulary_size": None,
            "output_dimension": dimension,
            "normalization": "l2" if self.normalize_embeddings else "none",
            "fitted_on": "fixed_pretrained_public_model_no_project_fit",
            "model_name": self.model_name,
            "model_revision": self.model_revision,
            "device": self.device,
            "local_files_only": self.local_files_only,
            "trust_remote_code": self.trust_remote_code,
            "batch_size": self.batch_size,
        }


def fit_public_embedder(
    texts: Iterable[str],
    *,
    family: EmbeddingFamily = "tfidf_word",
    min_df: int = 1,
    max_features: int | None = 50000,
    word_ngram_range: tuple[int, int] = (1, 2),
    char_ngram_range: tuple[int, int] = (3, 5),
    fitted_on: str = "public_atom_text_only",
    sentence_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    sentence_model_revision: str | None = None,
    sentence_batch_size: int = 64,
    sentence_normalize_embeddings: bool = True,
    sentence_device: str | None = None,
    sentence_local_files_only: bool = False,
    sentence_trust_remote_code: bool = False,
) -> PublicEmbedder:
    """Build one frozen public representation.

    The ``texts`` iterator is consumed only by TF--IDF families.  A sentence
    transformer is loaded as a fixed pretrained model and does not fit on these
    texts, which prevents accidental target-data adaptation.
    """
    if family == "sentence_transformer":
        return SentenceTransformerEmbedder.load(
            model_name=sentence_model_name,
            model_revision=sentence_model_revision,
            batch_size=sentence_batch_size,
            normalize_embeddings=sentence_normalize_embeddings,
            device=sentence_device,
            local_files_only=sentence_local_files_only,
            trust_remote_code=sentence_trust_remote_code,
        )
    if family not in {"tfidf_word", "tfidf_char"}:
        raise ValueError(
            f"Unsupported embedding family {family!r}; expected one of "
            f"{SUPPORTED_EMBEDDING_FAMILIES}"
        )
    return TfidfEmbedder.fit(
        texts,
        family=family,
        min_df=min_df,
        max_features=max_features,
        word_ngram_range=word_ngram_range,
        char_ngram_range=char_ngram_range,
        fitted_on=fitted_on,
    )


def atom_text(atom: dict) -> str:
    """Construct the public textual representation of an atom."""
    return f"{atom.get('label', '')}. {atom.get('description', '')}"
