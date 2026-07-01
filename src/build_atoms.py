"""Build decoder-visible semantic atom dictionaries from public or auxiliary data.

DP-SPIN separates atom naming from atom admission. 
Atom labels and descriptions must be fixed independently of the protected target split because they are visible to the decoder. 
Accepted sources are public data, benchmark metadata, or a disjoint auxiliary split and rejects ``source_role="private_target"``.

Two construction modes are available:

* ``metadata`` combines configured metadata fields, such as CFPB product, issue, and sub-issue fields;
* ``text_phrases`` extracts and ranks phrases from public or auxiliary text. Optional deduplication or LLM refinement operates only on auxiliary candidate labels and their auxiliary support counts.

A deployment that derives decoder-visible labels from protected records requires a separate differentially private text-release mechanism and corresponding privacy accounting.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Literal
import json
import re


from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS

from .utils import read_jsonl, write_json, ensure_dir, text_field
from .llm_client import maybe_llm

SourceRole = Literal["public", "auxiliary", "benchmark_public", "private_target"]
ReviewDomain = Literal["generic", "amazon", "yelp", "auto"]

PRIVACY_WARNING = """
PRIVACY WARNING:
  build_atoms.py creates decoder-visible atom labels and support diagnostics.
  It is safe for public data, public metadata, or an auxiliary split outside the
  protected target corpus. It must not be used non-privately on the same private
  target records protected by the DP sketch theorem. If atom labels are
  discovered from the protected corpus, atom discovery is itself a release and
  needs its own DP mechanism and privacy accounting.
""".strip()

# Review datasets such as Amazon and Yelp do not provide a CFPB-style issue
# taxonomy.  In ``text_phrases`` mode, the atom builder therefore extracts
# review-theme phrases from an auxiliary/public split.  The following filters
# remove frequent review boilerplate and generic sentiment expressions that are
# not useful semantic atoms. The filters are conservative and deterministic.
REVIEW_BOILERPLATE_TOKENS = {
    "review", "reviews", "reviewer", "reviewers", "received", "receive",
    "sample", "samples", "free", "complimentary", "discount", "discounted",
    "honest", "unbiased", "exchange", "testing", "tested", "vine",
}

REVIEW_GENERIC_TOKENS = {
    "great", "good", "nice", "excellent", "amazing", "awesome", "perfect",
    "wonderful", "best", "better", "bad", "poor", "okay", "ok", "fine",
    "like", "liked", "love", "loved", "really", "very", "just", "pretty", "quickly",
    "little", "bit", "lot", "lots", "thing", "things", "stuff", "one",
    "two", "got", "get", "gets", "getting", "make", "makes", "sure",
    "use", "used", "using", "work", "works", "worked", "does", "did",
    "don", "dont", "didn", "didnt", "doesn", "doesnt", "isn", "isnt",
    "product", "products", "item", "items", "purchase", "purchased", "buy",
    "bought", "ordered", "order", "recommend", "recommended", "highly",
    "try", "tried", "need", "needed", "want", "wanted",
}

# If a phrase contains at least one of these tokens, it is more likely to be a
# concrete review theme rather than generic praise or boilerplate.  The list is
# shared across the Yelp and Amazon review domains.
REVIEW_THEME_TOKENS = {
    # Amazon / product-experience themes
    "skin", "hair", "face", "cream", "lotion", "serum", "oil", "scent",
    "smell", "fragrance", "odor", "dry", "sensitive", "irritation", "rash",
    "acne", "moisturizer", "moisturizing", "shampoo", "conditioner", "curls",
    "curly", "fine", "thick", "thin", "pump", "dispenser", "bottle", "tube",
    "packaging", "package", "shipping", "delivery", "battery", "charger",
    "charge", "screen", "sound", "audio", "size", "fit", "durable", "durability",
    "broken", "defective", "leak", "leaking", "absorbs", "absorb", "apply",
    "application", "price", "value", "money", "waste", "return", "refund",
    "customer", "support", "service", "instructions", "setup", "quality",
    # Yelp / local-service themes
    "food", "meal", "dish", "dishes", "menu", "taste", "flavor", "portion",
    "portions", "server", "waiter", "waitress", "staff", "service", "reservation",
    "wait", "waiting", "line", "table", "seating", "clean", "cleanliness",
    "dirty", "noise", "noisy", "environment", "atmosphere", "parking", "location",
    "drink", "drinks", "coffee", "pizza", "burger", "sushi", "breakfast",
    "lunch", "dinner", "restaurant", "bar", "salon", "appointment",
}


REVIEW_ALLOWED_BIGRAMS = {
    ("dry", "skin"), ("sensitive", "skin"), ("oily", "skin"), ("soft", "skin"),
    ("smooth", "skin"), ("skin", "irritation"), ("skin", "rash"),
    ("skin", "cream"), ("skin", "care"),
    ("fine", "hair"), ("curly", "hair"), ("long", "hair"), ("hair", "loss"),
    ("hair", "growth"), ("pump", "dispenser"), ("strong", "smell"),
    ("absorbs", "quickly"), ("easy", "apply"), ("waste", "money"),
    ("great", "value"), ("good", "value"), ("customer", "service"),
    ("customer", "support"), ("shipping", "delay"), ("delivery", "delay"),
    ("battery", "life"), ("sound", "quality"), ("food", "quality"),
    ("friendly", "staff"), ("poor", "service"), ("slow", "service"),
    ("long", "wait"), ("wait", "time"), ("high", "price"),
    ("reservation", "problem"), ("parking", "problem"),
    ("noisy", "environment"), ("cleanliness", "concern"),
}

REVIEW_HEAD_TOKENS = {
    "skin", "hair", "irritation", "rash", "dispenser", "smell", "money",
    "value", "service", "support", "delay", "life", "quality", "staff",
    "wait", "time", "price", "problem", "environment", "concern",
    "packaging", "shipping", "delivery", "battery", "sound", "food",
}

REVIEW_BAD_BIGRAMS = {
    ("skin", "pump"), ("skin", "sensitive"), ("skin", "dry"), ("smell", "skin"), ("money", "dry"), ("hair", "waste"),
    ("price", "pump"), ("dispenser", "customer"), ("service", "strong"),
    ("irritation", "dry"), ("quality", "sensitive"), ("apply", "absorbs"),
    ("cream", "absorbs"), ("cream", "great"), ("helped", "dry"),
    ("helped", "fine"), ("lotion", "easy"), ("dispenser", "bad"),
    ("dispenser", "broke"),
}

REVIEW_BANNED_PHRASES = {
    "really like", "great product", "good product", "love product",
    "works great", "work great", "highly recommend", "good quality",
    "little bit", "feel like", "make sure", "pretty good", "don know",
    "does work", "honest unbiased", "honest review", "received product",
    "free product", "just received", "great price", "long time", "easy use",
}

# Additional stop words passed directly to CountVectorizer.  This prevents many
# generic unigrams and generic n-grams from becoming candidates in the first
# place.  We keep thematic words such as ``service`` and ``price`` because they
# are meaningful atoms for review data.
REVIEW_EXTRA_STOP_WORDS = REVIEW_GENERIC_TOKENS | REVIEW_BOILERPLATE_TOKENS

# Domain-specific validation and canonicalization for paper-grade review atoms.
# These rules use only candidate phrase labels and auxiliary support counts.
YELP_REJECT_PHRASES = {
    "favorite restaurant", "fast food", "food restaurant", "restaurant food",
    "service food", "food service", "customer food", "food customer",
    "staff food", "food staff", "wait staff", "wait come", "food wasn",
    "taste buds", "new restaurant", "staff members", "lunch dinner",
    "breakfast lunch", "food drinks", "drink orders", "bar food",
}
YELP_REJECT_TOKENS = {
    # Product/e-commerce artifacts that are not restaurant-review atoms.
    "skin", "hair", "cream", "lotion", "serum", "pump", "dispenser",
    "battery", "charger", "screen", "sound", "audio", "shipping", "delivery"
}
YELP_CANONICAL = {
    "quality food": "food quality",
    "food quality": "food quality",
    "food fresh": "fresh food",
    "food delicious": "delicious food",
    "long wait": "long wait",
    "wait time": "long wait",
    "waiting time": "long wait",
    "minute wait": "long wait",
    "food took": "long wait",
    "slow service": "slow service",
    "poor service": "poor service",
    "terrible service": "terrible service",
    "quick service": "fast service",
    "fast service": "fast service",
    "friendly staff": "friendly staff",
    "staff friendly": "friendly staff",
    "friendly service": "friendly staff",
    "customer service": "customer service",
    "food came": "service speed",
    "food comes": "service speed",
    "food arrived": "service speed",
    "portion size": "portion size",
    "parking problem": "parking problem",
    "noisy environment": "noisy environment",
    "reasonable price": "reasonable price",
    "price point": "price",
    "high price": "high price",
    "low price": "low price",
    "delicious food": "delicious food",
    "seating area": "seating",
}
YELP_ALLOWED_HEADS = {
    "food", "service", "staff", "server", "waiter", "waitress", "wait", "price",
    "portion", "size", "parking", "environment", "atmosphere", "cleanliness",
    "reservation", "table", "seating", "menu", "taste", "flavor", "drink",
    "coffee", "breakfast", "lunch", "dinner", "restaurant", "location",
    "speed", "quality", "place", "shop", "truck", "special"
}
YELP_ALLOWED_MODIFIERS = {
    "friendly", "slow", "fast", "poor", "great", "bad", "long", "short",
    "delicious", "fresh", "reasonable", "high", "low", "clean", "dirty",
    "noisy", "large", "small", "outdoor", "indoor", "quick", "terrible",
    "excellent", "good", "bad", "hot", "cold"
}
YELP_CUISINE_OR_CATEGORY_MODIFIERS = {
    "mexican", "chinese", "indian", "thai", "italian", "japanese",
    "korean", "vietnamese", "mediterranean", "greek", "sushi", "pizza",
    "burger", "bbq", "seafood", "breakfast", "lunch", "dinner",
    "coffee", "bar", "food", "taco", "ramen", "noodle", "vegan",
    "vegetarian", "brunch", "bakery", "dessert"
}
YELP_MENU_HEADS = {
    "food", "restaurant", "shop", "truck", "place", "bar", "menu",
    "special", "burrito", "sandwich", "pizza", "burger", "sushi",
    "taco", "ramen", "noodle", "coffee", "drink", "drinks"
}

AMAZON_REJECT_PHRASES = {
    # Cross-domain restaurant/local-service or electronics/audio artifacts.
    "wait time", "long wait", "food quality", "fast food", "portion size",
    "favorite restaurant", "quality food", "sound quality", "hair sound",
    "fast service",
    # Awkward, non-canonical, or non-interpretable All_Beauty phrase forms.
    "leaves hair", "left hair", "hold hair", "hair hold", "hair left",
    "daughter hair", "keeps hair", "type hair", "skin quality",
    "easy hair", "natural skin", "skin hair", "skin skin", "strong hair",
    "easy skin", "fast hair", "wet skin", "easy pump",
    # Tail scent/battery variants that are too weak or ambiguous as atom labels.
    "soft scent", "soft smell", "soft fragrance", "sensitive fragrance",
    "low battery",
}
AMAZON_REJECT_TOKENS = {
    # Restaurant/local-service artifacts that should not be All_Beauty atoms.
    "food", "restaurant", "wait", "waiting", "waiter", "waitress", "server",
    "table", "parking", "portion", "pizza", "burger", "sushi", "breakfast",
    "lunch", "dinner", "atmosphere", "sound", "audio"
}
AMAZON_CANONICAL = {
    "wasted money": "waste money",
    "waste money": "waste money",
    "waste time": "waste money",
    "loss hair": "hair loss",
    "high price": "high price",
    "low price": "low price",
    "reasonable price": "reasonable price",
    "good value": "reasonable price",
    "great value": "reasonable price",
    "quality hair": "hair quality",
    "length hair": "hair length",
    "curl hair": "curly hair",
    "curly hair": "curly hair",
    "dry hair": "dry hair",
    "soft hair": "soft hair",
    "natural hair": "natural hair",
    "human hair": "human hair",
    "synthetic hair": "synthetic hair",
    "wet hair": "wet hair",
    "straight hair": "straight hair",
    "wavy hair": "wavy hair",
    "hair loss": "hair loss",
    "hair growth": "hair growth",
    "skin irritation": "skin irritation",
    "irritate skin": "skin irritation",
    "skin rash": "skin rash",
    "dry skin": "dry skin",
    "sensitive skin": "sensitive skin",
    "oily skin": "oily skin",
    "smooth skin": "smooth skin",
    "soft skin": "soft skin",
    "dead skin": "dead skin",
    "leaves skin": "skin feel",
    "skin cream": "skin cream",
    "skin care": "skin care",
    "pump dispenser": "pump dispenser",
    "strong smell": "strong smell",
    "strong scent": "strong smell",
    "strong fragrance": "strong smell",
    "chemical smell": "chemical smell",
    "chemical scent": "chemical smell",
    "natural smell": "natural scent",
    "natural scent": "natural scent",
    "natural fragrance": "natural scent",
    "absorbs quickly": "absorbs quickly",
    "easy apply": "easy application",
    "easy application": "easy application",
    "smooth application": "smooth application",
    "fast shipping": "fast shipping",
    "fast delivery": "fast delivery",
    "customer service": "customer service",
    "customer support": "customer support",
    "customer care": "customer support",
    "battery life": "battery life",
    "long battery": "battery life",
    "high quality": "product quality",
    "product quality": "product quality",
    "low quality": "low quality",
}
AMAZON_ALLOWED_HEADS = {
    "skin", "hair", "irritation", "rash", "cream", "care", "smell", "scent",
    "fragrance", "dispenser", "pump", "shipping", "delivery", "price", "money",
    "value", "service", "support", "quality", "battery", "packaging",
    "application", "product", "life", "loss", "growth", "length"
}
AMAZON_ALLOWED_MODIFIERS = {
    "dry", "sensitive", "oily", "smooth", "soft", "dead", "irritate", "curly",
    "long", "short", "natural", "human", "synthetic", "wet", "straight", "wavy",
    "strong", "chemical", "fast", "high", "low", "reasonable", "easy", "customer",
    "pump", "skin", "product", "waste", "wasted", "absorbs", "battery"
}

def _resolve_review_domain(review_domain: str, output_path: str = "", records_path: str = "") -> str:
    d = (review_domain or "generic").lower().strip()
    if d != "auto":
        return d
    joined = f"{output_path} {records_path}".lower()
    if "yelp" in joined:
        return "yelp"
    if "amazon" in joined or "all_beauty" in joined or "all-beauty" in joined:
        return "amazon"
    return "generic"

def _canonicalize_domain_phrase(phrase: str, domain: str) -> str:
    """Map a candidate phrase to the deterministic domain-specific display label."""
    phrase = re.sub(r"\s+", " ", phrase.lower().strip())
    mapping = AMAZON_CANONICAL if domain == "amazon" else YELP_CANONICAL if domain == "yelp" else {}
    if phrase in mapping:
        return mapping[phrase]
    toks = _phrase_tokens(phrase)
    if len(toks) == 2:
        a, b = toks
        if domain == "yelp":
            if (a, b) == ("quality", "food"):
                return "food quality"
            if a == "food" and b in {"fresh", "delicious"}:
                return f"{b} food"
        if domain == "amazon":
            if (a, b) == ("quality", "hair"):
                return "hair quality"
            if (a, b) == ("length", "hair"):
                return "hair length"
    return phrase

def _is_domain_valid_review_phrase(phrase: str, domain: str) -> bool:
    """Domain-specific validity check for decoder-visible review atom labels."""
    phrase = re.sub(r"\s+", " ", phrase.lower().strip())
    toks = _phrase_tokens(phrase)
    if not toks:
        return False
    if domain == "generic":
        return True
    if domain == "yelp":
        if phrase in YELP_REJECT_PHRASES:
            return False
        if any(t in YELP_REJECT_TOKENS for t in toks):
            return False
        if phrase in YELP_CANONICAL.values():
            return True
        if len(toks) == 2:
            a, b = toks
            if a in YELP_CUISINE_OR_CATEGORY_MODIFIERS and b in YELP_MENU_HEADS:
                return True
            if b in YELP_ALLOWED_HEADS and (a in YELP_ALLOWED_MODIFIERS or a in YELP_ALLOWED_HEADS):
                return True
            if phrase in {"food quality", "service speed", "fresh food", "delicious food"}:
                return True
            return False
        return False
    if domain == "amazon":
        if phrase in AMAZON_REJECT_PHRASES:
            return False
        if any(t in AMAZON_REJECT_TOKENS for t in toks):
            return False
        if phrase in AMAZON_CANONICAL.values():
            return True
        if len(toks) == 2:
            a, b = toks
            if b in AMAZON_ALLOWED_HEADS and a in AMAZON_ALLOWED_MODIFIERS:
                return True
            return False
        return False
    return True

def _aggregate_domain_canonical_counts(items: list[tuple[str, int]], domain: str) -> list[tuple[str, int]]:
    """Canonicalize, validate, aggregate support, and rank domain-specific atoms."""
    counts: Counter[str] = Counter()
    for phrase, count in items:
        phrase = re.sub(r"\s+", " ", str(phrase).lower().strip())
        if not phrase:
            continue
        # Phrases explicitly rejected for the domain are never used as atom
        # evidence.  This prevents low-quality surface forms such as ``easy
        # pump`` or cross-domain phrases such as ``fast service`` from being
        # rescued by a later canonical label.
        if domain == "yelp" and phrase in YELP_REJECT_PHRASES:
            continue
        if domain == "amazon" and phrase in AMAZON_REJECT_PHRASES:
            continue
        # Canonicalization is applied before the domain check so acceptable
        # reversals such as ``quality hair`` can become ``hair quality`` rather
        # than being discarded as awkward surface forms.
        canon = _canonicalize_domain_phrase(phrase, domain)
        if not canon or not _is_domain_valid_review_phrase(canon, domain):
            continue
        counts[canon] += int(count)
    return sorted(counts.items(), key=lambda kv: (-int(kv[1]), -_review_phrase_quality(kv[0]), len(kv[0]), kv[0]))


def _phrase_tokens(phrase: str) -> list[str]:
    return [t for t in re.split(r"\s+", phrase.lower().strip()) if t]




def _review_phrase_quality(phrase: str) -> int:
    """Assign a deterministic ranking tier to a candidate review-theme phrase.
    
    The tier uses only the candidate label and separates interpretable theme phrases
    from generic or accidental n-grams.
    """
    toks = _phrase_tokens(phrase)
    if len(toks) == 2:
        pair = (toks[0], toks[1])
        if pair in REVIEW_BAD_BIGRAMS:
            return 0
        if pair in REVIEW_ALLOWED_BIGRAMS:
            return 3
        if toks[1] in REVIEW_HEAD_TOKENS and toks[0] not in REVIEW_GENERIC_TOKENS:
            return 2
        if any(t in REVIEW_THEME_TOKENS for t in toks) and not any(t in REVIEW_GENERIC_TOKENS for t in toks):
            return 1
        return 0
    if len(toks) == 3:
        # For trigrams, require that at least one adjacent bigram is a known or
        # plausible theme phrase.  This avoids ranking accidental cross-topic
        # sequences highly.
        q1 = _review_phrase_quality(" ".join(toks[:2]))
        q2 = _review_phrase_quality(" ".join(toks[1:]))
        return max(0, min(2, max(q1, q2)))
    return 0

def _is_valid_review_phrase(phrase: str, require_theme_token: bool = True) -> bool:
    """Test whether an auxiliary phrase is suitable as a decoder-visible review atom.
    
    The deterministic filter removes review boilerplate and sentiment-only phrases.
    It operates on public or auxiliary candidates and is not a privacy mechanism.
    """
    phrase = phrase.lower().strip()
    if not phrase or phrase in REVIEW_BANNED_PHRASES:
        return False
    if not re.fullmatch(r"[a-z0-9_\- ]+", phrase):
        return False
    tokens = _phrase_tokens(phrase)
    if len(tokens) < 2:
        return False
    if len(tokens) > 4:
        return False
    if any(t in REVIEW_BOILERPLATE_TOKENS for t in tokens):
        return False
    if any(t in ENGLISH_STOP_WORDS for t in tokens):
        return False
    if all(t in REVIEW_GENERIC_TOKENS for t in tokens):
        return False
    if tokens[0] in REVIEW_GENERIC_TOKENS and len(tokens) <= 2:
        # Reject sentiment-led two-word boilerplate such as ``great product``.
        # Meaningful phrases such as ``easy apply`` survive because ``apply`` is
        # a theme token and the phrase is not in the banned list only if users
        # explicitly allow theme-token rescue below.
        if not (tokens[0] == "easy" and any(t in REVIEW_THEME_TOKENS for t in tokens[1:])):
            return False
    if tokens[-1] in {"thing", "things", "stuff", "one", "product", "item"}:
        return False
    if require_theme_token and not any(t in REVIEW_THEME_TOKENS for t in tokens):
        return False
    # Avoid phrases that are mostly generic words with one weak token.
    generic_count = sum(t in REVIEW_GENERIC_TOKENS for t in tokens)
    if generic_count >= max(1, len(tokens) - 1) and not any(t in REVIEW_THEME_TOKENS for t in tokens):
        return False
    if _review_phrase_quality(phrase) <= 0:
        return False
    return True




def _review_phrase_analyzer(text: str, ngram_min: int, ngram_max: int, require_theme_token: bool) -> list[str]:
    """Extract candidate review-theme phrases within local sentence boundaries."""
    text = str(text).lower()
    phrases: list[str] = []
    # Split on punctuation that usually marks phrase boundaries.  This avoids
    # artificial n-grams such as ``skin pump`` across sentence boundaries.
    segments = re.split(r"[.!?;:\n\r]+", text)
    for seg in segments:
        tokens = re.findall(r"[a-z][a-z0-9_-]{2,}", seg)
        if not tokens:
            continue
        for n in range(int(ngram_min), int(ngram_max) + 1):
            if n <= 0 or len(tokens) < n:
                continue
            for i in range(0, len(tokens) - n + 1):
                phrase = " ".join(tokens[i : i + n])
                if _is_valid_review_phrase(phrase, require_theme_token=require_theme_token):
                    phrases.append(phrase)
    return phrases

def _rank_review_phrase_items(items: list[tuple[str, int]]) -> list[tuple[str, int]]:
    """Rank review phrases by auxiliary support with deterministic quality tie-breaks."""
    def key(kv: tuple[str, int]):
        phrase, count = kv
        toks = _phrase_tokens(phrase)
        quality = _review_phrase_quality(phrase)
        theme_hits = sum(t in REVIEW_THEME_TOKENS for t in toks)
        two_word_bonus = 1 if len(toks) == 2 else 0
        return (-int(count), -quality, -theme_hits, -two_word_bonus, len(phrase), phrase)
    return sorted(items, key=key)


def _check_source_role(source_role: SourceRole) -> None:
    if source_role == "private_target":
        raise ValueError(
            "Refusing to build decoder-visible atoms from private target records. "
            "Use --source-role public, auxiliary, or benchmark_public, or add a "
            "separate DP atom-discovery mechanism."
        )


def _valid_value(v) -> bool:
    if v is None:
        return False
    s = str(v).strip()
    return bool(s) and s.lower() not in {"nan", "none", "null"}


def _sorted_count_items(counter: Counter, max_atoms: int | None):
    """Sort by decreasing count and deterministic lexical key."""
    items = sorted(counter.items(), key=lambda kv: (-int(kv[1]), str(kv[0])))
    return items[:max_atoms] if max_atoms is not None else items



def _normalize_phrase_for_merge(s: str) -> str:
    """Normalize a phrase for deterministic auxiliary-only deduplication."""
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9_\-\s]", " ", s)
    tokens = [t for t in re.split(r"\s+", s) if t]
    # Conservative suffix normalization. This is not linguistic stemming; it is
    # only used to merge obvious plural variants in auxiliary candidate labels.
    out = []
    for t in tokens:
        if len(t) > 4 and t.endswith("ies"):
            t = t[:-3] + "y"
        elif len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
            t = t[:-1]
        out.append(t)
    return " ".join(out)


def _deduplicate_phrase_candidates(items: list[tuple[str, int]]) -> list[tuple[str, int]]:
    """Merge candidate phrases with the same normalized auxiliary form.
    
    The representative is the highest-support phrase, with deterministic lexical
    tie-breaking. Support counts are summed across merged forms.
    """
    merged: dict[str, dict] = {}
    for phrase, count in items:
        key = _normalize_phrase_for_merge(phrase)
        if not key:
            continue
        cur = merged.get(key)
        if cur is None:
            merged[key] = {"label": phrase, "count": int(count), "support_sum": int(count)}
        else:
            cur["support_sum"] += int(count)
            # Pick canonical label by max individual support, then shorter label,
            # then lexical order. Keep summed support for ranking.
            cand = (int(count), -len(phrase), phrase)
            old = (int(cur["count"]), -len(cur["label"]), cur["label"])
            if cand > old:
                cur["label"] = phrase
                cur["count"] = int(count)
    return sorted(
        [(v["label"], int(v["support_sum"])) for v in merged.values()],
        key=lambda kv: (-int(kv[1]), str(kv[0])),
    )


def _parse_llm_json_object(text: str) -> dict:
    """Parse the JSON object returned by an auxiliary-only label-refinement call."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    # If the response contains surrounding text, parse the first JSON-object block.
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        obj = json.loads(m.group(0))
        if isinstance(obj, dict):
            return obj
    raise ValueError("LLM refinement did not return a valid JSON object")


def _llm_refine_phrase_candidates(
    items: list[tuple[str, int]],
    max_atoms: int | None,
    backend: str,
    model: str,
    max_candidates: int,
    temperature: float = 0.0,
) -> list[tuple[str, int]]:
    """Optionally refine auxiliary phrase candidates using only candidate labels.

    The LLM receives no protected target records and no raw auxiliary records;
    it receives only candidate phrases and auxiliary document-frequency support.
    The returned labels are intersected with the candidate set, so the refinement
    cannot introduce new decoder-visible labels.
    """
    if backend in {"none", "template", "local"}:
        return items
    limit = min(len(items), int(max_candidates))
    candidate_items = items[:limit]
    wanted = int(max_atoms) if max_atoms is not None else min(200, limit)
    wanted = min(wanted, limit)
    candidate_lines = "\n".join(
        f"- {phrase} (auxiliary_support={count})" for phrase, count in candidate_items
    )
    prompt = f"""
You are refining a public atom dictionary for aggregate review summarization.
You will receive only candidate atom labels extracted from an auxiliary/public
review split, with auxiliary document-frequency support counts. You must not
invent new labels. Select exactly {wanted} labels from the provided candidates.
Prefer labels that are interpretable review themes, non-duplicative, and not
purely generic. Return only JSON with this schema:
{{"labels": ["label 1", "label 2", ...]}}

Candidate labels:
{candidate_lines}
""".strip()
    response = maybe_llm(prompt, backend=backend, model=model, temperature=temperature, max_tokens=4000)
    if response is None:
        return items
    obj = _parse_llm_json_object(response)
    labels = obj.get("labels")
    if not isinstance(labels, list):
        raise ValueError("LLM refinement JSON must contain a list field 'labels'")
    support = {phrase: int(count) for phrase, count in items}
    selected: list[tuple[str, int]] = []
    seen = set()
    for label in labels:
        if not isinstance(label, str):
            continue
        label = label.strip()
        if label in support and label not in seen:
            selected.append((label, support[label]))
            seen.add(label)
    if not selected:
        raise ValueError("LLM refinement selected no labels present in candidate set")
    # If the model returned too few labels, fill deterministically from remaining candidates.
    for phrase, count in items:
        if len(selected) >= wanted:
            break
        if phrase not in seen:
            selected.append((phrase, count))
            seen.add(phrase)
    return selected[:wanted]


def atoms_from_metadata(
    records_path: str,
    output_path: str,
    columns: list[str],
    max_atoms: int | None = None,
    source_role: SourceRole = "auxiliary",
    include_support: bool = True,
) -> None:
    """Build public/auxiliary atom labels by concatenating metadata columns."""
    _check_source_role(source_role)

    print(PRIVACY_WARNING)
    print(f"Atom source role: {source_role}")
    print(f"Construction mode: metadata")
    print(f"Metadata columns: {columns}")

    records = read_jsonl(records_path)
    counts: Counter[tuple[tuple[str, str], ...]] = Counter()

    for r in records:
        meta = r.get("metadata", {}) or {}
        components: list[tuple[str, str]] = []
        for c in columns:
            v = meta.get(c)
            if _valid_value(v):
                components.append((str(c), str(v).strip()))
        if components:
            counts[tuple(components)] += 1

    atoms = []
    for i, (component_tuple, count) in enumerate(_sorted_count_items(counts, max_atoms)):
        metadata = {c: v for c, v in component_tuple}
        label = " / ".join(metadata[c] for c, _ in component_tuple)
        atom = {
            "atom_id": i,
            "label": label,
            "description": f"Records associated with {label}.",
            "metadata": metadata,
            "components": metadata,
            "atom_source_role": source_role,
            "atom_source_mode": "metadata",
            "atom_source_columns": list(columns),
        }
        if include_support:
            atom["support_in_atom_build_data"] = int(count)
        atoms.append(atom)

    _write_atoms_and_manifest(
        atoms,
        output_path,
        records_path,
        source_role,
        mode="metadata",
        max_atoms=max_atoms,
        extra_manifest={
            "metadata_columns": list(columns),
            "num_unique_candidate_atoms": len(counts),
        },
        include_support=include_support,
        n_records=len(records),
    )


def atoms_from_text_phrases(
    records_path: str,
    output_path: str,
    max_atoms: int | None = None,
    source_role: SourceRole = "auxiliary",
    include_support: bool = True,
    ngram_min: int = 2,
    ngram_max: int = 2,
    min_df: int = 2,
    max_features: int | None = None,
    refinement: Literal["none", "deduplicate", "llm"] = "none",
    refine_backend: str = "template",
    refine_model: str = "gpt-4o-mini",
    refine_top_candidates: int = 1000,
    review_phrase_filter: bool = True,
    require_theme_token: bool = True,
    max_df: float = 1.0,
    review_domain: ReviewDomain = "generic",
) -> None:
    """Build public/auxiliary atom labels from frequent text phrases.

    This is intended for review datasets such as Yelp and Amazon, whose raw rows
    contain a stable user id but do not provide a CFPB-style issue taxonomy. The
    input must be public or auxiliary data. The extracted phrase labels are
    decoder-visible public atoms; protected target text is not used here.
    """
    _check_source_role(source_role)
    if ngram_min < 1 or ngram_max < ngram_min:
        raise ValueError("Require 1 <= ngram_min <= ngram_max")
    if min_df < 1:
        raise ValueError("min_df must be at least 1")

    print(PRIVACY_WARNING)
    print(f"Atom source role: {source_role}")
    print("Construction mode: text_phrases")
    print(f"n-gram range: ({ngram_min}, {ngram_max}), min_df={min_df}")
    resolved_review_domain = _resolve_review_domain(review_domain, output_path, records_path)
    print(f"Refinement: {refinement}")
    print(f"Review domain: {resolved_review_domain}")

    records = read_jsonl(records_path)
    texts = [text_field(r) for r in records]
    texts = [t for t in texts if isinstance(t, str) and t.strip()]
    if not texts:
        raise ValueError("No text available for text_phrases atom construction")

    if review_phrase_filter:
        # Use a custom analyzer so n-grams are constructed within sentence-like
        # boundaries and filtered before ranking.  This avoids generic atoms and
        # artificial cross-boundary phrases.
        vectorizer = CountVectorizer(
            lowercase=False,
            analyzer=lambda t: _review_phrase_analyzer(t, ngram_min, ngram_max, require_theme_token),
            min_df=min_df,
            max_df=max_df,
            max_features=max_features,
        )
    else:
        vectorizer = CountVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(ngram_min, ngram_max),
            min_df=min_df,
            max_df=max_df,
            max_features=max_features,
            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z][a-zA-Z0-9_-]+\b",
        )
    X = vectorizer.fit_transform(texts)
    vocab = vectorizer.get_feature_names_out()
    # Use document frequency, not raw term frequency, to prefer themes that
    # occur across many auxiliary records rather than repeated boilerplate.
    df = (X > 0).sum(axis=0).A1
    raw_counts: Counter[str] = Counter({str(term): int(freq) for term, freq in zip(vocab, df)})
    num_unique_before_filtering = len(raw_counts)
    if review_phrase_filter:
        counts: Counter[str] = Counter({
            phrase: int(count)
            for phrase, count in raw_counts.items()
            if _is_valid_review_phrase(phrase, require_theme_token=require_theme_token)
        })
        ranked_items = _aggregate_domain_canonical_counts(list(counts.items()), resolved_review_domain)
    else:
        counts = raw_counts
        ranked_items = _sorted_count_items(counts, None)
    num_unique_before_refinement = len(ranked_items)
    if not ranked_items:
        raise ValueError(
            "No candidate review-theme atoms survived filtering. "
            "Try lowering --min-df, increasing the auxiliary split size, or passing "
            "--disable-review-phrase-filter for debugging only."
        )
    if refinement == "deduplicate":
        ranked_items = _deduplicate_phrase_candidates(ranked_items)
    elif refinement == "llm":
        # First apply deterministic deduplication so the LLM receives fewer
        # redundant auxiliary candidates. The LLM can only select from these
        # candidates and cannot introduce new labels.
        ranked_items = _deduplicate_phrase_candidates(ranked_items)
        ranked_items = _llm_refine_phrase_candidates(
            ranked_items,
            max_atoms=max_atoms,
            backend=refine_backend,
            model=refine_model,
            max_candidates=refine_top_candidates,
        )
    elif refinement != "none":
        raise ValueError(f"Unknown refinement mode: {refinement}")

    if refinement in {"deduplicate", "llm"} and review_phrase_filter:
        ranked_items = _aggregate_domain_canonical_counts(ranked_items, resolved_review_domain)

    final_items = ranked_items[:max_atoms] if max_atoms is not None else ranked_items

    atoms = []
    for i, (phrase, count) in enumerate(final_items):
        label = str(phrase).strip()
        atom = {
            "atom_id": i,
            "label": label,
            "description": f"Records whose review text is related to {label}.",
            "metadata": {"phrase": label},
            "components": {"phrase": label},
            "atom_source_role": source_role,
            "atom_source_mode": "text_phrases",
            "atom_source_columns": ["text"],
            "ngram_range": [int(ngram_min), int(ngram_max)],
            "min_df": int(min_df),
            "atom_refinement": refinement,
        }
        if include_support:
            atom["support_in_atom_build_data"] = int(count)
        atoms.append(atom)

    _write_atoms_and_manifest(
        atoms,
        output_path,
        records_path,
        source_role,
        mode="text_phrases",
        max_atoms=max_atoms,
        extra_manifest={
            "ngram_range": [int(ngram_min), int(ngram_max)],
            "min_df": int(min_df),
            "max_df": float(max_df),
            "review_phrase_filter": bool(review_phrase_filter),
            "review_domain": resolved_review_domain,
            "require_theme_token": bool(require_theme_token),
            "num_unique_raw_text_candidates": num_unique_before_filtering,
            "num_unique_candidate_atoms": num_unique_before_refinement,
            "num_candidate_atoms_after_refinement": len(ranked_items),
            "atom_refinement": refinement,
            "refine_backend": refine_backend if refinement == "llm" else None,
            "refine_model": refine_model if refinement == "llm" else None,
            "refine_top_candidates": int(refine_top_candidates) if refinement == "llm" else None,
            "refinement_privacy_note": (
                "Phrase refinement, when enabled, uses only auxiliary/public candidate labels and support counts. "
                "It is not run on protected target records."
            ),
        },
        include_support=include_support,
        n_records=len(records),
    )


def _write_atoms_and_manifest(
    atoms: list[dict],
    output_path: str,
    records_path: str,
    source_role: SourceRole,
    mode: str,
    max_atoms: int | None,
    extra_manifest: dict,
    include_support: bool,
    n_records: int,
) -> None:
    out = Path(output_path)
    ensure_dir(out.parent)
    write_json(atoms, out)
    manifest = {
        "records": str(records_path),
        "output": str(output_path),
        "source_role": source_role,
        "construction_mode": mode,
        "num_input_records": int(n_records),
        "num_written_atoms": len(atoms),
        "max_atoms": max_atoms,
        "include_support_in_atom_build_data": bool(include_support),
        "privacy_note": (
            "The written labels are decoder-visible. This manifest is safe only "
            "when the records source is public, benchmark-public, or an auxiliary "
            "split outside the protected target corpus."
        ),
        **extra_manifest,
    }
    write_json(manifest, out.with_suffix(out.suffix + ".manifest.json"))
    print(f"Wrote {len(atoms)} atoms to {output_path}")
    print(f"Wrote manifest to {out.with_suffix(out.suffix + '.manifest.json')}")


def main() -> None:
    """Build and save a semantic atom dictionary from the command line."""
    parser = argparse.ArgumentParser(description="Build a public/auxiliary atom universe.")
    parser.add_argument("--records", required=True, help="Input JSONL records from public/auxiliary/benchmark-public data.")
    parser.add_argument("--output", required=True, help="Output atoms JSON.")
    parser.add_argument("--mode", choices=["metadata", "text_phrases"], default="metadata")
    parser.add_argument("--metadata-cols", nargs="+", default=None, help="Metadata columns to concatenate into atom labels when --mode metadata.")
    parser.add_argument("--max-atoms", type=int, default=None)
    parser.add_argument(
        "--source-role",
        choices=["public", "auxiliary", "benchmark_public", "private_target"],
        default="auxiliary",
        help="Role of the records used for atom construction. private_target is rejected by design.",
    )
    parser.add_argument("--omit-build-support", action="store_true")
    parser.add_argument("--ngram-min", type=int, default=2, help="Minimum n-gram length for text_phrases. Default 2 avoids generic unigram atoms for review datasets.")
    parser.add_argument("--ngram-max", type=int, default=2, help="Maximum n-gram length for text_phrases. Default 2 favors concise review-theme labels.")
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--max-df", type=float, default=1.0, help="Maximum document-frequency fraction for text_phrases. Default 1.0 avoids dropping frequent but meaningful review themes; generic phrases are removed by the review-theme filter.")
    parser.add_argument("--max-features", type=int, default=None)
    parser.add_argument("--disable-review-phrase-filter", action="store_true", help="Debug option only: keep generic review phrases instead of applying the deterministic review-theme filter.")
    parser.add_argument("--review-domain", choices=["generic", "amazon", "yelp", "auto"], default="generic", help="Domain-specific deterministic phrase filtering/canonicalization for review atoms.")
    parser.add_argument("--allow-phrases-without-theme-token", action="store_true", help="Debug option only: do not require a concrete review-theme token in text_phrases candidates.")
    parser.add_argument(
        "--refinement",
        choices=["none", "deduplicate", "llm"],
        default="none",
        help="Optional auxiliary-only refinement for text_phrases atoms. none keeps raw ranked phrases; deduplicate merges near-identical auxiliary phrases; llm asks an LLM to select labels from auxiliary candidates only.",
    )
    parser.add_argument("--refine-backend", choices=["template", "openai", "none", "local"], default="template")
    parser.add_argument("--refine-model", default="gpt-4o-mini")
    parser.add_argument("--refine-top-candidates", type=int, default=1000)
    args = parser.parse_args()
    if args.mode == "metadata":
        if not args.metadata_cols:
            raise ValueError("--metadata-cols is required when --mode metadata")
        atoms_from_metadata(
            args.records,
            args.output,
            args.metadata_cols,
            args.max_atoms,
            source_role=args.source_role,
            include_support=not args.omit_build_support,
        )
    else:
        atoms_from_text_phrases(
            args.records,
            args.output,
            args.max_atoms,
            source_role=args.source_role,
            include_support=not args.omit_build_support,
            ngram_min=args.ngram_min,
            ngram_max=args.ngram_max,
            min_df=args.min_df,
            max_features=args.max_features,
            refinement=args.refinement,
            refine_backend=args.refine_backend,
            refine_model=args.refine_model,
            refine_top_candidates=args.refine_top_candidates,
            review_phrase_filter=not args.disable_review_phrase_filter,
            require_theme_token=not args.allow_phrases_without_theme_token,
            max_df=args.max_df,
            review_domain=args.review_domain,
        )


if __name__ == "__main__":
    main()
