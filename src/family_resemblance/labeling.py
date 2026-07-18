from __future__ import annotations

from collections import Counter

import numpy as np

from .text import Occurrence, STOPWORDS, normalize_token


def source_kind(source: str) -> str:
    if source.startswith("synthetic://"):
        return "synthetic"
    if source.startswith("pasted://"):
        return "pasted"
    return "corpus"


def evidence_label(corpus_count: int, pasted_count: int, synthetic_count: int) -> str:
    real_count = corpus_count + pasted_count
    if real_count > 0 and synthetic_count == 0:
        if corpus_count > 0 and pasted_count > 0:
            return "corpus + pasted evidence"
        if pasted_count > 0:
            return "pasted evidence"
        return "corpus evidence"
    if real_count > 0 and synthetic_count > 0:
        return "hybrid evidence"
    if synthetic_count > 0:
        return "synthetic only"
    return "no evidence"


def evidence_summary(points: list[dict]) -> dict[str, dict[str, int | str]]:
    by_word: dict[str, Counter[str]] = {}
    for point in points:
        word = str(point["word"])
        by_word.setdefault(word, Counter())[str(point.get("sourceKind", "corpus"))] += 1

    summary: dict[str, dict[str, int | str]] = {}
    for word, counts in by_word.items():
        corpus_count = counts.get("corpus", 0)
        pasted_count = counts.get("pasted", 0)
        synthetic_count = counts.get("synthetic", 0)
        summary[word] = {
            "label": evidence_label(corpus_count, pasted_count, synthetic_count),
            "corpus": corpus_count,
            "pasted": pasted_count,
            "synthetic": synthetic_count,
            "total": corpus_count + pasted_count + synthetic_count,
        }
    return summary


def summarize_clusters(
    word: str,
    occurrences: list[Occurrence],
    labels: np.ndarray,
    max_terms: int = 4,
    max_examples: int = 3,
) -> list[dict]:
    summaries: list[dict] = []
    word = normalize_token(word)

    for label in sorted(set(labels.tolist())):
        member_indices = [index for index, value in enumerate(labels.tolist()) if value == label]
        member_occurrences = [occurrences[index] for index in member_indices]
        terms = characteristic_terms(word, member_occurrences, max_terms=max_terms)
        domains = Counter(occurrence.domain for occurrence in member_occurrences)
        kinds = Counter(source_kind(occurrence.source) for occurrence in member_occurrences)
        examples = [occurrence.sentence for occurrence in member_occurrences[:max_examples]]

        if label < 0:
            name = "bridges / outliers"
        elif terms:
            name = " / ".join(terms[:3])
        else:
            name = f"cluster {label}"

        summaries.append(
            {
                "cluster": int(label),
                "name": name,
                "terms": terms,
                "count": len(member_occurrences),
                "domains": dict(domains.most_common()),
                "sources": dict(kinds.most_common()),
                "examples": examples,
            }
        )
    return summaries


def characteristic_terms(word: str, occurrences: list[Occurrence], max_terms: int = 4) -> list[str]:
    counts: Counter[str] = Counter()
    domain_words = {
        "auto_science", "auto_law", "auto_forum", "auto_fiction", "auto_politics",
        "auto_economy", "auto_medicine", "auto_technology", "pasted",
    }

    for occurrence in occurrences:
        for token in occurrence.context:
            token = normalize_token(token)
            if token == word or token in STOPWORDS or token in domain_words or len(token) <= 2:
                continue
            counts[token] += 1

    return [token for token, _ in counts.most_common(max_terms)]
