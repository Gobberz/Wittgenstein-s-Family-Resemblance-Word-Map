from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


CYRILLIC_LETTERS = r"\u0401\u0451\u0410-\u044f"
TOKEN_RE = re.compile(rf"[A-Za-z{CYRILLIC_LETTERS}]+(?:[-'][A-Za-z{CYRILLIC_LETTERS}]+)?|\d+(?:[.,]\d+)?")
SENTENCE_RE = re.compile(r"(?<=[.!?;:])\s+|\n+")

CYRILLIC_STOPWORDS = {
    "\u0430", "\u0431\u0435\u0437", "\u0431\u044b", "\u0432", "\u0432\u043e",
    "\u0434\u043b\u044f", "\u0434\u043e", "\u0435\u0433\u043e", "\u0435\u0435",
    "\u0435\u0441\u043b\u0438", "\u0436\u0435", "\u0437\u0430", "\u0438",
    "\u0438\u0437", "\u0438\u043b\u0438", "\u043a\u0430\u043a", "\u043a",
    "\u043a\u043e", "\u043b\u0438", "\u043d\u0430", "\u043d\u0435",
    "\u043d\u043e", "\u043e", "\u043e\u0431", "\u043e\u0442", "\u043f\u043e",
    "\u043f\u0440\u0438", "\u0441", "\u0441\u043e", "\u0442\u0430\u043a",
    "\u0442\u043e", "\u0443", "\u0447\u0442\u043e", "\u044d\u0442\u043e",
    "\u044d\u0442\u043e\u0442",
}

STOPWORDS = {
    *CYRILLIC_STOPWORDS,
    "the", "a", "an", "and", "or", "of", "in", "to", "for", "with", "is",
}


@dataclass(frozen=True)
class Document:
    domain: str
    source: str
    text: str


@dataclass(frozen=True)
class Occurrence:
    id: str
    target: str
    domain: str
    source: str
    sentence: str
    form: str
    tokens: tuple[str, ...]
    index: int
    left: tuple[str, ...]
    right: tuple[str, ...]

    @property
    def context(self) -> tuple[str, ...]:
        return self.left + self.right


def normalize_token(token: str) -> str:
    return token.lower().replace("\u0451", "\u0435")


def tokenize(text: str) -> list[str]:
    return [normalize_token(match.group(0)) for match in TOKEN_RE.finditer(text)]


def split_sentences(text: str) -> list[str]:
    return [part.strip() for part in SENTENCE_RE.split(text) if part.strip()]


def load_documents(corpus_dir: Path) -> list[Document]:
    files = sorted(corpus_dir.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No .txt files found in {corpus_dir}")

    documents: list[Document] = []
    for path in files:
        documents.append(
            Document(
                domain=path.stem,
                source=str(path),
                text=path.read_text(encoding="utf-8"),
            )
        )
    return documents


def sentence_token_stream(documents: list[Document]) -> list[list[str]]:
    sentences: list[list[str]] = []
    for doc in documents:
        for sentence in split_sentences(doc.text):
            tokens = tokenize(sentence)
            if tokens:
                sentences.append(tokens)
    return sentences


def extract_occurrences(
    documents: list[Document],
    targets: list[str],
    window: int = 8,
    match_mode: str = "exact",
) -> list[Occurrence]:
    matcher = TargetMatcher(targets, match_mode=match_mode)
    occurrences: list[Occurrence] = []

    for doc in documents:
        for sentence_no, sentence in enumerate(split_sentences(doc.text), start=1):
            tokens = tokenize(sentence)
            for index, token in enumerate(tokens):
                target = matcher.match(token)
                if target is None:
                    continue

                left = tuple(tokens[max(0, index - window):index])
                right = tuple(tokens[index + 1:index + 1 + window])
                occurrence_id = f"{doc.domain}:{sentence_no}:{index}:{len(occurrences)}"
                occurrences.append(
                    Occurrence(
                        id=occurrence_id,
                        target=target,
                        domain=doc.domain,
                        source=doc.source,
                        sentence=sentence,
                        form=token,
                        tokens=tuple(tokens),
                        index=index,
                        left=left,
                        right=right,
                    )
                )

    return occurrences


class TargetMatcher:
    def __init__(self, targets: list[str], match_mode: str = "exact") -> None:
        self.targets = [normalize_token(target) for target in targets if target.strip()]
        self.match_mode = match_mode
        self.exact = {target: target for target in self.targets}
        self.stems: dict[str, str] = {}
        for target in self.targets:
            self.stems.setdefault(rough_stem(target), target)

    def match(self, token: str) -> str | None:
        token = normalize_token(token)
        if token in self.exact:
            return self.exact[token]
        if self.match_mode == "exact":
            return None
        if self.match_mode == "prefix":
            return self._prefix_match(token)
        if self.match_mode == "lemma-lite":
            return self.stems.get(rough_stem(token)) or self._prefix_match(token)
        raise ValueError(f"Unsupported match mode: {self.match_mode}")

    def _prefix_match(self, token: str) -> str | None:
        for target in self.targets:
            if len(target) >= 3 and token.startswith(target):
                return target
            if len(token) >= 3 and target.startswith(token):
                return target
        return None


def rough_stem(token: str) -> str:
    token = normalize_token(token)
    if len(token) <= 4:
        return token

    cyrillic_endings = (
        "\u0438\u044f\u043c\u0438", "\u044f\u043c\u0438", "\u0430\u043c\u0438",
        "\u043e\u0433\u043e", "\u0435\u043c\u0443", "\u0438\u043c\u0438",
        "\u044b\u043c\u0438", "\u0438\u0435\u0439", "\u043e\u0441\u0442\u044c",
        "\u043e\u0441\u0442\u044c\u044e", "\u0438\u044f\u0445", "\u0438\u044e",
        "\u044c\u044e", "\u0438\u044f", "\u0438\u0435", "\u044b\u0445",
        "\u0438\u0445", "\u043e\u0439", "\u0438\u0439", "\u044b\u0439",
        "\u0430\u044f", "\u043e\u0435", "\u044b\u0435", "\u0443\u044e",
        "\u044e\u044e", "\u043e\u043c", "\u0435\u043c", "\u0430\u0445",
        "\u044f\u0445", "\u0430\u043c", "\u044f\u043c", "\u043e\u0432",
        "\u0435\u0432", "\u0430", "\u044f", "\u044b", "\u0438", "\u0443",
        "\u044e", "\u0435", "\u043e", "\u044c",
    )
    english_endings = ("ingly", "edly", "ing", "ed", "ness", "tion", "s")

    for ending in cyrillic_endings + english_endings:
        if token.endswith(ending) and len(token) - len(ending) >= 3:
            return token[: -len(ending)]
    return token


def token_counts(documents: list[Document]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for document in documents:
        for token in tokenize(document.text):
            counts[token] = counts.get(token, 0) + 1
    return counts


def suggest_targets(documents: list[Document], query: str, limit: int = 10) -> list[dict[str, int | str]]:
    query = normalize_token(query)
    stem = rough_stem(query)
    counts = token_counts(documents)
    candidates: list[tuple[int, int, str]] = []

    for token, count in counts.items():
        if token in STOPWORDS or len(token) <= 2:
            continue
        token_stem = rough_stem(token)
        if token == query:
            score = 0
        elif token.startswith(query) or query.startswith(token):
            score = 1
        elif token_stem == stem:
            score = 2
        elif query in token or token in query:
            score = 3
        else:
            continue
        candidates.append((score, -count, token))

    candidates.sort()
    if not candidates:
        candidates = [
            (9, -count, token)
            for token, count in counts.items()
            if token not in STOPWORDS and len(token) > 2
        ]
        candidates.sort()
    return [{"word": token, "count": -negative_count} for score, negative_count, token in candidates[:limit]]
