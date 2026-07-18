from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .density import hdbscan_like
from .embedding import occurrence_vectors, train_word_embeddings
from .labeling import evidence_summary, source_kind, summarize_clusters
from .metrics import empty_metrics, target_metrics
from .reduction import spectral_umap_like
from .synthetic import generate_synthetic_documents
from .text import (
    Document,
    extract_occurrences,
    load_documents,
    normalize_token,
    sentence_token_stream,
    suggest_targets,
)


@dataclass(frozen=True)
class AtlasConfig:
    corpus: Path
    targets: tuple[str, ...] = ("игра", "справедливость", "узел")
    embedding_dim: int = 48
    max_vocab: int = 5000
    min_count: int = 1
    training_window: int = 5
    context_window: int = 8
    neighbors: int = 8
    min_cluster_size: int = 3
    min_samples: int = 2
    match_mode: str = "exact"
    allow_synthetic: bool = False
    synthetic_min_contexts: int = 12


class CorpusModel:
    def __init__(
        self,
        config: AtlasConfig,
        documents: list[Document],
        sentences: list[list[str]],
        vocab: dict[str, int],
        word_embeddings,
    ) -> None:
        self.config = config
        self.documents = documents
        self.sentences = sentences
        self.vocab = vocab
        self.word_embeddings = word_embeddings

    @classmethod
    def train(cls, config: AtlasConfig) -> "CorpusModel":
        documents = load_documents(config.corpus)
        sentences = sentence_token_stream(documents)
        vocab, word_embeddings = train_word_embeddings(
            sentences,
            dim=config.embedding_dim,
            max_vocab=config.max_vocab,
            min_count=config.min_count,
            window=config.training_window,
        )
        return cls(config, documents, sentences, vocab, word_embeddings)

    def analyze(
        self,
        targets: list[str] | tuple[str, ...],
        match_mode: str | None = None,
        min_cluster_size: int | None = None,
        min_samples: int | None = None,
        allow_synthetic: bool | None = None,
        extra_documents: list[Document] | None = None,
    ) -> dict:
        normalized_targets = tuple(normalize_token(target) for target in targets if target.strip())
        if not normalized_targets:
            raise ValueError("Введите хотя бы одно слово.")

        active_match_mode = match_mode or self.config.match_mode
        active_allow_synthetic = self.config.allow_synthetic if allow_synthetic is None else allow_synthetic
        extra_documents = extra_documents or []
        base_documents = self.documents + extra_documents
        extra_sentences = sentence_token_stream(extra_documents) if extra_documents else []
        base_sentences = self.sentences + extra_sentences

        occurrences = extract_occurrences(
            base_documents,
            list(normalized_targets),
            window=self.config.context_window,
            match_mode=active_match_mode,
        )
        real_counts = {
            target: sum(1 for occurrence in occurrences if occurrence.target == target)
            for target in normalized_targets
        }
        synthetic_targets = [
            target
            for target, count in real_counts.items()
            if active_allow_synthetic and count < self.config.synthetic_min_contexts
        ]
        synthetic_documents = generate_synthetic_documents(synthetic_targets) if synthetic_targets else []

        vocab = self.vocab
        word_embeddings = self.word_embeddings
        if extra_documents or synthetic_documents:
            analysis_documents = base_documents + synthetic_documents
            occurrences = extract_occurrences(
                analysis_documents,
                list(normalized_targets),
                window=self.config.context_window,
                match_mode=active_match_mode,
            )
            sentences = base_sentences + sentence_token_stream(synthetic_documents)
            vocab, word_embeddings = train_word_embeddings(
                sentences,
                dim=self.config.embedding_dim,
                max_vocab=self.config.max_vocab,
                min_count=self.config.min_count,
                window=self.config.training_window,
            )

        if not occurrences:
            first = normalized_targets[0]
            suggestions = suggest_targets(self.documents, first, limit=8)
            suggestion_text = ", ".join(str(item["word"]) for item in suggestions) or "нет похожих слов"
            raise ValueError(
                f"В выбранном корпусе нет контекстов для '{first}'. "
                f"Добавьте .txt тексты с этим словом или выберите слово из корпуса. "
                f"Подсказки: {suggestion_text}."
            )

        vectors = occurrence_vectors(
            occurrences,
            vocab,
            word_embeddings,
            dim=self.config.embedding_dim,
        )

        atlas_points: list[dict] = []
        metrics_by_word: dict[str, dict] = {}
        centroids: dict[str, list[float]] = {}
        clusters_by_word: dict[str, list[dict]] = {}
        missing: list[dict] = []

        for target in normalized_targets:
            indices = [index for index, occurrence in enumerate(occurrences) if occurrence.target == target]
            if not indices:
                metrics_by_word[target] = empty_metrics()
                centroids[target] = [0.0, 0.0]
                missing.append(
                    {
                        "word": target,
                        "suggestions": suggest_targets(self.documents, target, limit=8),
                    }
                )
                continue

            target_occurrences = [occurrences[index] for index in indices]
            target_vectors = vectors[indices]
            coords = spectral_umap_like(target_vectors, n_neighbors=self.config.neighbors)
            clusters = hdbscan_like(
                target_vectors,
                min_cluster_size=min_cluster_size or self.config.min_cluster_size,
                min_samples=min_samples or self.config.min_samples,
            )
            metrics = target_metrics(target_vectors, coords, clusters.labels)
            metrics_by_word[target] = metrics
            centroids[target] = coords.mean(axis=0).round(6).tolist() if len(coords) else [0.0, 0.0]
            cluster_summaries = summarize_clusters(target, target_occurrences, clusters.labels)
            clusters_by_word[target] = cluster_summaries
            cluster_names = {summary["cluster"]: summary["name"] for summary in cluster_summaries}

            medoid_local_index = int(metrics.get("medoid_index", -1))
            for local_index, occurrence in enumerate(target_occurrences):
                kind = source_kind(occurrence.source)
                atlas_points.append(
                    {
                        "id": occurrence.id,
                        "word": target,
                        "matchedForm": occurrence.form,
                        "domain": occurrence.domain,
                        "source": occurrence.source,
                        "sourceKind": kind,
                        "isSynthetic": kind == "synthetic",
                        "isPasted": kind == "pasted",
                        "sentence": occurrence.sentence,
                        "cluster": int(clusters.labels[local_index]),
                        "clusterName": cluster_names.get(int(clusters.labels[local_index]), "cluster"),
                        "x": float(round(coords[local_index, 0], 6)),
                        "y": float(round(coords[local_index, 1], 6)),
                        "isMedoid": local_index == medoid_local_index,
                    }
                )

        found_words = [target for target in normalized_targets if metrics_by_word[target]["occurrences"] > 0]
        if not found_words:
            raise ValueError("В корпусе нет употреблений для выбранных слов.")

        domains = sorted({document.domain for document in base_documents})
        synthetic_summary = [
            {
                "word": target,
                "contexts": sum(
                    1
                    for point in atlas_points
                    if point["word"] == target and point["isSynthetic"]
                ),
            }
            for target in normalized_targets
            if any(point["word"] == target and point["isSynthetic"] for point in atlas_points)
        ]
        pasted_summary = [
            {
                "word": target,
                "contexts": sum(
                    1
                    for point in atlas_points
                    if point["word"] == target and point.get("isPasted")
                ),
            }
            for target in normalized_targets
            if any(point["word"] == target and point.get("isPasted") for point in atlas_points)
        ]
        evidence = evidence_summary(atlas_points)
        source_mode = source_mode_label(evidence)
        return {
            "metadata": {
                "corpus": str(self.config.corpus),
                "domains": domains,
                "matchMode": active_match_mode,
                "missing": missing,
                "synthetic": synthetic_summary,
                "pasted": pasted_summary,
                "evidence": evidence,
                "sourceMode": source_mode,
                "method": {
                    "encoder": "local PPMI + SVD occurrence vectors with lexical-hash context features",
                    "projection": "local UMAP-like spectral embedding",
                    "clustering": "NumPy density clustering inspired by HDBSCAN mutual reachability",
                    "external_api": False,
                },
            },
            "words": found_words,
            "metrics": {word: metrics_by_word[word] for word in found_words},
            "clusters": {word: clusters_by_word[word] for word in found_words},
            "centroids": {word: centroids[word] for word in found_words},
            "points": atlas_points,
        }


def build_atlas(config: AtlasConfig) -> dict:
    model = CorpusModel.train(config)
    return model.analyze(
        config.targets,
        match_mode=config.match_mode,
        min_cluster_size=config.min_cluster_size,
        min_samples=config.min_samples,
        allow_synthetic=config.allow_synthetic,
    )


def source_mode_label(evidence: dict[str, dict[str, int | str]]) -> str:
    labels = {str(item["label"]) for item in evidence.values()}
    if not labels:
        return "no evidence"
    if labels == {"corpus evidence"}:
        return "corpus evidence"
    if labels == {"pasted evidence"}:
        return "pasted evidence"
    if labels <= {"corpus evidence", "pasted evidence", "corpus + pasted evidence"}:
        return "corpus + pasted evidence"
    if labels == {"synthetic only"}:
        return "synthetic only"
    if "hybrid evidence" in labels:
        return "hybrid evidence"
    return "mixed evidence"
