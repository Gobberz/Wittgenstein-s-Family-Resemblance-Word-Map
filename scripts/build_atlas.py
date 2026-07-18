from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from family_resemblance.export import write_outputs
from family_resemblance.pipeline import AtlasConfig, build_atlas


def config_from_args(args: argparse.Namespace) -> AtlasConfig:
    return AtlasConfig(
        corpus=args.corpus,
        targets=tuple(args.targets),
        embedding_dim=args.embedding_dim,
        max_vocab=args.max_vocab,
        min_count=args.min_count,
        training_window=args.training_window,
        context_window=args.context_window,
        neighbors=args.neighbors,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        match_mode=args.match_mode,
        allow_synthetic=args.auto_contexts,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a local Family Resemblance atlas.")
    parser.add_argument("--corpus", type=Path, default=ROOT / "data" / "sample_corpus")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs")
    parser.add_argument("--targets", nargs="+", default=["игра", "справедливость", "узел"])
    parser.add_argument("--embedding-dim", type=int, default=48)
    parser.add_argument("--max-vocab", type=int, default=5000)
    parser.add_argument("--min-count", type=int, default=1)
    parser.add_argument("--training-window", type=int, default=5)
    parser.add_argument("--context-window", type=int, default=8)
    parser.add_argument("--neighbors", type=int, default=8)
    parser.add_argument("--min-cluster-size", type=int, default=3)
    parser.add_argument("--min-samples", type=int, default=2)
    parser.add_argument("--match-mode", choices=["exact", "prefix", "lemma-lite"], default="exact")
    parser.add_argument("--auto-contexts", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    atlas = build_atlas(config_from_args(args))
    html_path, json_path, report_path = write_outputs(atlas, args.out)
    print(f"Wrote {html_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
