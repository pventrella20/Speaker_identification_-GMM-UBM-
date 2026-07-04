"""Command-line interface.

The legacy `main.py` drove the whole system through a `while True: input(...)`
loop mixing prompts, validation, and business logic in `if __name__ == '__main__'`.
That makes the system impossible to script, impossible to unit test, and forces a
human to type 'train'/'test'/'yes'/'no' every single run.

This CLI uses explicit subcommands and flags (a standard, scriptable, 12-factor
style interface: `speaker-id train --n-gaussians 256`), so the same commands work
identically from a terminal, a cron job, or a CI pipeline. Every subcommand maps
directly onto one use-case from `application.use_cases`.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from sklearn.metrics import accuracy_score

from .config import DataPaths, TrainingConfig
from .features.mfcc import MFCCFeatureExtractor
from .ml.persistence import ModelRepository
from .application.use_cases import TrainModelsUseCase, IdentifySpeakersUseCase, SplitAudioUseCase
from .reporting import build_confusion_matrix, display_confusion_matrix

logger = logging.getLogger("speaker_id")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="speaker-id", description="GMM-UBM speaker identification system")
    parser.add_argument("--data-root", type=Path, default=Path("data"), help="Root of the data/ folder tree")
    parser.add_argument("-v", "--verbose", action="store_true")

    sub = parser.add_subparsers(dest="command", required=True)

    train_p = sub.add_parser("train", help="Train the UBM and MAP-adapt one GMM per enrolled speaker")
    train_p.add_argument("--n-gaussians", type=int, default=256, help="Number of Gaussian components (power of 2)")
    train_p.add_argument("--relevance-factor", type=float, default=16.0, help="MAP adaptation relevance factor")

    test_p = sub.add_parser("test", help="Run identification on data/test and report accuracy")
    test_p.add_argument("--plot", action="store_true", help="Display the confusion matrix")

    split_p = sub.add_parser("split", help="Split every .wav file in data/temp into fixed-length segments")
    split_p.add_argument("seconds", type=int, help="Segment length in seconds")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s")

    paths = DataPaths(root=args.data_root)

    try:
        if args.command == "split":
            SplitAudioUseCase(paths).execute(args.seconds)
            return 0

        extractor = MFCCFeatureExtractor()
        model_repository = ModelRepository(paths.models)

        if args.command == "train":
            config = TrainingConfig(n_gaussians=args.n_gaussians, map_relevance_factor=args.relevance_factor)
            TrainModelsUseCase(paths, config, extractor, model_repository).execute()
            return 0

        if args.command == "test":
            results = IdentifySpeakersUseCase(paths, extractor, model_repository).execute()
            y_true = [r.true_label or "unknown" for r in results]
            y_pred = [r.predicted_label for r in results]
            accuracy = accuracy_score(y_true, y_pred)
            logger.info("Accuracy: %.3f", accuracy)

            if args.plot:
                labels = sorted(set(y_true) | set(y_pred))
                cm = build_confusion_matrix(y_true, y_pred, labels)
                display_confusion_matrix(cm, labels, accuracy)
            return 0

    except FileNotFoundError as exc:
        logger.error(str(exc))
        return 1

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
