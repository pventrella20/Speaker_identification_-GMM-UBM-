"""Centralized, typed configuration.

The legacy system scattered magic path strings ('./data/model', 'data/model/ubm.pkl', ...)
across main.py and speaker_identification.py. Here every path is derived once, in one
place, from a single `root`, and every hyperparameter is validated at construction time
instead of failing deep inside training.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import math


@dataclass(frozen=True)
class DataPaths:
    """Resolves every data sub-folder from a single configurable root.

    Keeping the legacy folder names/roles (gmm_dataset, ubm_dataset, test, temp,
    splitted, model) so existing datasets can be reused without reorganizing files.
    """

    root: Path = field(default_factory=lambda: Path("data"))

    @property
    def gmm_dataset(self) -> Path:
        return self.root / "gmm_dataset"

    @property
    def ubm_dataset(self) -> Path:
        return self.root / "ubm_dataset"

    @property
    def test(self) -> Path:
        return self.root / "test"

    @property
    def temp(self) -> Path:
        return self.root / "temp"

    @property
    def splitted(self) -> Path:
        return self.root / "splitted"

    @property
    def models(self) -> Path:
        return self.root / "model"


@dataclass(frozen=True)
class TrainingConfig:
    """Training/adaptation hyperparameters, validated eagerly.

    Defaults follow the classic GMM-UBM speaker-recognition literature (Reynolds,
    Quatieri & Dunn, 2000) and are tuned for this dataset's small per-speaker
    enrollment:

    - `map_relevance_factor = 16`: with `alpha_i = n_i / (n_i + r)`, a Gaussian
      needs ~16 assigned frames before its adapted mean is 50% data / 50% UBM, so
      sparsely-populated components stay close to the robust UBM prior. (The
      legacy value 0.01 made `alpha_i ~= 1` for almost any frame count, i.e.
      effectively *no* regularization -- adapted means jumped entirely onto noisy
      enrollment statistics.)
    - `n_gaussians = 256`: a good accuracy/robustness/speed trade-off given the
      large UBM data but small per-speaker enrollment; only means are adapted, so
      a larger mixture mostly adds empty components and cost. Raise to 512 only
      with substantially more enrollment audio per speaker.

    Both are first-class, validated, tunable parameters (CLI flags / GUI fields).
    """

    n_gaussians: int = 256
    map_relevance_factor: float = 16.0
    n_mfcc: int = 20
    frame_size_sec: float = 0.025
    frame_step_sec: float = 0.01

    def __post_init__(self) -> None:
        if self.n_gaussians < 1 or not math.log2(self.n_gaussians).is_integer():
            raise ValueError(
                f"n_gaussians must be a power of 2 (e.g. 128, 256, 512), got {self.n_gaussians}"
            )
        if self.map_relevance_factor < 0:
            raise ValueError("map_relevance_factor must be >= 0")
