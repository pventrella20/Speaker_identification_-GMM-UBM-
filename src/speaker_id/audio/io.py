"""Filesystem access to raw audio, isolated behind a small repository interface.

Legacy `files_manager.read_files` / `file_processing` filtered filenames with
`re.search('[\\w].ogg', filename)` -- a regex that matches "any single word
character followed by the literal text '.ogg'" and is *not* an extension check
(e.g. it also matches 'foo.oggexample', and misses 'x.ogg' with no preceding
word char in some edge cases). It also duplicated the same broken pattern four
times across the codebase. `Path.suffix` replaces all of that with one obviously
correct, testable line.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
from scipy.io import wavfile


@dataclass(frozen=True, slots=True)
class AudioClip:
    sample_rate: int
    samples: np.ndarray


class AudioRepository:
    """Reads .wav clips from a single directory."""

    def __init__(self, directory: Path):
        self.directory = Path(directory)

    def list_stems(self) -> list[str]:
        """Filenames (without extension) of every .wav file in the directory, sorted."""
        if not self.directory.exists():
            return []
        return sorted(
            p.stem for p in self.directory.iterdir() if p.is_file() and p.suffix.lower() == ".wav"
        )

    def load(self, stem: str) -> AudioClip:
        path = self.directory / f"{stem}.wav"
        sample_rate, samples = wavfile.read(path)
        return AudioClip(sample_rate=sample_rate, samples=samples)
