"""Splits long audio recordings into fixed-length .wav segments.

Behavioural equivalent of legacy `pre_processing.SplitWavAudio`, but:
- last segment is clamped to the real audio length instead of `to_sec` possibly
  running past the end of the clip (pydub silently clamps, but the intent is now
  explicit rather than accidental);
- returns the list of produced paths instead of only printing progress, so
  callers (and tests) can assert on results instead of scraping stdout.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import logging

logger = logging.getLogger(__name__)


@dataclass
class AudioSplitter:
    source_dir: Path
    target_dir: Path

    def split(self, stem: str, seconds_per_segment: int) -> list[Path]:
        if seconds_per_segment <= 0:
            raise ValueError("seconds_per_segment must be > 0")

        # Imported lazily so that `train`/`test` (and the GUI) keep working even
        # when pydub's audio backend is unavailable -- e.g. on Python 3.13, where
        # the stdlib `audioop` module pydub relies on was removed (install the
        # `audioop-lts` backport, declared in pyproject, to enable splitting).
        from pydub import AudioSegment

        audio = AudioSegment.from_wav(self.source_dir / f"{stem}.wav")
        total_segments = math.ceil(audio.duration_seconds / seconds_per_segment)
        self.target_dir.mkdir(parents=True, exist_ok=True)

        output_paths: list[Path] = []
        for i in range(total_segments):
            start_ms = i * seconds_per_segment * 1000
            end_ms = min((i + 1) * seconds_per_segment * 1000, len(audio))
            segment = audio[start_ms:end_ms]
            out_path = self.target_dir / f"{stem}_{i}.wav"
            segment.export(out_path, format="wav")
            output_paths.append(out_path)
            logger.info("%s -> %s (%d/%d)", stem, out_path.name, i + 1, total_segments)
        return output_paths

    def split_all(self, stems: list[str], seconds_per_segment: int) -> dict[str, list[Path]]:
        return {stem: self.split(stem, seconds_per_segment) for stem in stems}
