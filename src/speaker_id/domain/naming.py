"""Parsing of the speaker enrollment filename convention.

Legacy convention (unchanged): '#_name_surname_note.wav', e.g. '01_maurizio_crozza_renzi.wav'.

The original `files_manager.file_processing` derived speaker identity with
`filename.split("_")` and *positional* indexing (curr_elem[1], curr_elem[2]) with
no validation: a malformed filename raised a bare, uninformative IndexError deep
inside a loop. Here parsing is centralized, validated, and raises a domain-specific,
actionable error -- and single-file parsing is unit-testable in isolation.
"""
from __future__ import annotations

from dataclasses import dataclass
import re

_PATTERN = re.compile(r"^(?P<index>[^_]+)_(?P<first_name>[^_]+)_(?P<last_name>[^_]+)_(?P<note>.+)$")


class SpeakerFilenameError(ValueError):
    """Raised when a filename doesn't follow the '#_name_surname_note' convention."""


@dataclass(frozen=True, slots=True)
class SpeakerLabel:
    label: str          # storage key, e.g. "maurizio_crozza" -- stable, used to name model files
    display_name: str   # e.g. "crozza" -- legacy convention used for confusion-matrix axis labels


def parse_speaker_label(stem: str) -> SpeakerLabel:
    """Extract a `SpeakerLabel` from an audio file stem (filename without extension)."""
    match = _PATTERN.match(stem)
    if not match:
        raise SpeakerFilenameError(
            f"'{stem}' does not follow the '#_name_surname_note' naming convention "
            "(e.g. '01_maurizio_crozza_renzi')"
        )
    first_name = match.group("first_name")
    last_name = match.group("last_name")
    return SpeakerLabel(label=f"{first_name}_{last_name}", display_name=last_name)
