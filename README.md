# Speaker Identification (GMM-UBM)

Speaker identification system based on GMM-UBM with MAP adaptation.

## Project structure

```
.
├── pyproject.toml          # packaging, dependencies and test configuration
├── src/speaker_id/         # Python package (layered architecture)
│   ├── config.py           # typed paths and hyperparameters
│   ├── cli.py              # command-line interface
│   ├── gui.py              # graphical interface (Tkinter)
│   ├── reporting.py        # metrics and confusion matrix
│   ├── domain/             # value objects (speaker naming)
│   ├── audio/              # audio I/O and splitting
│   ├── features/           # MFCC + delta extraction
│   ├── ml/                 # GMM-UBM, MAP adaptation, classifier, persistence
│   └── application/        # use-cases (train / test / split)
├── tests/                  # unit tests (pytest)
├── data/
│   ├── gmm_dataset/        # enrollment files  '#_name_surname_note.wav'
│   ├── ubm_dataset/        # files for UBM training
│   ├── test/               # testing files
│   ├── temp/               # input for splitting
│   ├── splitted/           # split output
│   └── model/              # trained .pkl models (regenerable)
└── results/                # experiment plots
```

## Installation

```powershell
pip install -e .          # runtime
pip install -e .[dev]     # runtime + pytest
```

Audio files must be in `.wav` format (16 kHz). A `convert.sh` script is available
in the `data` folders to convert/resample the audio.

Training files must follow the naming convention `#_name_surname_note.wav`
(e.g. `AA_alberto_angela_0.wav` or `01_maurizio_crozza_renzi.wav`), where `#` is
an identifier and `note` is any free-form text useful to the user (for example
the impersonated character).

## Usage (CLI)

```powershell
# Train the UBM and MAP-adapt one GMM per speaker (data/ubm_dataset + data/gmm_dataset)
speaker-id train --n-gaussians 256 --relevance-factor 16

# Identification on the files in data/test (with confusion matrix)
speaker-id test --plot

# Split the files in data/temp into N-second segments (output in data/splitted)
speaker-id split 5
```

Global options: `--data-root <path>` (default `data`) and `-v/--verbose`.
Trained models are saved in `data/model` (one per speaker, keyed by label, plus
the UBM). The training defaults (`256` Gaussian components, relevance factor `16`)
follow the classic GMM-UBM literature and suit this dataset's small per-speaker
enrollment.

## Graphical interface (GUI)

For a visual workflow a desktop app (Tkinter) is available:

```powershell
speaker-id-gui          # or: python -m speaker_id.gui
```

The window offers three tabs — **Training**, **Identification**, **Audio split** —
with:

- a data-folder selector at the top;
- parameters (Gaussian components, relevance factor, seconds per segment);
- operations run in the background (the window stays responsive) with a
  **progress bar** and an embedded **live log**;
- a per-file results table (predicted vs. actual) with hit/miss highlighting,
  the **accuracy** value and a **confusion matrix** drawn directly in the window.

> Note: on Python 3.13 the audio split requires the `audioop-lts` backport
> (already included in the dependencies) because `pydub` relies on the `audioop`
> module removed from the standard library.

## Tests

```powershell
pytest
```

