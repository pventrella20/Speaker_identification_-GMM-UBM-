"""Persists and retrieves trained models -- a Repository around joblib/pickle.

Legacy bug this fixes: `speaker_identification.SpeakerRecognition.fit_model` saved
each model as `'data/model/gmm' + str(i + 1) + "_" + self.speakers_names[i] + '.pkl'`,
and `load_model` reconstructed the *same* index-based filename from a *freshly
recomputed* `self.speakers_names` list in a brand-new instance (train() and test()
each build their own `SpeakerRecognition`). Both `speakers_names` lists come from
`dict.fromkeys` over a directory listing, so they only line up correctly if the
GMM dataset directory has not changed between training and testing (add/remove one
speaker's files and every other speaker's model becomes unreadable, or silently
loads the wrong speaker, because the index shifts). Here every model is keyed
purely by its speaker label, so storage no longer depends on directory-listing
order matching between two unrelated runs.
"""
from __future__ import annotations

from pathlib import Path
import joblib
from sklearn.mixture import GaussianMixture


class ModelRepository:
    UBM_FILENAME = "ubm.pkl"

    def __init__(self, models_dir: Path):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def save_ubm(self, ubm: GaussianMixture) -> Path:
        path = self.models_dir / self.UBM_FILENAME
        joblib.dump(ubm, path)
        return path

    def load_ubm(self) -> GaussianMixture:
        path = self.models_dir / self.UBM_FILENAME
        if not path.exists():
            raise FileNotFoundError(f"UBM model not found at {path}. Run 'train' first.")
        return joblib.load(path)

    def save_speaker_model(self, label: str, gmm: GaussianMixture) -> Path:
        path = self._speaker_path(label)
        joblib.dump(gmm, path)
        return path

    def load_speaker_model(self, label: str) -> GaussianMixture:
        path = self._speaker_path(label)
        if not path.exists():
            raise FileNotFoundError(f"Speaker model not found at {path}.")
        return joblib.load(path)

    def list_speaker_labels(self) -> list[str]:
        prefix_len = len("gmm_")
        return sorted(p.stem[prefix_len:] for p in self.models_dir.glob("gmm_*.pkl"))

    def _speaker_path(self, label: str) -> Path:
        return self.models_dir / f"gmm_{label}.pkl"
