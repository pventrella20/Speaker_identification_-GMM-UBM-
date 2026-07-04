from pathlib import Path
import pytest
from speaker_id.config import DataPaths, TrainingConfig


def test_data_paths_derive_from_root():
    paths = DataPaths(root=Path("/tmp/my_data"))
    assert paths.gmm_dataset == Path("/tmp/my_data/gmm_dataset")
    assert paths.ubm_dataset == Path("/tmp/my_data/ubm_dataset")
    assert paths.test == Path("/tmp/my_data/test")
    assert paths.models == Path("/tmp/my_data/model")


@pytest.mark.parametrize("n", [128, 256, 512, 1024])
def test_valid_power_of_two_gaussians(n):
    TrainingConfig(n_gaussians=n)  # must not raise


@pytest.mark.parametrize("n", [0, 3, 100, 511])
def test_rejects_non_power_of_two_gaussians(n):
    with pytest.raises(ValueError):
        TrainingConfig(n_gaussians=n)
