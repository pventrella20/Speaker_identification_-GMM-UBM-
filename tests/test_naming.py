import pytest
from speaker_id.domain.naming import parse_speaker_label, SpeakerFilenameError


@pytest.mark.parametrize(
    "stem,expected_label,expected_display",
    [
        ("01_maurizio_crozza_renzi", "maurizio_crozza", "crozza"),
        ("AA_alberto_angela_0", "alberto_angela", "angela"),
        ("07_maria_rossi_impersonation_of_someone", "maria_rossi", "rossi"),
    ],
)
def test_parses_valid_filenames(stem, expected_label, expected_display):
    result = parse_speaker_label(stem)
    assert result.label == expected_label
    assert result.display_name == expected_display


@pytest.mark.parametrize("stem", ["", "onlyoneword", "two_words", "convert"])
def test_rejects_malformed_filenames(stem):
    with pytest.raises(SpeakerFilenameError):
        parse_speaker_label(stem)
