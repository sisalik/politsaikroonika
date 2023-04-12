import math
import pytest

from auto_politseikroonika.make_episode import _seconds_to_frame_splits


@pytest.mark.parametrize(
    "seconds, expected_frames",
    [
        (0.1, [16]),
        (2.0, [16]),
        (2.5, [20]),
        (6.0, [48]),
    ],
)
def test_seconds_to_frames_single(seconds, expected_frames):
    assert _seconds_to_frame_splits(seconds) == expected_frames


@pytest.mark.parametrize(
    "seconds, expected_splits",
    [
        (6.1, 2),
        (12.0, 2),
        (12.1, 3),
    ],
)
def test_seconds_to_frames_multiple(seconds, expected_splits):
    # Repeat the test multiple times since the split points are random
    for _ in range(1000):
        splits = _seconds_to_frame_splits(seconds)
        assert len(splits) == expected_splits
        assert sum(splits) == math.ceil(seconds * 8)
        for split in splits:
            assert 16 <= split <= 48
