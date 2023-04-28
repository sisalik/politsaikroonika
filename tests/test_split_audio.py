import re

import pytest

from politsaikroonika.language_consts import EST_ALPHABET_REGEX
from politsaikroonika.split_audio import (
    fix_word_timestamps,
    fix_split_or_merged_words,
    make_sentences,
    words_are_equal_except_for_punctuation,
    Word,
)


def _make_word(text, start=0, end=1):
    return Word(text, start, end)


def _make_words_list(text, start=0, end=None):
    # Split text into words using a regex to define the separators. This is to match the
    # behaviour of the tekstiks.ee ASR pipeline which groups punctuation with the
    # preceding word.
    words = re.split(rf" (?=[{EST_ALPHABET_REGEX}0-9])", text)
    if end is None:
        end = start + len(words)
    duration = end - start
    word_duration = duration / len(words)
    word_gap = 0.1  # Interval between words

    word_objs = []
    for i, word in enumerate(words):
        word_start = start + i * word_duration
        word_end = word_start + word_duration - word_gap
        word_objs.append(_make_word(word, word_start, word_end))
    return word_objs


def test_fix_word_timestamps_middle():
    """A word in the middle of the sentence is missing timestamps."""
    words = [
        _make_word("Sõna 1", 0, 1),
        _make_word("sõna 2", None, None),
        _make_word("sõna 3", 2, 3),
    ]
    fixed_words = list(fix_word_timestamps(words))
    assert (fixed_words[0].start, fixed_words[0].end) == (0, 1)
    assert (fixed_words[1].start, fixed_words[1].end) == (1, 2)
    assert (fixed_words[2].start, fixed_words[2].end) == (2, 3)


def test_fix_word_timestamps_initial():
    """Initial words in the sentence are missing timestamps."""
    words = [
        _make_word("Sõna 1", None, None),
        _make_word("sõna 2", None, None),
        _make_word("sõna       3", 2, 3),  # Double the length of the other words
    ]
    fixed_words = list(fix_word_timestamps(words))
    # The first two words should be half the duration of the third word since they have
    # half the number of characters
    assert (fixed_words[0].start, fixed_words[0].end) == (1.0, 1.5)
    assert (fixed_words[1].start, fixed_words[1].end) == (1.5, 2)
    assert (fixed_words[2].start, fixed_words[2].end) == (2, 3)


def test_fix_word_timestamps_final():
    """Final words in the sentence are missing timestamps."""
    words = [
        _make_word("Sõna 1", 0, 1),
        _make_word("sõna 2", 1, 2),
        _make_word("sõna 3", None, None),
        _make_word("sõna 4", None, None),
    ]
    fixed_words = list(fix_word_timestamps(words))
    assert (fixed_words[0].start, fixed_words[0].end) == (0, 1)
    assert (fixed_words[1].start, fixed_words[1].end) == (1, 2)
    assert (fixed_words[2].start, fixed_words[2].end) == (2, 3)
    assert (fixed_words[3].start, fixed_words[3].end) == (3, 4)


@pytest.mark.parametrize(
    "words, expected_words, expected_durations",
    [
        (["Vasikaid ", "oli ", "5."], ["Vasikaid", "oli", "5."], None),
        (["Vasi", "kaid ol", "i 5. "], ["Vasikaid", "oli", "5."], None),
        (
            ["Kui ", "jood,", " ", "siis ä", "ra sõida."],
            ["Kui", "jood,", "siis", "ära", "sõida."],
            # Word lengths get split up unevenly. "jood," and " " are 1 second each,
            # so "jood, " ends up being 2 seconds. The remaining 2 seconds are split
            # between 3 words, each of which is 0.66 seconds long.
            [1, 2, 2 / 3, 2 / 3, 2 / 3],
        ),
    ],
)
def test_fix_words(words, expected_words, expected_durations):
    word_objs = [_make_word(word, i, i + 1) for i, word in enumerate(words)]
    fixed_words = list(fix_split_or_merged_words(word_objs))
    for i, (word, expected_word) in enumerate(zip(fixed_words, expected_words)):
        assert word.text == expected_word
        if expected_durations is None:
            assert word.start == i
            assert word.end == i + 1
        else:
            assert word.start == sum(expected_durations[:i])
            assert word.end == word.start + expected_durations[i]


@pytest.mark.parametrize(
    "word1, word2, should_be_equal",
    [
        ("a", "a", True),
        ("a", "a.", True),
        ("a", "a,", True),
        ("a", "a!", True),
        ("a", "a?", True),
        ("a", "a:", True),
        ("a", "a;", True),
        ("a", "a - ", True),
        ("a", "   .a.   ", True),
        ("a", "b", False),
        ("a-a", "aa", False),
    ],
)
def test_words_are_equal_except_for_punctuation(word1, word2, should_be_equal):
    assert words_are_equal_except_for_punctuation(word1, word2) == should_be_equal


@pytest.mark.parametrize(
    "words, expected_sentences",
    [
        ("Vasikaid oli 5.", ["Vasikaid oli 5."]),
        ("Vasikaid oli 5. Kanu oli 20.", ["Vasikaid oli 5.", "Kanu oli 20."]),
        ("Vasikaid oli 5. laudas 20.", ["Vasikaid oli 5. laudas 20."]),
        (
            "1675. aastal sündinud 12. klassi õpilane - see on erakordne.",
            ["1675. aastal sündinud 12. klassi õpilane - see on erakordne."],
        ),
    ],
)
def test_make_sentences(words, expected_sentences):
    words = _make_words_list(words)
    sentences = list(make_sentences(words))
    for sentence, expected_sentence in zip(sentences, expected_sentences):
        assert sentence.text == expected_sentence
