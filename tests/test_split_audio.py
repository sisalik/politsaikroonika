import re

import pytest

from auto_politseikroonika.language_consts import EST_ALPHABET_REGEX
from auto_politseikroonika.split_audio import (
    fix_words,
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
    words = re.split(fr" (?=[{EST_ALPHABET_REGEX}0-9])", text)
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


@pytest.mark.parametrize(
    "words, expected_words",
    [
        (["Vasikaid ", "oli ", "5."], ["Vasikaid", "oli", "5."]),
        (["Vasi", "kaid ol", "i 5. "], ["Vasikaid", "oli", "5."]),
    ],
)
def test_fix_words(words, expected_words):
    word_objs = [_make_word(word, i, i + 1) for i, word in enumerate(words)]
    fixed_words = list(fix_words(word_objs))
    for i, (word, expected_word) in enumerate(zip(fixed_words, expected_words)):
        assert word.text == expected_word
        assert word.start == i
        assert word.end == i + 1


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


# def test_make_sentences_with_multiple_words_in_a_word():
#     """Sometimes the transcript includes several words where there should be one."""
#     words = _make_words_list("Vasikaid oli 20.")
#     words[1].text = "oli 5. laudas"
#     sentences = list(make_sentences(words))
#     assert sentences[0].text == "Vasikaid oli 5. laudas 20."
