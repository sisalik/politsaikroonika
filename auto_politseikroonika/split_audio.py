import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass

from estnltk import Text
from loguru import logger

from auto_politseikroonika.morph_categories import WordClass

# The speaker IDs that are used in the transcript JSON file to tag the speech we are
# interested in
SPEAKER_IDS = ["VO", "Mic"]
# Maximum desired sentence duration in seconds
MAX_DURATION = 10
# Minimum desired sentence duration in seconds
MIN_DURATION = 1


@dataclass
class Word:
    text: str
    start: float
    end: float

    def __str__(self):
        return self.text


class Sentence:
    def __init__(self, words):
        self.words = words

    @property
    def text(self):
        return " ".join([word.text for word in self.words])

    @property
    def start(self):
        return self.words[0].start

    @property
    def end(self):
        return self.words[-1].end


def extract_words(transcript):
    """Extract all words from the transcript JSON file."""
    for block in transcript["content"]:
        if block["type"] != "speaker":
            continue
        if block["attrs"]["data-name"] not in SPEAKER_IDS:
            continue

        block_words = []
        for element in block["content"]:
            if element["type"] != "text":
                continue

            block_words.append(
                Word(
                    text=element["text"].strip(),
                    start=float(element["marks"][0]["attrs"]["start"]),
                    end=float(element["marks"][0]["attrs"]["end"]),
                )
            )
        # In case the last string doesn't end with a period, but the block ends,
        # add the missing period
        if block_words[-1].text[-1] != ".":
            block_words[-1].text += "."
        yield from block_words


def make_sentences(words):
    """Draw sentence boundaries, addressing ordinal numbers and abbreviations etc."""
    all_words = list(words)  # We need to iterate over the words twice
    paragraph = Text(" ".join(word.text for word in all_words))
    paragraph.analyse("morphology")
    word_obj_iterator = iter(all_words)
    for sentence in paragraph.sentences:
        sentence_words = []
        for word in sentence.words:
            # Ignore punctuation
            if word.morph_analysis.partofspeech[0] == WordClass.PUNCTUATION:
                continue

            word_obj = next(word_obj_iterator)
            assert words_are_equal_except_for_punctuation(
                word_obj.text, word.text
            ), f"Word mismatch: {word_obj.text} != {word.text}"  # Sanity check - should never happen
            sentence_words.append(word_obj)
        yield Sentence(sentence_words)


def words_are_equal_except_for_punctuation(word1, word2):
    """Check if two words are equal, except for surrounding punctuation."""
    punctuation = " .,-+!?;:()[]{}\"'"
    return word1.strip(punctuation) == word2.strip(punctuation)


def conform_sentences(sentences):
    """Ensure that sentences are not too long and split them if necessary."""
    for sentence in sentences:
        yield from split_sentence_if_needed(sentence)


def split_sentence_if_needed(sentence):
    duration = sentence.end - sentence.start
    if duration <= MAX_DURATION:
        yield sentence
        return

    logger.debug(
        f"Sentence too long ({duration:.2f} s > {MAX_DURATION} s)\n  '{sentence.text}'"
    )
    # Try to split the sentence at a conjunction
    split_idx = find_split_idx_conjunction(sentence)
    if split_idx is None or not split_sentence_lengths_are_ok(sentence, split_idx):
        # No conjunction found, or the split would result in a sentence too short
        for split_idx in find_split_intervals(sentence):
            if split_sentence_lengths_are_ok(sentence, split_idx):
                break
        else:
            # No split interval found
            raise Exception("No suitable split interval found")

    part_1 = Sentence(sentence.words[: split_idx + 1])
    part_2 = Sentence(sentence.words[split_idx + 1 :])
    logger.debug(f"Split into:\n  '{part_1.text}'\n  '{part_2.text}'")
    yield from split_sentence_if_needed(part_1)
    yield from split_sentence_if_needed(part_2)


def find_split_idx_conjunction(sentence):
    """Find the index of the last word before a conjunction."""
    # Find the locations of all conjunctions in the sentence
    conj_indices = []
    text = Text(sentence.text).analyse("morphology")
    word_idx = 0
    for word in text.words:
        part_of_speech = word.morph_analysis.partofspeech[0]
        # Ignore punctuation
        if part_of_speech == WordClass.PUNCTUATION:
            continue
        # Record the index of the conjunction
        if part_of_speech == WordClass.CONJUNCTION:
            conj_indices.append(word_idx)
        word_idx += 1

    # Find the index of the conjunction that is closest to the midpoint of the sentence.
    # This results in a more balanced split.
    if conj_indices:
        mid_idx = len(sentence.words) // 2
        split_idx = conj_indices[0]
        for conj_idx in conj_indices:
            if abs(conj_idx - mid_idx) < abs(split_idx - mid_idx):
                split_idx = conj_idx
        return split_idx - 1  # We want to split just before the conjunction
    return None


def find_split_intervals(sentence):
    """Get a sorted list of the inter-word intervals in the sentence, longest first."""
    intervals = {}
    for i in range(len(sentence.words) - 1):
        intervals[i] = sentence.words[i + 1].start - sentence.words[i].end
    return sorted(intervals.keys(), key=lambda x: intervals[x], reverse=True)


def split_sentence_lengths_are_ok(sentence, split_idx):
    """Check that the split sentence lengths are within the allowed range."""
    part_1 = Sentence(sentence.words[: split_idx + 1])
    part_2 = Sentence(sentence.words[split_idx + 1 :])
    duration_1 = part_1.end - part_1.start
    duration_2 = part_2.end - part_2.start
    return duration_1 > MIN_DURATION and duration_2 > MIN_DURATION


def convert_sentences(sentences):
    """Convert the sentences by expanding abbreviations, numbers, etc."""
    # TODO: Reimplement this
    yield from sentences


def extract_clip(input_wav, output_wav, start, end):
    """Extract a section from the input WAV file into a new file."""
    subprocess.check_call(
        [
            "ffmpeg",
            "-y",
            "-i",
            input_wav,
            "-ss",
            str(start),
            "-to",
            str(end),
            "-c",
            "copy",
            output_wav,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )


def main(args):
    _setup_logging(args.verbose)
    # Delete the output transcript file if it exists
    if os.path.exists(args.output_transcript):
        os.remove(args.output_transcript)
    sentence_idx = 1
    # Iterate over all .wav files in the input directory
    for wav_file in os.listdir(args.input_wav_dir):
        # Get the corresponding transcript file
        transcript_file = os.path.join(
            args.transcript_dir, wav_file.replace(".wav", ".json")
        )
        # Skip the file if the transcript file doesn't exist and print a warning
        if not os.path.exists(transcript_file):
            logger.warning(f"Transcript file {transcript_file} doesn't exist!")
            continue
        # Read the transcript JSON file
        with open(transcript_file, "r", encoding="utf-8") as f:
            transcript = json.load(f)
        # Get the timestamped words from the transcript
        words = extract_words(transcript)
        # Group words into sentences
        sentences = make_sentences(words)
        # Ensure that sentences are not too long and split them if necessary
        sentences = conform_sentences(sentences)
        # Expand abbreviations, numbers, etc. into full phonetic words
        sentences = convert_sentences(sentences)
        # Iterate over the sentences to split the audio file and create the transcript
        for sentence in sentences:
            sentence_id = f"{sentence_idx:04d}"
            # Split the audio file
            extract_clip(
                os.path.join(args.input_wav_dir, wav_file),
                os.path.join(args.output_wav_dir, sentence_id + ".wav"),
                sentence.start,
                sentence.end,
            )
            # Append the sentence to the transcript file
            with open(args.output_transcript, "a", encoding="utf-8") as f:
                f.write(sentence_id + ".wav|" + sentence.text + "\n")

            sentence_idx += 1
        return
    pass


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_wav_dir", default="input\\wav_full", help="Input audio files"
    )
    parser.add_argument(
        "--transcript_dir", default="input\\transcript", help="Transcript files"
    )
    parser.add_argument(
        "--output_wav_dir", default="input\\wav", help="Output audio files"
    )
    parser.add_argument(
        "--output_transcript",
        default="input\\metadata.csv",
        help="Output transcript file",
    )
    parser.add_argument(
        "-v", "--verbose", action="count", default=0, help="Verbosity (-v, -vv, etc)"
    )
    return parser.parse_args()


def _setup_logging(verbosity):
    """Set up logging."""
    logger.remove()
    verbosity_to_level = {
        0: "WARNING",  # No -v argument
        1: "INFO",  # -v
        2: "DEBUG",  # -vv
    }
    try:
        logger.add(sys.stderr, level=verbosity_to_level[verbosity])
    except KeyError:
        raise ValueError(f"Invalid verbosity level: {verbosity}")


if __name__ == "__main__":
    main(_parse_args())
