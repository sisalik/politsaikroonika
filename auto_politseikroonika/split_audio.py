import argparse
import json
import os
import subprocess
from dataclasses import dataclass

from estnltk import Text
from tts_preprocess_et.convert import convert_sentence


# The speaker IDs that are used in the transcript JSON file to tag the speech we are
# interested in
SPEAKER_IDS = ["VO", "Mic"]
# Maximum desired sentence duration in seconds
MAX_DURATION = 10
# Minimum desired sentence duration in seconds
MIN_DURATION = 1
# EstNLTK part-of-speech tag for conjunctions. See:
# https://github.com/estnltk/estnltk/blob/main/tutorials/nlp_pipeline/B_morphology/00_tables_of_morphological_categories.ipynb
TAG_CONJUNCTION = "J"


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
        return " ".join([word.text for word in self.words]) + "."

    @property
    def start(self):
        return self.words[0].start

    @property
    def end(self):
        return self.words[-1].end


def parse_args():
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
    return parser.parse_args()


def extract_sentences(transcript):
    """Extract the sentences from the transcript JSON file."""
    sentences = []
    for block in transcript["content"]:
        if block["type"] != "speaker":
            continue
        if block["attrs"]["data-name"] not in SPEAKER_IDS:
            continue

        words = []
        for element in block["content"]:
            if element["type"] != "text":
                continue

            words.append(
                Word(
                    text=element["text"].strip(),
                    start=float(element["marks"][0]["attrs"]["start"]),
                    end=float(element["marks"][0]["attrs"]["end"]),
                )
            )
            # If the word is the last in the sentence, add the sentence to the list
            if words[-1].text.endswith("."):
                words[-1].text = words[-1].text[:-1]  # Remove the period
                sentences.append(Sentence(words))
                words = []  # Reset the word list
        # In case the last string doesn't end with a period, but the block ends,
        # consider it a sentence
        if words:
            sentences.append(Sentence(words))
    return sentences


def convert_sentences(sentences):
    """Convert the sentences by expanding abbreviations, numbers, etc."""
    converted_text = convert_sentence(" ".join(sentence.text for sentence in sentences))
    # Match the converted text to the original sentences. This is tricky because some
    # words are expanded, so we need to find the original word in the list of all words
    # and replace it with the expanded version.
    all_words = [word for sentence in sentences for word in sentence.words]
    converted_sentences = []
    word_idx = 0
    for sentence in converted_text.split("."):
        sentence = sentence.strip()
        if not sentence:
            continue

        converted_words = []
        for word in sentence.split(" "):
            word = word.strip()
            if not word:
                continue

            if word == all_words[word_idx].text:
                converted_words.append(all_words[word_idx])
                word_idx += 1
            # The word has been expanded
            elif word != all_words[word_idx + 1].text:
                new_word = Word(
                    text=word,
                    start=all_words[word_idx].start,
                    end=all_words[word_idx].end,
                )
                converted_words.append(new_word)
            else:
                converted_words.append(all_words[word_idx + 1])
                word_idx += 2

        converted_sentences.append(Sentence(converted_words))
    return converted_sentences


def conform_sentences(sentences):
    """Ensure that sentences are not too long and split them if necessary."""
    out_sentences = []
    for sentence in sentences:
        duration = sentence.end - sentence.start
        print(
            f"{sentence.start} - {sentence.end} ({duration:.2f} s):\n  {sentence.text}"
        )
        out_sentences.extend(split_sentence_if_needed(sentence))

    return out_sentences


def split_sentence_if_needed(sentence):
    duration = sentence.end - sentence.start
    if duration <= MAX_DURATION:
        return [sentence]

    print(f"WARNING: Sentence too long ({duration:.2f} s > {MAX_DURATION} s)")
    # Try to split the sentence at a conjunction
    split_idx = find_split_idx_conjunction(sentence)
    if split_idx is None or not split_sentence_lengths_are_ok(sentence, split_idx):
        # No conjunction found, or the split would result in a sentence too short
        for split_idx in find_split_intervals(sentence):
            if split_sentence_lengths_are_ok(sentence, split_idx):
                break
        else:
            # No split interval found
            print("ERROR: No suitable split interval found")
            return [sentence]

    part_1 = Sentence(sentence.words[: split_idx + 1])
    part_2 = Sentence(sentence.words[split_idx + 1 :])
    return split_sentence_if_needed(part_1) + split_sentence_if_needed(part_2)


def find_split_idx_conjunction(sentence):
    """Find the index of the last word before a conjunction."""
    # Find the locations of all conjunctions in the sentence
    conj_indices = []
    text = Text(sentence.text).analyse("morphology")
    for i, pos_tags in enumerate(text.morph_analysis.partofspeech):
        pos_tag = pos_tags[0]
        if pos_tag == TAG_CONJUNCTION:
            conj_indices.append(i)
    # Find the index of the conjunction that is closest to the midpoint of the sentence.
    # This results in a more balanced split.
    if conj_indices:
        mid_idx = len(sentence.words) // 2
        split_idx = conj_indices[0]
        for conj_idx in conj_indices:
            if abs(conj_idx - mid_idx) < abs(split_idx - mid_idx):
                split_idx = conj_idx
        return split_idx
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
            print(f"WARNING: Transcript file {transcript_file} doesn't exist!")
            continue
        # Read the transcript JSON file
        with open(transcript_file, "r", encoding="utf-8") as f:
            transcript = json.load(f)
        # Get the timestamped sentences
        sentences = extract_sentences(transcript)
        # Expand abbreviations, numbers, etc. into full phonetic words
        sentences = convert_sentences(sentences)
        # Ensure that sentences are not too long and split them if necessary
        sentences = conform_sentences(sentences)
        # return
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
                f.write(sentence_id + "|" + sentence.text + "\n")

            sentence_idx += 1
        return
    pass


if __name__ == "__main__":
    main(parse_args())
