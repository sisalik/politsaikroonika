import argparse
import contextlib
import math
import os
import random
import re
import shutil
import subprocess
from pathlib import Path

import openai
from estnltk import Text
from loguru import logger
from tqdm import tqdm

from tts_preprocess_et.convert import convert_sentence

PYTHON_VENVS = {
    "Voice-Cloning-App": Path("Voice-Cloning-App/.venv/Scripts/python.exe"),
    "sd-webui-text2video": Path("sd-webui-text2video/.venv/Scripts/python.exe"),
}
# Silence between sentences in seconds
SILENCE_PADDING = 0.3
# Offset the start time of the subtitles to allow for the intro
INTRO_LENGTH = 4.08
# Video generation prompt for the first and last shots with the reporter
REPORTER_PROMPT = (
    "german male reporter talking into microphone, brown mullet hair, adult, "
    "90s black leather jacket, portrait shot, looking at camera, "
    "standing on 80s russian city street at night, dark"
)
# Shared negative video generation prompt
NEGATIVE_PROMPT = (
    "camera pan, moving camera, dynamic shot, text, watermark, copyright, blurry"
)
# Append style parameters to each video generation prompt
VIDEO_STYLE_EXTRA = ", 90s, russia, eastern europe"
# Output video frames per second (after 3x interpolation)
VIDEO_FPS = 8
# Minimum number of frames to generate for a video (otherwise the video is just noise)
VIDEO_FRAMES_MIN = 16
# Maximum number of frames to generate for a video (otherwise you run out of VRAM)
VIDEO_FRAMES_MAX = 48
# OpenAI model to use for text generation
OPENAI_MODEL = "gpt-3.5-turbo"


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--count", type=int, default=1, help="Number of episodes")
    parser.add_argument(
        "-r", "--redo", action="store_true", help="Redo existing episodes"
    )
    parser.add_argument(
        "-n",
        "--no-openai",
        action="store_true",
        help="Skip OpenAI and use hardcoded responses",
    )
    return parser.parse_args()


def _prompt_openai_model(prompt, max_tokens=300, temperature=0.7):
    """Initialize OpenAI API and make a request."""
    openai.api_key = os.environ["OPENAI_API_KEY"]
    response = openai.ChatCompletion.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    response_text = response["choices"][0]["message"]["content"]
    stop_reason = response["choices"][0]["finish_reason"]
    if stop_reason == "length":
        logger.warning("OpenAI API returned a truncated response")
    elif stop_reason != "stop":
        logger.warning(f"OpenAI API returned an unexpected stop reason: {stop_reason}")
    logger.debug(f"Total tokens: {response['usage']['total_tokens']}")
    logger.debug(f"Stop reason: {stop_reason}")
    logger.debug(f"Response text:\n{response_text}")
    return response_text


def _run_in_venv(venv, *commands):
    """Run commands in a specified Python virtual environment."""
    assert Path(venv).is_dir(), f"Invalid venv path: {venv}"
    return subprocess.run(
        [PYTHON_VENVS[venv], *commands], check=True, capture_output=True, cwd=venv
    )


@contextlib.contextmanager
def _run_in_venv_and_monitor_output(venv, *commands):
    """Run commands in a specified Python virtual environment and monitor output."""
    assert Path(venv).is_dir(), f"Invalid venv path: {venv}"
    with subprocess.Popen(
        [PYTHON_VENVS[venv], *commands],
        cwd=venv,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        universal_newlines=True,
    ) as proc:
        yield proc

    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, proc.args)


def _get_audio_file_duration(audio_file):
    """Get audio file duration in seconds."""
    output = subprocess.run(
        "ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1".split()
        + [audio_file],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(output.stdout)


def _seconds_to_frame_splits(seconds):
    """Convert seconds to video frames."""
    if seconds == 0:
        raise ValueError("Cannot convert 0 seconds to frames")
    frames = math.ceil(seconds * VIDEO_FPS)
    if frames <= VIDEO_FRAMES_MIN:
        return [VIDEO_FRAMES_MIN]
    elif frames > VIDEO_FRAMES_MAX:
        # Initially split the video into multiple parts with even lengths
        n_parts = math.ceil(frames / VIDEO_FRAMES_MAX)
        # even_splits = [math.ceil(frames / n_parts)] * n_parts
        # even_splits[-1] -= sum(even_splits) - frames
        # Randomly shuffle the split points, ensuring that each split between
        # VIDEO_FRAMES_MIN and VIDEO_FRAMES_MAX frames long
        splits = []
        for i in range(n_parts - 1):
            remaining_frames = frames - sum(splits)
            # Lower bound, assuming that the remaining splits are all at the
            # maximum length
            lower_bound = max(
                VIDEO_FRAMES_MIN,
                remaining_frames - VIDEO_FRAMES_MAX * (n_parts - i - 1),
            )
            # Upper bound, leaving room for a minimum length split at the end
            upper_bound = min(VIDEO_FRAMES_MAX, remaining_frames - VIDEO_FRAMES_MIN)
            splits.append(random.randint(lower_bound, upper_bound))
        splits.append(frames - sum(splits))
        random.shuffle(splits)  # For good measure
        return splits
    else:
        return [frames]


def gen_title(no_openai=False):
    """Generate a title for the episode."""
    if no_openai:
        return "Vanaproua tõstis oma korteris üles kasvatatud krokodilli politsei sekkumiseta"
    prompt = """
Generate titles for an Estonian police and crime news TV segment. There should be 5 titles and each one should be numbered. Each title should be in Estonian and describe an incredibly bizarre, tragic and specific criminal event. It should include who was involved and where it happened. Some examples:
- Honda juht sõitis meelega otsa ohutussaarel olnud inimestele
- Jõgi neelas veoki ja kaks traktorit
- 82-aastane vanahärra keeras auto katusele
- Lapsed kihutasid Datsuniga sillalt alla
- Lennuk maandus Pirita rannas
- Maardu kandis jäi põder auto alla
- Mees jäi hammastega rooli külge kinni
- Purjus ja lubadeta mootorrattur rammis bussi
- 62-aastane Uno röövis panka
- Röövlid õhkasid Õismäel pangaautomaadi
- Tabati leidlikud salapiirituse valmistajad
- Politseireid Sõle tänava bordelli kulges vägivaldselt
- Narkomaanid varastasid raamatukogu tühjaks
- Politsei arreteeris Kopli narkodiilerid suurte jõududega
- Vanas sõjaväetelgis avastati ebahügieeniline vorstivabrik"""
    response = _prompt_openai_model(prompt.strip())
    # Convert the numbered list to a Python list
    title_candidates = re.findall(r"^\d+\. (.+)$", response, re.MULTILINE)
    # Convert back to a string for the next prompt
    title_candidates_str = "\n".join(
        f"{i + 1}. {title}" for i, title in enumerate(title_candidates)
    )
    prompt = f"""
Which one of these sentences in Estonian stands out as the one you would least expect to see as a news headline? Pick the most unexpected and weird one. Only reply with the number.
{title_candidates_str}"""
    title = _prompt_openai_model(prompt.strip(), max_tokens=1)  # Only need the number
    try:
        selected_title = title_candidates[int(title) - 1]
    except ValueError:
        raise ValueError(f"Invalid title number: {title}")
    # Remove the number from the title
    return selected_title


def gen_summary(title, no_openai=False):
    """Generate a summary for the episode."""
    if no_openai:
        return "An elderly woman in Estonia raised a crocodile in her apartment without any police intervention. The woman claimed the crocodile was her late husband's pet and she couldn't bear to part with it. Neighbors reported the animal to authorities, but the woman was allowed to keep it after proving she could care for it properly."
    prompt = f"""
Imagine there is a crime news article in Estonian titled "{title}". Can you make up a short 3-sentence summary of the events that took place, including a bizarre reason/explanation/motive?"""
    return _prompt_openai_model(prompt.strip())


def gen_script(summary, no_openai=False):
    """Generate a script for the episode."""
    if no_openai:
        return "Tere õhtust ja tere tulemast meie uudistesse! Täna räägime teile ühest kummalisest juhtumist Eesti linnas. Nimelt avastasid naabrid, et üks vanem naine kasvatas oma korteris krokodilli! Naine väitis, et tegemist on tema hiljuti surnud abikaasa lemmikloomaga ning ta ei taha sellest loobuda. Pärast politsei sekkumist suutis naine tõestada, et ta on looma eest hoolitsemiseks piisavalt pädev ning krokodill lubati tema juures edasi elada. Kuidas see võimalik oli? Kas politsei tegi õigesti? Kas on turvaline elada krokodilli kõrval? Küsimused, mis jätavad meid mõtlema."
    prompt = f"""
Generate the script for an Estonian police and crime news TV segment. The segment is written in the Estonian language and its short summary is as follows:

"{summary}"

Constraints are listed below, in no particular order. Do not follow these as plot points in chronological order; use them as guidance throughout the script, in random order.
- The word count should be up to 120 words
- The script is intended to be read out by the news reporter for a made-up TV channel
- Start by addressing the TV channel viewers and stating the location and time of the event
- The criminal events should be rather strange and oddly specific
- Describe the tragic events and casualties in an edgy and poetic, yet graphic and detailed manner
- Speak somewhat demeaningly of the victims
- Describe the actions of the police officers
- Use old-fashioned metaphors and proverbs
- End with one sentence with a thought-provoking statement that is not obvious or cliché, e.g. crime is bad
- Do not address the viewers again or sign off at the end of the segment"""
    return _prompt_openai_model(prompt.strip())


def split_sentences(script):
    """Split script into sentences using EstNLTK."""
    script_text = Text(script)
    script_text.analyse("morphology")
    for sentence in script_text.sentences:
        yield sentence.enclosing_text


def gen_audio(sentences, ep_idx):
    """Generate audio files for the sentences using TTS."""
    filenames = [
        # Use absolute paths to avoid problems with relative paths in the venv
        Path(f"output/episode_{ep_idx:03d}/sentences/sentence_{i+1:02d}.wav").resolve()
        for i in range(len(sentences))
    ]
    # Make sure the output directory exists
    filenames[0].parent.mkdir(parents=True, exist_ok=True)

    args = []
    for sentence, filename in zip(sentences, filenames):
        args.extend(["-t", sentence, "-a", filename])

    with _run_in_venv_and_monitor_output(
        "Voice-Cloning-App",
        Path("synthesis/synthesize.py"),
        "--model_path",
        Path("data/models/peeter2/checkpoint_30000"),
        "--vocoder_model_path",
        Path("data/hifigan/vctk/model.pt"),
        "--hifigan_config_path",
        Path("data/hifigan/vctk/config.json"),
        "--alphabet",
        Path("data/languages/Estonian/alphabet.txt"),
        *args,
    ) as proc:
        with tqdm(total=len(sentences)) as pbar:
            for line in proc.stdout:
                if line.startswith("Audio file saved"):
                    pbar.update(1)
    return filenames


def create_subtitles(sentences, audio_lenghts, ep_idx):
    """Create subtitles for the episode, in SRT format."""
    length_accumulator = INTRO_LENGTH
    filename = Path(f"output/episode_{ep_idx:03d}/subtitles.srt").resolve()
    with open(filename, "w", encoding="utf-8") as f:
        for i, (sentence, audio_length) in enumerate(zip(sentences, audio_lenghts)):
            f.write(f"{i+1}\n")
            start_time = f"00:00:{length_accumulator:.03f}".replace(".", ",")
            end_time = f"00:00:{length_accumulator+audio_length:.03f}".replace(".", ",")
            f.write(f"{start_time} --> {end_time}\n")
            f.write(sentence + "\n\n")
            length_accumulator += audio_length + SILENCE_PADDING
    return filename


def gen_video_prompts(summary, no_openai=False):
    """Generate video prompts for the episode."""
    if no_openai:
        video_prompts = [
            "elderly woman in floral dress holding a small crocodile, apartment interior with yellow walls, potted plants, cluttered",
            "neighbors gathered outside apartment, pointing and gesturing, police officers in blue uniforms, clipboard, serious expressions",
            "elderly woman smiling, petting the crocodile on her lap, police officers nodding in approval, apartment background with patterned curtains and wooden furniture",
            "close-up of the crocodile's scaly skin and sharp teeth, woman feeding it raw meat, kitchen in background with stainless steel appliances and tiled backsplash",
            "woman walking the crocodile on a leash in a nearby park, green grass, trees, passerby staring in disbelief",
        ]
        video_prompts = [prompt + VIDEO_STYLE_EXTRA for prompt in video_prompts]
        return [REPORTER_PROMPT] + video_prompts + [REPORTER_PROMPT]
    prompt = f"""
Generate captions for a photographic storyboard for an Estonian police and crime news TV segment. The segment is written in the Estonian language and its short summary is as follows:

"{summary}"

There should be 5 captions in total. The captions should be:
- written in very terse, news style English
- describe photographic stills of the news segment
- one per line
- formatted as a comma-separated list of key words and phrases, omitting verbs
- include detailed information about the subject (color, shape, texture, size), background and image style
- in chronological order to form a coherent story

Examples:
- close-up of a green rusty door, small hidden opening, people entering, flashlight
- interior of abandoned building, large vats and pipes, various bottles and containers, cobwebs, dark, ambient lighting
- factory exterior, large crowd of onlookers, police officers, police cars, flashing lights"""
    response = _prompt_openai_model(prompt.strip())
    # Convert the Markdown bulleted list of prompts into a Python list
    video_prompts = re.findall(r"[-+*]\s+(.*)", response)
    # Append the style prompt to each API response
    video_prompts = [prompt + VIDEO_STYLE_EXTRA for prompt in video_prompts]
    return [REPORTER_PROMPT] + video_prompts + [REPORTER_PROMPT]


def distribute_prompts(prompts, audio_lengths):
    """Calculate the length of each prompt based on the audio length."""
    # Handle opening and closing shots separately
    opening_length = _seconds_to_frame_splits(audio_lengths[0])[0]
    closing_length = _seconds_to_frame_splits(audio_lengths[-1])[0]
    # Split the remaining duration among the other prompts
    total_audio_length = sum(audio_lengths) + SILENCE_PADDING * (len(audio_lengths) - 1)
    remaining_length = (
        total_audio_length - (opening_length + closing_length) / VIDEO_FPS
    )
    splits = _seconds_to_frame_splits(remaining_length)
    if len(prompts) - 2 > len(splits):
        logger.warning(
            f"Too many prompts ({len(prompts)}) for the audio ({len(splits)})."
        )
        # Slice out some of the middle prompts
        prompts = [prompts[0]], *prompts[1 : len(splits) + 1], [prompts[-1]]
    # Distribute prompts evenly, kind of like nearest neighbor scaling
    dist_prompts = []
    for i in range(len(splits)):
        prompt_idx = round(i * (len(prompts) - 2) / len(splits)) + 1
        dist_prompts.append(prompts[prompt_idx])
    dist_prompts = [prompts[0]] + dist_prompts + [prompts[-1]]
    lenghts = [opening_length] + splits + [closing_length]
    return dist_prompts, lenghts


def gen_video_clips(prompts, lengths, ep_idx):
    """Generate video clips using the prompts and lengths."""
    dirs = [
        # Use absolute paths to avoid problems with relative paths in the venv
        Path(f"output/episode_{ep_idx:03d}/clips/clip_{i+1:02d}").resolve()
        for i in range(len(prompts))
    ]
    args = []
    for prompt, length, outdir in zip(prompts, lengths, dirs):
        args.extend(["--prompt", prompt, "--frames", str(length), "--outdir", outdir])

    with _run_in_venv_and_monitor_output(
        "sd-webui-text2video",
        Path("scripts/text2vid_cli.py"),
        "--n_prompt",
        NEGATIVE_PROMPT,
        "--cfg_scale",
        "10",
        *args,
    ) as proc:
        lenghts_iter = iter(lengths)
        with tqdm(total=sum(lengths), unit="frame") as pbar:
            for line in proc.stdout:
                # logger.debug(line)
                if line.startswith("t2v complete"):
                    pbar.update(next(lenghts_iter))
    return dirs


def make_episode(ep_idx, no_openai=False):
    """Make a single episode with a given index."""
    print()
    logger.info(f"Making episode {ep_idx:03d}...")
    logger.info("Generating episode title...")
    title = gen_title(no_openai)
    logger.info(f"Episode title: {title}")

    logger.info("Generating episode summary...")
    summary = gen_summary(title, no_openai)
    logger.info(f"Episode summary:\n{summary}")

    logger.info("Generating episode script...")
    script = gen_script(summary, no_openai)
    logger.info(f"Episode script:\n{script}")
    sentences = [convert_sentence(s) for s in split_sentences(script)]

    logger.info(f"Generating audio for {len(sentences)} sentences...")
    audio_files = gen_audio(sentences, ep_idx)
    audio_lengths = [_get_audio_file_duration(filename) for filename in audio_files]
    total_audio_length = sum(audio_lengths) + SILENCE_PADDING * (len(audio_lengths) - 1)
    for sentence, length in zip(sentences, audio_lengths):
        short_sentence = sentence[:50] + "..." if len(sentence) > 50 else sentence
        logger.debug(f"  {short_sentence}: {length:.2f}s")
    logger.info(f"Total audio length: {total_audio_length:.2f}s")

    logger.info("Creating subtitles...")
    subtitles_file = create_subtitles(sentences, audio_lengths, ep_idx)

    logger.info("Generating video prompts...")
    prompts = gen_video_prompts(summary, no_openai)
    prompts, prompt_lenghts = distribute_prompts(prompts, audio_lengths)
    for prompt, length in zip(prompts, prompt_lenghts):
        short_prompt = prompt[:50] + "..." if len(prompt) > 50 else prompt
        logger.debug(f"  {short_prompt}: {length} frames")

    logger.info("Generating video clips...")
    clip_dirs = gen_video_clips(prompts, prompt_lenghts, ep_idx)
    logger.success(f"Episode {ep_idx:03d} complete!")


def make_episodes(args):
    """Make episodes."""
    # Find the last episode index from the output directory
    episode_dirs = sorted(Path("output").glob("episode_*"))
    last_episode_idx = int(episode_dirs[-1].name.split("_")[-1])
    if args.redo:
        # Delete the last args.count episodes
        start_idx = max(0, last_episode_idx - args.count + 1)
    else:
        start_idx = last_episode_idx + 1
    for i in range(start_idx, start_idx + args.count):
        # Delete the episode directory if it exists
        episode_dir = Path(f"output/episode_{i:03d}")
        if episode_dir.exists():
            shutil.rmtree(episode_dir)
        make_episode(i, args.no_openai)


if __name__ == "__main__":
    with logger.catch():
        make_episodes(_parse_args())
