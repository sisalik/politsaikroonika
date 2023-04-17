import argparse
import contextlib
import math
import os
import random
import re
import shutil
import subprocess
import time
from pathlib import Path

import openai
import toml
from estnltk import Text
from loguru import logger
from tqdm import tqdm

from tts_preprocess_et.convert import convert_sentence
from auto_politseikroonika.google_drive import FolderUploader

PYTHON_VENVS = {
    "Voice-Cloning-App": Path("Voice-Cloning-App/.venv/Scripts/python.exe"),
    "sd-webui-text2video": Path("sd-webui-text2video/.venv/Scripts/python.exe"),
}
# Silence between sentences in seconds
SILENCE_PADDING = 0.3
# Video generation prompt for the first and last shots with the reporter
REPORTER_PROMPT = (
    "norwegian adult male reporter talking into microphone, (tony hawk hair:0.9), "
    "90s black leather jacket, portrait shot, tripod, looking at camera, "
    "standing on 80s russian city street"
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
# Topaz Video AI model path
TOPAZ_MODEL_DIR = Path("C:/ProgramData/Topaz Labs LLC/Topaz Video AI/models")
# Topaz Video AI ffmpeg executable path
TOPAZ_FFMPEG = Path("C:/Program Files/Topaz Labs LLC/Topaz Video AI/ffmpeg.exe")


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--count", type=int, default=1, help="Number of episodes")
    parser.add_argument(
        "-r", "--redo", action="store_true", help="Redo existing episodes"
    )
    parser.add_argument("-a", "--avoid", type=str, action="append", help="Avoid topics")
    parser.add_argument(
        "-k", "--keep_intermediate", action="store_true", help="Keep intermediate files"
    )
    parser.add_argument(
        "-w",
        "--when_done",
        type=str,
        choices=["sleep"],
        help="Action to perform after episode generation",
    )
    parser.add_argument(
        "-n",
        "--no-openai",
        action="store_true",
        help="Skip OpenAI and use hardcoded responses",
    )
    args = parser.parse_args()
    # Allow "--avoid topic1,topic2" syntax as well as "--avoid topic1 --avoid topic2"
    if len(args.avoid) == 1 and "," in args.avoid[0]:
        args.avoid = args.avoid[0].split(",")
    return args


def _prompt_openai_model(
    prompt, max_tokens=256, temperature=0.7, allow_truncated=False, max_attempts=10
):
    """Initialize OpenAI API and make a request."""
    openai.api_key = os.environ["OPENAI_API_KEY"]
    for attempt in range(max_attempts):
        response = openai.ChatCompletion.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        response_text = response["choices"][0]["message"]["content"]
        stop_reason = response["choices"][0]["finish_reason"]
        logger.debug(
            f"Tokens: {response['usage']['prompt_tokens']} (prompt) + "
            f"{response['usage']['completion_tokens']} (completion) = "
            f"{response['usage']['total_tokens']} (total)"
        )
        logger.debug(f"Response text:\n{response_text}")
        if stop_reason != "stop" and not allow_truncated:
            logger.warning(
                f"OpenAI API returned an unexpected stop reason: {stop_reason}"
            )
            logger.info(f"Retrying ({attempt + 1}/{max_attempts})...")
            continue

        return response_text
    raise Exception("OpenAI API failed to return a valid response")


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


def _pc_sleep():
    """Put the computer to sleep."""
    subprocess.run("rundll32.exe powrprof.dll,SetSuspendState 0,1,0".split())


def _get_media_file_duration(media_file):
    """Get audio/video file duration in seconds."""
    output = subprocess.run(
        "ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1".split()
        + [str(media_file)],
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
        # Randomly shuffle the split points, ensuring that each split between
        # VIDEO_FRAMES_MIN and VIDEO_FRAMES_MAX frames long
        splits = []
        n_parts = math.ceil(frames / VIDEO_FRAMES_MAX)
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


def _record_metadata(file, metadata):
    """Append metadata to a TOML file."""
    if Path(file).is_file():
        with open(file) as f:
            data = toml.load(f)
    else:
        data = {}
    data.update(metadata)
    with open(file, "w") as f:
        toml.dump(data, f)


def gen_title(avoid_topics=None, no_openai=False):
    """Generate a title for the episode."""
    if no_openai:
        return "Vanaproua tõstis oma korteris üles kasvatatud krokodilli politsei sekkumiseta"
    if avoid_topics:
        avoid_prompt = (
            f"Avoid mentioning the following topics: {', '.join(avoid_topics)}. "
        )
    else:
        avoid_prompt = ""
    prompt = f"""
Generate titles for an Estonian police and crime news TV segment. There should be 10 titles and each one should be numbered. Each title should be in Estonian and describe an incredibly bizarre, tragic and specific criminal event. It should include who was involved and where it happened. {avoid_prompt}Some examples:
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
    response = _prompt_openai_model(prompt.strip(), max_tokens=360)
    # Convert the numbered list to a Python list
    title_candidates = re.findall(r"^\d+\. (.+)$", response, re.MULTILINE)
    # Convert back to a string for the next prompt
    title_candidates_str = "\n".join(
        f"{i + 1}. {title}" for i, title in enumerate(title_candidates)
    )
    prompt = f"""
Which one of these sentences in Estonian stands out as the one you would least expect to see as a news headline? Pick the most unexpected and weird one. Only reply with the number.
{title_candidates_str}"""
    # Only need the number, so allow truncated responses of length 1
    title = _prompt_openai_model(prompt.strip(), max_tokens=1, allow_truncated=True)
    try:
        selected_title = title_candidates[int(title) - 1]
    except ValueError:
        raise ValueError(f"Invalid title number: {title}")
    # Remove the number from the title
    return selected_title


def gen_summary(title, avoid_topics=None, no_openai=False):
    """Generate a summary for the episode."""
    if no_openai:
        return "An elderly woman in Estonia raised a crocodile in her apartment without any police intervention. The woman claimed the crocodile was her late husband's pet and she couldn't bear to part with it. Neighbors reported the animal to authorities, but the woman was allowed to keep it after proving she could care for it properly."
    prompt = f"""
Imagine there is a crime news article in Estonian titled "{title}". Can you make up a short 3-sentence summary of the events that took place, including a bizarre reason/explanation/motive?"""
    if avoid_topics:
        prompt += f"Avoid mentioning the following topics: {', '.join(avoid_topics)}."
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
- The criminal event should be rather strange and oddly specific
- Only write about a single criminal event, not several
- Describe the tragic events and casualties in an edgy and poetic, yet graphic and detailed manner
- Speak somewhat demeaningly of the victims
- Describe the actions of the police officers
- Use old-fashioned metaphors and proverbs
- End with one sentence with a thought-provoking statement that is not obvious or cliché, e.g. crime is bad
- Do not address the viewers again or sign off at the end of the segment"""
    # Limit the number of tokens to yield roughly 120 words or 50 seconds of audio
    script = _prompt_openai_model(prompt.strip(), max_tokens=360)
    # Sometimes the entire response is in quotes, so remove them
    if script.startswith('"') and script.endswith('"'):
        script = script[1:-1]
    return script


def split_sentences(script):
    """Split script into sentences using EstNLTK."""
    # Swap any quotes and periods that throw off the sentence splitter
    script = script.replace('."', '".')
    script_text = Text(script)
    script_text.analyse("morphology")
    for sentence in script_text.sentences:
        yield sentence.enclosing_text


def convert_sentences(raw_sentences):
    """Convert sentences to a more pronouncable format suitable for TTS."""
    custom_replacements = [
        # The T is too hard
        ("kurjategija", "kurjadegija"),
        ("kuritegu", "kuridegu"),
        ("kuriteo", "kurideo"),
    ]
    for raw in raw_sentences:
        converted = convert_sentence(raw)
        for old, new in custom_replacements:
            converted = converted.lower().replace(old, new)
        yield converted


def gen_audio(sentences, ep_dir):
    """Generate audio files for the sentences using TTS."""
    filenames = [
        # Use absolute paths to avoid problems with relative paths in the venv
        (ep_dir / f"sentences/sentence_{i+1:02d}.wav").resolve()
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
        with tqdm(total=len(sentences), unit="sentence") as pbar:
            for line in proc.stdout:
                if line.startswith("Audio file saved"):
                    pbar.update(1)
    return filenames


def merge_audio(audio_files):
    """Merge audio files into one, inserting silence between each file."""
    # Create a temporary audio file that contains silence for SILENCE_PADDING seconds
    silence_file = audio_files[0].parent / "silence.wav"
    subprocess.run(
        [
            "ffmpeg",
            "-f",
            "lavfi",
            "-i",
            f"anullsrc=r=22050:cl=mono",
            "-t",
            str(SILENCE_PADDING),
            str(silence_file),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    # Add a silence file between each audio file. This is done by duplicating each
    # list element (except for the last one) and replacing every second element with
    # the silence file.
    audio_files = [audio_file for audio_file in audio_files for _ in range(2)][:-1]
    audio_files[1::2] = [silence_file for _ in range(len(audio_files) // 2)]
    # Merge the audio files
    input_args = []
    filter_args = ""
    for i, audio_file in enumerate(audio_files):
        input_args.extend(["-i", audio_file])
        filter_args += f"[{i}:0]"

    out_file = audio_files[0].parent / "sentences.wav"
    subprocess.run(
        [
            "ffmpeg",
            *input_args,
            "-filter_complex",
            f"{filter_args}concat=n={len(audio_files)}:v=0:a=1[out]",
            "-map",
            "[out]",
            str(out_file),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    # Remove the temporary silence file
    silence_file.unlink()
    return out_file


def create_subtitles(sentences, audio_lenghts, ep_dir):
    """Create subtitles for the episode, in SRT format."""
    length_accumulator = 0
    filename = (ep_dir / "subtitles.srt").resolve()
    with open(filename, "w", encoding="utf-8") as f:
        for i, (sentence, audio_length) in enumerate(zip(sentences, audio_lenghts)):
            f.write(f"{i+1}\n")
            start_time = f"00:00:{length_accumulator:.03f}".replace(".", ",")
            end_time = f"00:00:{length_accumulator+audio_length:.03f}".replace(".", ",")
            f.write(f"{start_time} --> {end_time}\n")
            if sentence.endswith("."):
                sentence = sentence[:-1]
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

Avoid these keywords: aerial view, crowd

Examples:
- close-up of a green rusty door, small hidden opening, people entering, flashlight
- interior of abandoned building, large vats and pipes, various bottles and containers, cobwebs, dark, ambient lighting
- factory exterior, large crowd of onlookers, police officers, police cars, flashing lights"""
    response = _prompt_openai_model(prompt.strip())
    video_prompts = (line.strip() for line in response.splitlines() if line.strip())
    # Remove bullet points in case there are any
    video_prompts = [re.sub(r"^\s*[-*+]\s*", "", prompt) for prompt in video_prompts]
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
        prompts = [prompts[0], *prompts[1 : len(splits) + 1], prompts[-1]]
    # Distribute prompts evenly, kind of like nearest neighbor scaling
    dist_prompts = []
    for i in range(len(splits)):
        prompt_idx = round(i * (len(prompts) - 2) / len(splits)) + 1
        dist_prompts.append(prompts[prompt_idx])
    dist_prompts = [prompts[0]] + dist_prompts + [prompts[-1]]
    lenghts = [opening_length] + splits + [closing_length]
    return dist_prompts, lenghts


def gen_video_clips(prompts, lengths, ep_dir):
    """Generate video clips using the prompts and lengths."""
    dirs = [
        # Use absolute paths to avoid problems with relative paths in the venv
        (ep_dir / f"clips/clip_{i+1:02d}").resolve()
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


def enhance_video_clips(clips_dirs):
    """Interpolate and upscale video frames using Topaz Video AI."""
    # Set Topaz Video AI environment variables
    os.environ["TVAI_MODEL_DATA_DIR"] = str(TOPAZ_MODEL_DIR)
    os.environ["TVAI_MODEL_DIR"] = str(TOPAZ_MODEL_DIR)
    output_dirs = []
    total_frames = sum(len(list(clip_dir.glob("*.png"))) for clip_dir in clips_dirs)
    with tqdm(total=total_frames, unit="frame") as pbar:
        for clip_dir in clips_dirs:
            input_sequence = clip_dir / "%06d.png"
            output_sequence = clip_dir / "enhanced" / "%06d.png"
            output_sequence.parent.mkdir(exist_ok=True)
            output_dirs.append(output_sequence.parent)
            # TODO: Thread this and run two instances of ffmpeg at the same time
            subprocess.run(
                [
                    TOPAZ_FFMPEG,
                    "-nostdin",
                    "-y",
                    "-framerate",
                    "24",
                    "-start_number",
                    "0",
                    "-i",
                    input_sequence,
                    "-sws_flags",
                    "spline+accurate_rnd+full_chroma_int",
                    "-color_trc",
                    "2",
                    "-colorspace",
                    "0",
                    "-color_primaries",
                    "2",
                    "-filter_complex",
                    "tvai_fi=model=apo-8:slowmo=3:rdt=0.01:device=0:vram=1:instances=1,tvai_up=model=thd-3:scale=0:w=1024:h=1024:noise=0:blur=0:compression=0:device=0:vram=1:instances=1,scale=w=1024:h=1024:flags=lanczos:threads=0",
                    "-c:v",
                    "png",
                    "-pix_fmt",
                    "rgb24",
                    "-start_number",
                    "0",
                    output_sequence,
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            pbar.update(len(list(clip_dir.glob("*.png"))))
    return output_dirs


def merge_video_clips(clip_dirs):
    """Merge the video clips (image sequences) into a single video."""
    output_dir = clip_dirs[0].parent.parent / "merged"
    output_dir.mkdir(exist_ok=True)

    frame_idx = 0
    for clip_dir in clip_dirs:
        for png_file in clip_dir.glob("*.png"):
            new_filename = f"{frame_idx:06d}.png"
            shutil.copy(png_file, output_dir / new_filename)
            frame_idx += 1
    return output_dir


def merge_video_audio_subtitles(video_dir, audio_file, subtitles_file):
    """Merge the audio and video files into a single video file and add subtitles."""
    # TODO: Combine this step with the final render to avoid re-encoding the video
    merged_file = video_dir.parent.parent / "clips" / "merged.mp4"
    # Escape characters that cause problems with the ffmpeg filter syntax
    subtitle_path = str(subtitles_file).replace("\\", "/\\").replace(":", "\\\\:")
    subprocess.run(
        [
            "ffmpeg",
            "-framerate",
            "24",
            "-i",
            str(video_dir / "%06d.png"),
            "-i",
            str(audio_file),
            # Apply subtitles with a fixed font size
            "-vf",
            f"subtitles={subtitle_path}:force_style='Fontsize=12'",
            # Render a lossless video to avoid quality loss in this intermediate step
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-qp",
            "0",
            "-c:a",
            "libmp3lame",
            "-b:a",
            "192k",
            # Might be needed if there is a mismatch in the audio and video lengths
            # "-shortest",
            str(merged_file),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    # Delete the PNG video frames
    shutil.rmtree(video_dir)
    return merged_file


def prepend_intro_and_final_render(input_file, title):
    """Prepend the intro and final render the video file."""
    episode_dir = input_file.parent.parent
    # Include the episode title in the output file name
    output_file = (
        episode_dir / f"{episode_dir.name}_{title.lower().replace(' ', '.')}.mp4"
    )
    intro_file = Path("resources/intro.mp4")
    subprocess.run(
        [
            "ffmpeg",
            "-i",
            str(intro_file),
            "-i",
            str(input_file),
            # Merge the two videos (audio and video streams) and scale the video to 720p
            "-filter_complex",
            "[0:v][0:a][1:v][1:a]concat=n=2:v=1:a=1[outv][outa]; "
            "[outv]scale=-2:720[scaledv]",
            "-map",
            "[scaledv]",
            "-map",
            "[outa]",
            # Render the video with a reasonably high quality
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            "-crf",
            "23",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            str(output_file),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    # Delete the temporary video file
    input_file.unlink()
    return output_file


def make_episode(ep_dir, avoid_topics, no_openai=False):
    """Make a single episode with a given index."""
    process_start_time = time.time()
    print()
    logger.info(f"Making episode {ep_dir.name}...")
    logger.info("Generating episode title...")
    title = gen_title(avoid_topics, no_openai)
    logger.info(f"Episode title: {title}")

    logger.info("Generating episode summary...")
    summary = gen_summary(title, avoid_topics, no_openai)
    logger.info(f"Episode summary:\n{summary}")

    logger.info("Generating episode script...")
    script = gen_script(summary, no_openai)
    logger.info(f"Episode script:\n{script}")
    raw_sentences = list(split_sentences(script))
    converted_sentences = list(convert_sentences(raw_sentences))

    logger.info(f"Generating audio for {len(raw_sentences)} sentences...")
    audio_files = gen_audio(converted_sentences, ep_dir)
    audio_lengths = [_get_media_file_duration(filename) for filename in audio_files]
    merged_audio_file = merge_audio(audio_files)
    total_audio_length = sum(audio_lengths) + SILENCE_PADDING * (len(audio_lengths) - 1)
    for sentence, length in zip(raw_sentences, audio_lengths):
        short_sentence = sentence[:50] + "..." if len(sentence) > 50 else sentence
        logger.debug(f"  {short_sentence}: {length:.2f}s")
    logger.info(f"Total audio length: {total_audio_length:.2f}s")

    logger.info("Generating video prompts...")
    prompts = gen_video_prompts(summary, no_openai)
    prompts, prompt_lenghts = distribute_prompts(prompts, audio_lengths)
    for prompt, length in zip(prompts, prompt_lenghts):
        short_prompt = prompt[:50] + "..." if len(prompt) > 50 else prompt
        logger.debug(f"  {short_prompt}: {length} frames")

    logger.info("Generating video clips...")
    clip_dirs = gen_video_clips(prompts, prompt_lenghts, ep_dir)

    logger.info("Enhancing video clips...")
    enhanced_clip_dirs = enhance_video_clips(clip_dirs)
    merged_video_dir = merge_video_clips(enhanced_clip_dirs)

    logger.info("Merging audio and video and subtitles...")
    subtitles_file = create_subtitles(raw_sentences, audio_lengths, ep_dir)
    merged_video_file = merge_video_audio_subtitles(
        merged_video_dir, merged_audio_file, subtitles_file
    )
    final_video_file = prepend_intro_and_final_render(merged_video_file, title)

    process_duration = time.time() - process_start_time
    metadata_file = ep_dir / f"{ep_dir.name}.toml"
    _record_metadata(
        metadata_file,
        {
            "title": title,
            "summary": summary,
            "script": script,
            "prompts": prompts,
            "audio_duration": total_audio_length,
            "total_duration": _get_media_file_duration(final_video_file),
            "process_duration": process_duration,
        },
    )
    logger.success(f"Episode {ep_dir.name} completed in {process_duration/60:.1f} min")
    return final_video_file, metadata_file


def make_episodes(args):
    """Make episodes."""
    # Find the last episode index from the output directory
    episode_dirs = sorted(Path("output").glob("ep_*"))
    if episode_dirs:
        last_episode_idx = int(episode_dirs[-1].name.split("_")[1])
    else:
        last_episode_idx = -1  # Start from 0

    if args.redo:
        # Delete the last args.count episodes
        start_idx = max(0, last_episode_idx - args.count + 1)
    else:
        start_idx = last_episode_idx + 1
    for i in range(start_idx, start_idx + args.count):
        # Delete the episode directory if it exists
        episode_dir = Path(f"output/ep_{i:03d}")
        if episode_dir.exists():
            logger.debug(f"Deleting {episode_dir}")
            shutil.rmtree(episode_dir)
        episode_dir.mkdir(parents=True)
        episode_logger = logger.add(episode_dir / "log.txt")
        episode_files = make_episode(episode_dir, args.avoid, args.no_openai)

        logger.info("Uploading episode to Google Drive...")
        uploader = FolderUploader(os.environ["GOOGLE_DRIVE_FOLDER_ID"])
        for file in episode_files:
            uploader.upload(file)
        if not args.keep_intermediate:
            logger.debug("Deleting intermediate files...")
            shutil.rmtree(episode_dir / "clips")
            shutil.rmtree(episode_dir / "sentences")
        logger.remove(episode_logger)
    if args.when_done == "sleep":
        _pc_sleep()


if __name__ == "__main__":
    with logger.catch():
        make_episodes(_parse_args())
