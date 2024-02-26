import argparse
import math
import os
import random
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

from loguru import logger
from tqdm import tqdm

import politsaikroonika.utils as utils
import politsaikroonika.text_synthesis as text_synthesis
from politsaikroonika.google_drive import FolderUploader
from politsaikroonika.video_synthesis import VideoGenPipeline

# Silence between sentences in seconds
SILENCE_PADDING = 0.3
# Maximum audio length in seconds, for the entire episode to fit within 60 seconds
AUDIO_LENGTH_MAX = 56.0
# Shared negative video generation prompt
NEGATIVE_PROMPT = "watermark, copyright, blurry, blood, gore, wounds"
# Width and height of the generated video (in pixels)
VIDEO_SIZE = 256
# Generated video frame rate
VIDEO_GEN_FPS = 8
# Output video frame rate
VIDEO_OUT_FPS = 24
# Minimum number of frames to generate for a video (otherwise the video is just noise)
VIDEO_FRAMES_MIN = 16
# Maximum number of frames to generate for a video (otherwise you run out of VRAM)
VIDEO_FRAMES_MAX = 28
# The reporter prompt seems to allow longer video to be generated without too many
# artifacts, so use a separate maximum for it
VIDEO_FRAMES_MAX_REPORTER = 48
# Social media caption template
SOCIAL_MEDIA_CAPTION_TEMPLATE = """
Osa x: {title}

100% AI-genereeritud krimiuudised 🤖👮 uus osa aeg-ajalt
Tehnoloogiad: GPT-3.5, ModelScope text2video, Tacotron 2 TTS, EstNLTK, ffmpeg, Python

#tehisintellekt #kuritegevus #uudised #politseikroonika #krimi #90ndad #satiir #naljakas #eesti #meem #ai #aiart #crime #news #deepfake #artificialintelligence #satire #funny #chatgpt #stablediffusion"""


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--count", type=int, default=1, help="Number of episodes")
    parser.add_argument(
        "-r", "--redo", action="store_true", help="Redo existing episodes"
    )
    parser.add_argument(
        "-i", "--interactive", action="store_true", help="Interactive mode"
    )
    parser.add_argument("-t", "--title", type=str, help="Episode title")
    parser.add_argument("-a", "--avoid", type=str, action="append", help="Avoid topics")
    parser.add_argument(
        "-l", "--include", type=str, action="append", help="Include topics"
    )
    parser.add_argument(
        "-k", "--keep_intermediate", action="store_true", help="Keep intermediate files"
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Logging verbosity (-v, -vv, etc)",
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
    if args.avoid and len(args.avoid) == 1 and "," in args.avoid[0]:
        args.avoid = args.avoid[0].split(",")
    # Same for "--include"
    if args.include and len(args.include) == 1 and "," in args.include[0]:
        args.include = args.include[0].split(",")
    # --interactive cannot be used together with --no-openai
    if args.interactive and args.no_openai:
        parser.error("--interactive cannot be used together with --no-openai")
    return args


def _setup_logging(verbosity):
    """Set up logging with the specified verbosity level."""
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


def _interactive_override_wrapper(
    interactive, output_name, func, args, multiline=False
):
    """Run a function, optionally prompting the user for input."""
    if not interactive:
        return func(*args)
    output_accepted = False
    # Loop until the user accepts the output
    while not output_accepted:
        output = func(*args)
        # Assemble an input prompt based on the output type
        if isinstance(output, list) or isinstance(output, tuple):
            input_prompt_extra = " (separated by semicolons)"
        else:
            input_prompt_extra = ""
        input_prompt = (
            f"  Y: accept this {output_name}\n"
            f"  N: regenerate the {output_name}\n"
            "  Q: quit\n"
            f"  Any other text to override the {output_name}{input_prompt_extra}: "
        )
        # Loop until the user provides valid input
        while True:
            print(f"Generated {output_name}:\n{output}\n")
            if multiline:
                response = utils.input_multiline(input_prompt, tuple("ynqYNQ")).strip()
            else:
                response = input(input_prompt).strip()
            print()
            if response.lower() == "y":
                logger.debug(f"User selected to accept the {output_name}")
                output_accepted = True
                break
            elif response.lower() == "n":
                logger.debug(f"User selected to re-generate the {output_name}")
                break
            elif response.lower() == "q":
                logger.debug("User selected to quit")
                sys.exit(0)
            elif not response:
                logger.error("No input provided")
                continue
            else:
                logger.debug(f"User selected to override the {output_name}")
                if isinstance(output, list) or isinstance(output, tuple):
                    output = [element.strip() for element in response.split(";")]
                else:
                    output = response
                output_accepted = True
                break
    return output


def _interactive_select_wrapper(
    interactive, output_name, gen_func, gen_args, select_func, multiline=False
):
    """Run a function to pick an item from a list, allowing the user to override."""
    if not interactive:
        return select_func(gen_func(*gen_args))
    output_accepted = False
    # Loop until the user accepts the output
    while not output_accepted:
        items = gen_func(*gen_args)
        selection = select_func(items)
        items_str = ""
        for i, item in enumerate(items):
            if i == selection:
                items_str += f" >{i + 1}. {item}\n"
            else:
                items_str += f"  {i + 1}. {item}\n"

        input_prompt = (
            f"Generated {output_name}s:\n{items_str}\n"
            f"  Y: accept {output_name} {selection + 1}\n"
            f"  N: regenerate the {output_name}\n"
            f"  1-{len(items)}: select a different {output_name}\n"
            "  Q: quit\n"
            f"  Any other text to override the {output_name}: "
        )
        # Loop until the user provides valid input
        while True:
            if multiline:
                terminators = tuple("ynqYNQ") + tuple(
                    str(n) for n in range(1, len(items) + 1)
                )
                response = utils.input_multiline(input_prompt, terminators).strip()
            else:
                response = input(input_prompt).strip()
            print()
            if response.lower() == "y":
                logger.debug(f"User selected to accept the {output_name}")
                output = items[selection]
                output_accepted = True
                break
            elif response.isnumeric():
                selection = int(response) - 1
                if selection < 0 or selection > len(items):
                    logger.error("Invalid selection")
                    continue
                logger.debug(f"User selected to re-select the {output_name}")
                output = items[int(response) - 1]
                output_accepted = True
                break
            elif response.lower() == "n":
                logger.debug(f"User selected to re-generate the {output_name}")
                break
            elif response.lower() == "q":
                logger.debug("User selected to quit")
                sys.exit(0)
            elif not response:
                logger.error("No input provided")
                continue
            else:
                logger.debug(f"User selected to override the {output_name}")
                output = response
                output_accepted = True
                break
    return output


def _seconds_to_frame_splits(
    seconds, frames_min=VIDEO_FRAMES_MIN, frames_max=VIDEO_FRAMES_MAX, variability=0.5
):
    """Convert seconds to video frames.

    Args:
        seconds: The duration of the video clip in seconds.
        variability: The amount of variability allowed in the frame splits. A value
            of 0.0 means that the splits will be the maximum length possible, while
            a value of 1.0 means that the splits will be the minimum length possible.
    """
    if seconds == 0:
        raise ValueError("Cannot convert 0 seconds to frames")
    frames = math.ceil(seconds * VIDEO_GEN_FPS)
    if frames <= frames_min:
        return [frames_min]
    elif frames > frames_max:
        # Randomly shuffle the split points, ensuring that each split between
        # frames_min and frames_max frames long
        splits = []
        avg_duration = frames_max - variability * (frames_max - frames_min)
        n_parts = math.ceil(frames / avg_duration)
        for i in range(n_parts - 1):
            remaining_frames = frames - sum(splits)
            # Lower bound, assuming that the remaining splits are all at the
            # maximum length
            lower_bound = max(
                frames_min,
                remaining_frames - frames_max * (n_parts - i - 1),
            )
            # Upper bound, leaving room for a minimum length split at the end
            upper_bound = min(frames_max, remaining_frames - frames_min)
            # If the lower bound is greater than the upper bound, then there won't be
            # enough frames left to split another time. Break and add the remaining
            # frames to the last split.
            if lower_bound > upper_bound:
                break
            splits.append(random.randint(lower_bound, upper_bound))
        splits.append(frames - sum(splits))
        random.shuffle(splits)  # For good measure
        return splits
    else:
        return [frames]


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

    with utils.run_in_venv_and_monitor_output(
        "Voice-Cloning-App",
        Path("synthesis/synthesize.py"),
        "--model_path",
        Path("data/models/reporter/checkpoint_30000"),
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
    # list element and replacing every second element with the silence file.
    audio_files = [audio_file for audio_file in audio_files for _ in range(2)]
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


def distribute_prompts(prompts, audio_lengths):
    """Calculate the length of each prompt based on the audio length."""
    # Handle opening and closing shots separately. Use 0 variability to ensure the clips
    # are the maximum possible length. Also, use a higher max frame count.
    opening_length = _seconds_to_frame_splits(
        audio_lengths[0], frames_max=VIDEO_FRAMES_MAX_REPORTER, variability=0.0
    )[0]
    closing_length = _seconds_to_frame_splits(
        audio_lengths[-1], frames_max=VIDEO_FRAMES_MAX_REPORTER, variability=0.0
    )[0]
    # Split the remaining duration among the other prompts
    total_audio_length = sum(audio_lengths) + SILENCE_PADDING * len(audio_lengths)
    remaining_length = (
        total_audio_length - (opening_length + closing_length) / VIDEO_GEN_FPS
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
    # Custom formatting for tqdm to only show 2 decimal places
    with tqdm(unit="frame") as pbar, VideoGenPipeline(progress_bar=pbar) as pipeline:
        # Add video generation tasks to the pipeline
        for idx, (prompt, length) in enumerate(zip(prompts, lengths)):
            out_path = (ep_dir / f"clips/clip_{idx+1:02d}").resolve()
            # Calculate the number of inference steps based on the length. The longer
            # the clip, the less inference steps we need to take to get a good result.
            steps = max(10, round(25 - (length - 16) * 10 / 8))
            pipeline.push(
                prompt=prompt,
                negative_prompt=NEGATIVE_PROMPT,
                num_frames=length,
                num_inference_steps=steps,
                guidance_scale=9.0,
                width=VIDEO_SIZE,
                height=VIDEO_SIZE,
                out_path=out_path,
            )
        # Process the entire pipeline in order
        out_paths = pipeline.process()
    return out_paths


def enhance_video(input_dir):
    """Interpolate and upscale video frames using Topaz Video AI."""
    output_dir = input_dir.parent / "enhanced"
    output_dir.mkdir(exist_ok=True)
    total_frames = len(list(input_dir.glob("*.png"))) * VIDEO_OUT_FPS // VIDEO_GEN_FPS
    # TODO: Thread this and run two instances of ffmpeg at the same time
    with tqdm(total=total_frames, unit="frame") as pbar:
        # In order to update the progress bar, monitor the output directory for new
        # files in a separate thread
        monitor_thread = threading.Thread(
            target=utils.monitor_dir_file_count,
            args=(output_dir, total_frames, pbar.update),
        )
        monitor_thread.start()
        subprocess.run(
            [
                os.getenv("TVAI_FFMPEG"),
                "-nostdin",
                "-y",
                "-framerate",
                str(VIDEO_OUT_FPS),
                "-start_number",
                "0",
                "-i",
                input_dir / "%06d.png",
                "-sws_flags",
                "spline+accurate_rnd+full_chroma_int",
                "-color_trc",
                "2",
                "-colorspace",
                "0",
                "-color_primaries",
                "2",
                "-filter_complex",
                "tvai_fi=model=apo-8:slowmo=3:rdt=0.01:device=0:vram=1:instances=0,tvai_up=model=thd-3:scale=0:w=1024:h=1024:noise=0:blur=0:compression=0:device=0:vram=1:instances=0,scale=w=1024:h=1024:flags=lanczos:threads=0",
                "-c:v",
                "png",
                "-pix_fmt",
                "rgb24",
                "-start_number",
                "0",
                output_dir / "%06d.png",
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        monitor_thread.join()
    return output_dir


def merge_video_clips(clip_dirs):
    """Merge the video clips (image sequences) into a single video."""
    output_dir = clip_dirs[0].parent / "merged"
    output_dir.mkdir(exist_ok=True)

    frame_idx = 0
    for clip_dir in clip_dirs:
        for png_file in clip_dir.glob("*.png"):
            new_filename = f"{frame_idx:06d}.png"
            shutil.copy(png_file, output_dir / new_filename)
            frame_idx += 1
    return output_dir


def merge_video_audio_subtitles(
    video_dir, audio_file, subtitles_file, frame_rate=VIDEO_OUT_FPS
):
    """Merge the audio and video files into a single video file and add subtitles."""
    # TODO: Combine this step with the final render to avoid re-encoding the video
    merged_file = video_dir.parent.parent / "clips" / "merged.mp4"
    # Escape characters that cause problems with the ffmpeg filter syntax
    subtitle_path = str(subtitles_file).replace("\\", "/\\").replace(":", "\\\\:")
    subprocess.run(
        [
            "ffmpeg",
            "-framerate",
            str(frame_rate),
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
    # Include the episode title in the output file name, removing punctuation
    safe_title = re.sub(r"[\.,:;%\"'?!]", "", title).replace(" ", ".")
    output_file = episode_dir / f"{episode_dir.name}_{safe_title}.mp4"
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


def make_episode(
    ep_dir,
    title=None,
    interactive=False,
    include_topics=None,
    avoid_topics=None,
    no_openai=False,
):
    """Make a single episode with a given index."""
    process_start_time = time.time()
    print()
    logger.info(f"Making episode {ep_dir.name}...")
    if title is None:
        logger.info("Generating episode title...")
        title = _interactive_select_wrapper(
            interactive,
            "title",
            text_synthesis.gen_titles,
            (include_topics, avoid_topics, no_openai),
            text_synthesis.select_title,
        )
    logger.info(f"Episode title: {title}")

    logger.info("Generating episode summary...")
    summary = _interactive_override_wrapper(
        interactive,
        "summary",
        text_synthesis.gen_summary,
        (title, include_topics, avoid_topics, no_openai),
        multiline=True,
    )
    logger.info(f"Episode summary:\n{summary}")

    logger.info("Generating episode script...")
    script = _interactive_override_wrapper(
        interactive,
        "script",
        text_synthesis.gen_script,
        (summary, no_openai),
        multiline=True,
    )
    logger.info(f"Episode script:\n{script}")
    raw_sentences = list(text_synthesis.split_sentences(script))
    converted_sentences = list(text_synthesis.convert_sentences(raw_sentences))

    logger.info("Generating video prompts...")
    prompts = _interactive_override_wrapper(
        interactive,
        "video prompts",
        text_synthesis.gen_video_prompts,
        (summary, no_openai),
    )

    logger.info(f"Generating audio for {len(raw_sentences)} sentences...")
    audio_files = gen_audio(converted_sentences, ep_dir)
    audio_lengths = [
        utils.get_media_file_duration(filename) for filename in audio_files
    ]
    merged_audio_file = merge_audio(audio_files)
    total_audio_length = sum(audio_lengths) + SILENCE_PADDING * len(audio_lengths)
    for sentence, length in zip(raw_sentences, audio_lengths):
        short_sentence = sentence[:50] + "..." if len(sentence) > 50 else sentence
        logger.debug(f"  {short_sentence}: {length:.2f}s")
    logger.info(f"Total audio length: {total_audio_length:.2f}s")
    if total_audio_length > AUDIO_LENGTH_MAX:
        logger.warning(f"Total audio length exceeds maximum of {AUDIO_LENGTH_MAX:.2f}s")

    prompts, prompt_lenghts = distribute_prompts(prompts, audio_lengths)
    for prompt, length in zip(prompts, prompt_lenghts):
        short_prompt = prompt[:50] + "..." if len(prompt) > 50 else prompt
        logger.debug(f"  {short_prompt}: {length} frames")

    logger.info("Generating video clips...")
    clip_dirs = gen_video_clips(prompts, prompt_lenghts, ep_dir)
    merged_video_dir = merge_video_clips(clip_dirs)

    if all(
        var in os.environ
        for var in ("TVAI_MODEL_DIR", "TVAI_MODEL_DATA_DIR", "TVAI_FFMPEG")
    ):
        logger.info("Enhancing video clips...")
        enhanced_video_dir = enhance_video(merged_video_dir)
        frame_rate = VIDEO_OUT_FPS
    else:
        logger.warning(
            "Skipping video enhancement because the environment variables "
            "TVAI_MODEL_DIR, TVAI_MODEL_DATA_DIR and TVAI_FFMPEG are not set"
        )
        enhanced_video_dir = merged_video_dir
        frame_rate = VIDEO_GEN_FPS

    logger.info("Merging audio and video and subtitles...")
    subtitles_file = create_subtitles(raw_sentences, audio_lengths, ep_dir)
    merged_video_file = merge_video_audio_subtitles(
        enhanced_video_dir, merged_audio_file, subtitles_file, frame_rate
    )
    final_video_file = prepend_intro_and_final_render(merged_video_file, title)

    process_duration = time.time() - process_start_time
    metadata_file = ep_dir / f"{ep_dir.name}.toml"
    utils.record_metadata(
        metadata_file,
        {
            "title": title,
            "summary": summary,
            "script": script,
            "prompts": prompts,
            "caption": SOCIAL_MEDIA_CAPTION_TEMPLATE.format(title=title).strip(),
            "audio_duration": total_audio_length,
            "total_duration": utils.get_media_file_duration(final_video_file),
            "process_duration": process_duration,
            "git_commit": utils.get_git_revision(),
        },
    )
    logger.success(f"Episode {ep_dir.name} completed in {process_duration/60:.1f} min")
    return final_video_file, metadata_file


def make_episodes(args):
    """Make episodes."""
    _setup_logging(args.verbose)
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
        episode_files = make_episode(
            episode_dir,
            args.title,
            args.interactive,
            args.include,
            args.avoid,
            args.no_openai,
        )

        if "GOOGLE_DRIVE_FOLDER_ID" in os.environ:
            logger.info("Uploading episode to Google Drive...")
            uploader = FolderUploader(os.environ["GOOGLE_DRIVE_FOLDER_ID"])
            for file in episode_files:
                uploader.upload(file)
        else:
            logger.warning(
                "Skipping upload to Google Drive because the environment variable "
                "GOOGLE_DRIVE_FOLDER_ID is not set"
            )

        if not args.keep_intermediate:
            logger.debug("Deleting intermediate files...")
            shutil.rmtree(episode_dir / "clips")
            shutil.rmtree(episode_dir / "sentences")
        logger.remove(episode_logger)
    if args.when_done == "sleep":
        utils.pc_sleep()


if __name__ == "__main__":
    with logger.catch():
        make_episodes(_parse_args())
