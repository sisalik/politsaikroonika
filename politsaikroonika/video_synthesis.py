import enum
import time
from pathlib import Path
from typing import Optional, Union

import cv2
import diffusers.utils
import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.models.attention_processor import AttnProcessor2_0
from tqdm import tqdm


class OutputFormat(enum.Enum):
    """Output format for the video."""

    MP4 = "mp4"
    PNG = "png"


class VideoGenPipeline:
    """A context manager for the Diffusion pipeline, with support for multiple jobs."""

    def __init__(
        self,
        model: str = "damo-vilab/text-to-video-ms-1.7b",
        low_vram: bool = False,
        output_format: OutputFormat = OutputFormat.PNG,
        progress_bar: Optional[tqdm] = None,
    ):
        self._output_format = output_format
        self._progress_bar = progress_bar

        self._queue = []
        # Initialize the pipeline
        self._pipe = DiffusionPipeline.from_pretrained(
            model, torch_dtype=torch.float16, variant="fp16"
        )
        self._pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self._pipe.scheduler.config
        )
        # Disable the built-in progress bar
        self._pipe.set_progress_bar_config(disable=True)
        # Enable some optimizations to lower the VRAM usage
        if low_vram:
            self._pipe.enable_sequential_cpu_offload()  # Large VRAM savings
        else:
            self._pipe.enable_model_cpu_offload()  # Medium VRAM savings
        self._pipe.enable_vae_slicing()
        self._pipe.enable_vae_tiling()
        self._pipe.unet.set_attn_processor(AttnProcessor2_0())
        # self._pipe.unet = torch.compile(self._pipe.unet)

    def push(self, **kwargs):
        """Push a new video generation job to the queue."""
        # Set some default values
        kwargs.setdefault("num_inference_steps", 50)
        kwargs.setdefault("num_frames", 16)
        kwargs.setdefault("seed", -1)  # -1 = random seed
        self._queue.append(kwargs)

    def process(self):
        """Start processing the pipeline queue."""
        # Set up the progress bar if it was provided
        if self._progress_bar is not None:
            total_frames = sum(kwargs["num_frames"] for kwargs in self._queue)
            self._progress_bar.reset(total=total_frames)
        # Process the queue
        out_paths = []
        frames_processed = 0
        for kwargs in self._queue:
            if self._progress_bar is not None:
                # The callback is called after each inference step, not after each
                # frame, so we calculate the equivalent frame number here and set the
                # progress bar to that value
                def callback(step, _timestep, _latents):
                    return self._set_progress_bar(
                        self._progress_bar,
                        round(
                            (step + 1)
                            / kwargs["num_inference_steps"]
                            * kwargs["num_frames"]
                            + frames_processed,
                            2,
                        ),
                    )

                kwargs.update(callback=callback)
            # Initialise the RNG with the seed, if provided
            if kwargs["seed"] != -1:
                generator = torch.Generator(device="cuda")
                generator.manual_seed(kwargs["seed"])
                kwargs.update(generator=generator)
            kwargs.pop("seed")  # Don't pass the seed to the pipeline
            # Get the output path from the kwargs but don't pass it to the pipeline
            out_path = kwargs.pop("out_path", f"vid_{time.time():.0f}")
            if self._output_format == OutputFormat.MP4:
                out_path += ".mp4"
            video_frames = self._pipe(**kwargs).frames
            frames_processed += len(video_frames)
            # Save the output
            if self._output_format == OutputFormat.MP4:
                Path(out_path).parent.mkdir(parents=True, exist_ok=True)
                video_path = diffusers.utils.export_to_video(video_frames, out_path)
                out_paths.append(video_path)
            elif self._output_format == OutputFormat.PNG:
                Path(out_path).mkdir(parents=True, exist_ok=True)
                for i, frame in enumerate(video_frames):
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(f"{out_path}/{i:06d}.png", frame_rgb)
                out_paths.append(out_path)
        self._queue = []
        return out_paths

    @staticmethod
    def _set_progress_bar(progress_bar: tqdm, n: Union[int, float]):
        """Set a tqdm progress bar to a given value."""
        progress_bar.n = n
        progress_bar.refresh()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # Free up the GPU memory
        del self._pipe
        torch.cuda.empty_cache()


if __name__ == "__main__":
    # Example usage
    from tqdm import tqdm

    with tqdm(unit="frame") as pbar, VideoGenPipeline(
        output_format=OutputFormat.MP4, progress_bar=pbar
    ) as pipeline:
        pipeline.push(
            prompt="An astronaut riding a horse.",
            num_inference_steps=25,
            num_frames=16,
            seed=1,
            out_path="astronaut.mp4",
        )
        pipeline.process()
