import enum
import time
from pathlib import Path

import cv2
import diffusers.utils
import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.models.attention_processor import AttnProcessor2_0


class OutputFormat(enum.Enum):
    """Output format for the video."""

    MP4 = "mp4"
    PNG = "png"


class VideoGenPipeline:
    """A context manager for the Diffusion pipeline, with support for multiple jobs."""

    def __init__(
        self,
        model="damo-vilab/text-to-video-ms-1.7b",
        low_vram=False,
        output_format=OutputFormat.PNG,
        callback=None,
    ):
        self._output_format = output_format
        self._callback = callback

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
        self._queue.append(kwargs)

    def process(self):
        """Start processing the pipeline queue."""
        out_paths = []
        for kwargs in self._queue:
            if self._callback is not None:
                # Pre-process the data passed to the callback to calculate the progress
                pre_callback = lambda step, timestep, latents: self._callback(
                    1 / kwargs["num_inference_steps"] * kwargs["num_frames"]
                )
                kwargs.update(callback=pre_callback)
            # Get the output path from the kwargs but don't pass it to the pipeline
            out_path = kwargs.pop("out_path", f"vid_{time.time():.0f}")
            video_frames = self._pipe(**kwargs).frames
            # Save the output
            Path(out_path).mkdir(parents=True, exist_ok=True)  # Ensure the dir exists
            if self._output_format == OutputFormat.MP4:
                video_path = diffusers.utils.export_to_video(
                    video_frames, f"{out_path}.mp4"
                )
                out_paths.append(video_path)
            elif self._output_format == OutputFormat.PNG:
                for i, frame in enumerate(video_frames):
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(f"{out_path}/{i:06d}.png", frame_rgb)
                out_paths.append(out_path)
        self._queue = []
        return out_paths

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # Free up the GPU memory
        del self._pipe
        torch.cuda.empty_cache()
