<div align="center">

<h1>politsAIkroonika</h1>

[![Python version](https://img.shields.io/badge/python-v3.8-blue)]()
[![Python version](https://img.shields.io/badge/python-v3.9-blue)]()
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)

</div>

## About

This is the code behind the politsAIkroonika project on [Instagram](https://instagram.com/politsaikroonika) and [YouTube](https://www.youtube.com/@politsAIkroonika). With just a single command, you can produce a short video clip featuring a fictional crime news story in the style of a certain Estonian 90s TV show. The story, audio and video are all 100% AI-generated using various models. The Estonian text-to-speech model used for the news reporter's voice has been custom trained for maximum authenticity.

A brief overview of the process:

1. Generate story title, summary and script using OpenAI GPT-3.5 and GPT-4
2. Convert script to audio using [Voice Cloning App](https://github.com/BenAAndrew/Voice-Cloning-App) by [BenAAndrew](https://github.com/BenAAndrew)
3. Generate video clips to illustrate the story using [ModelScope Text-to-Video](https://huggingface.co/docs/diffusers/main/en/api/pipelines/text_to_video)
4. Enhance the video using [Topaz Video AI](https://www.topazlabs.com/topaz-video-ai) (optional but highly recommended - improve resolution and frame rate)
5. Merge video clips, audio and subtitles using [ffmpeg](https://ffmpeg.org/)
6. Upload to Google Drive (optional - for the convenience of sharing the clip)

## Installation

Installing the package and its dependencies is a bit more involved than usual due to the need to install and configure the Voice Cloning App and Topaz Video AI. The following instructions are for Windows, but should be easily adaptable to Linux.

### Prerequisites

- Python 3.8 or 3.9
- [Poetry](https://python-poetry.org/) (tested with 1.6.1)
- ffmpeg
- NVIDIA GPU with at least 8 GB of VRAM (tested on a GTX 1070)
- OpenAI account with API key (paid subscription required, but the cost is a few cents per episode)

Optional:

- Topaz Video AI (tested with version 3.2.0)  
  *Whilst this is paid software, it is currently the best available option for frame interpolation and upscaling. Open source options do exist (RIFE/CAIN/DAIN etc) but would require additional development to implement.*

### Installing the package

Clone the repository and install the dependencies using Poetry:

```bash
poetry install
```

### Voice-Cloning-App

[Voice Cloning App](https://github.com/BenAAndrew/Voice-Cloning-App) is used for the text-to-speech functionality and is executed under its own virtual environment. This is because it requires specific versions of various libraries that may conflict with the versions required by this package.

Follow the manual install instructions [here](https://github.com/sisalik/Voice-Cloning-App/blob/a3eba29cbe0bdad3ffaf873032d5518875df8037/install.md#manual-install-linux-windows), except install the requirements into a virtual environment under `/Voice-Cloning-App/.venv`:

```bash
cd Voice-Cloning-App
python -m venv .venv
.venv\Scripts\activate  # Or the Linux equivalent
pip install -r requirements.txt
```

### Environment variables

The code requires several environment variables to be configured. You may choose to set these in your system environment variables, or in a `.env` file in the root of the repository. For the latter option, you need to install the [Poetry dotenv plugin](https://github.com/mpeteuil/poetry-dotenv-plugin).

The following environment variables are required:

- **`OPENAI_API_KEY`** - get from your OpenAI account (instructions [here](https://platform.openai.com/docs/quickstart?context=python))
- The `ffmpeg` executable must be in your **`PATH`** variable

If you are using Topaz Video AI, the following environment variables are also required:

- **`TVAI_MODEL_DIR`** and **`TVAI_MODEL_DATA_DIR`** - set according to instructions [here](https://docs.topazlabs.com/video-ai/advanced-functions-in-topaz-video-ai/command-line-interface#environment-variables-c3d2)
- **`TVAI_FFMPEG`** - set to the path of the `ffmpeg` executable in your Topaz Video AI installation (e.g. `C:\Program Files\Topaz Labs LLC\Topaz Video AI\ffmpeg.exe`)

If you are using Google Drive, the following environment variable is also required:

- **`GOOGLE_DRIVE_FOLDER_ID`** - the ID of the Google Drive folder where the videos will be uploaded. This is a long string of letters and numbers that can be found in the URL of the folder in Google Drive.

### Models

Once everything has been installed, you will need to download and place the below models in the correct directories (relative to the Voice-Cloning-App directory). If the directories do not exist, create them.

- **Voice model** - download from [here](https://drive.google.com/file/d/10CrznSkb_h60GNXeOvLOXIboryZ1saaN/view?usp=sharing) and place in  `data/models/reporter`
- **Vocoder model** - download from [here](https://github.com/sisalik/Voice-Cloning-App/blob/a3eba29cbe0bdad3ffaf873032d5518875df8037/synthesis/synthesis.md), rename from `g_02500000` to `model.pt` and place in  `data/hifigan/vctk`
- **Vocoder model config file** - download from [here](https://drive.google.com/file/d/1pAB2kQunkDuv6W5fcJiQ0CY8xcJKB22e/view?usp=sharing) and place in `data/hifigan/vctk`
- **Alphabet file** - copy from `alphabets/Estonian.txt` to `data/languages/Estonian` and rename to `alphabet.txt`

The text-to-video model is automatically downloaded by the code.

For Topaz Video AI, if you have a fresh install, you may need to run the GUI first to download the required models. Simply load a video file and process it using the same models that the code uses:

- Apollo v8 (`apo-8`) - frame interpolation
- Theia Fine Tune Detail v3 (`thd-3`) - upscaling

## Usage

If everything has been installed correctly, you should be able to run the following command to generate a new episode:

```bash
poetry run python .\politsaikroonika\make_episode.py
```

The above command will generate a new episode using the default settings. You can customise the episode using various command line arguments. For example, to avoid the topics of animals, theft, stealing and robbery, and to include fireworks and "new year's celebration", you can run the following command:

```bash
poetry run python .\politsaikroonika\make_episode.py -v --interactive --avoid animals,theft,stealing,robbery --include fireworks --include "new year's celebration"
```

The `-v` flag is for verbose output, and the `--interactive` flag is for interactive mode, which will prompt you to confirm the generated text parts before proceeding, and lets you override them if you wish.

More information on the available command line arguments can be found by running:

```bash
poetry run python .\politsaikroonika\make_episode.py --help
```

## Training

If you are interested in training your own text-to-speech model, you can follow the instructions in the [Voice Cloning App](https://github.com/BenAAndrew/Voice-Cloning-App) repository. For reference, the training data used for the Estonian model included over 1000 sentences with a total duration of around 1.5 hours. The training took approximately 2 days on a GTX 1070.

To gather the training data, I processed all publicly available clips of the original TV show and extracted the audio track. Then, I transcribed the audio using [tekstiks.ee](https://tekstiks.ee/) (with a fair amount of manual corrections) and used `split_audio.py` and various scripts under `scripts` to split the audio into individual sentences. Background noise was removed using OpenVINO's [noise-suppression-poconetlike-0001](https://docs.openvino.ai/2022.3/omz_models_model_noise_suppression_poconetlike_0001.html) model. Finally, the audio was upsampled using [NU-Wave2](https://github.com/maum-ai/nuwave2).
