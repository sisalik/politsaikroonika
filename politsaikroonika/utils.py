import contextlib
import subprocess
import time
from pathlib import Path

import toml

PYTHON_VENVS = {
    "Voice-Cloning-App": Path("Voice-Cloning-App/.venv/Scripts/python.exe"),
    "sd-webui-text2video": Path("sd-webui-text2video/.venv/Scripts/python.exe"),
}


def input_multiline(prompt="", terminators=()):
    """Get multiline input from the user."""
    print(prompt, end="")  # Same behavior as input()
    lines = []
    while True:
        line = input()
        lines.append(line)
        if line in terminators + ("",):  # Empty line or any of the terminators
            break
    return "\n".join(lines)


def get_git_revision():
    """Get the current git revision."""
    output = subprocess.run(
        "git describe --tags --always --dirty=-dev".split(),
        capture_output=True,
        text=True,
    )
    return output.stdout.strip()


def get_media_file_duration(media_file):
    """Get audio/video file duration in seconds."""
    output = subprocess.run(
        "ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1".split()
        + [str(media_file)],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(output.stdout)


def monitor_dir_file_count(dir, target_count, callback):
    """Monitor the number of files in a directory."""
    prev_count = 0
    while True:
        file_count = len(list(dir.iterdir()))
        callback(file_count - prev_count)
        if file_count >= target_count:
            break
        prev_count = file_count
        time.sleep(1)


def record_metadata(file, metadata):
    """Append metadata to a TOML file."""
    if Path(file).is_file():
        with open(file, encoding="utf-8") as f:
            data = toml.load(f)
    else:
        data = {}
    data.update(metadata)
    with open(file, "w", encoding="utf-8") as f:
        toml.dump(data, f, encoder=MultilinePreferringTomlEncoder())


@contextlib.contextmanager
def run_in_venv_and_monitor_output(venv, *commands):
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


def pc_sleep():
    """Put the computer to sleep."""
    subprocess.run("rundll32.exe powrprof.dll,SetSuspendState 0,1,0".split())


def _dump_str_prefer_multiline(v):
    """Dump a string to TOML, preferring the multiline string format."""
    multilines = v.split("\n")
    if len(multilines) > 1:
        return toml.encoder.unicode(
            '"""\n' + v.replace('"""', '\\"""').strip() + '\n"""'
        )
    else:
        return toml.encoder._dump_str(v)


class MultilinePreferringTomlEncoder(toml.encoder.TomlEncoder):
    """A TOML encoder that prefers the multiline string format if possible."""

    def __init__(self, _dict=dict, preserve=False):
        super(MultilinePreferringTomlEncoder, self).__init__(
            _dict=dict, preserve=preserve
        )
        self.dump_funcs[str] = _dump_str_prefer_multiline
