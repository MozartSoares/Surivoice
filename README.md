# Surivoice

**Local-first speech transcription with speaker diarization.**

Surivoice is an open-source tool that takes a video or audio file, transcribes speech to text using [Whisper](https://github.com/openai/whisper), identifies speakers using [pyannote.audio](https://github.com/pyannote/pyannote-audio), and outputs a structured Markdown transcript — all running locally on your machine.

## Features

- 🎙️ Speech-to-text transcription (via [faster-whisper](https://github.com/SYSTRAN/faster-whisper))
- 👥 Speaker diarization (via [pyannote.audio](https://github.com/pyannote/pyannote-audio))
- 📝 Structured Markdown output with timestamps
- 🔒 Fully local — no cloud APIs, your data stays on your machine
- ⚡ GPU-accelerated with CPU fallback
- 🖥️ GUI with dark/light theme — or use the powerful CLI

## Requirements

- Python 3.10+
- **OS**: Linux, macOS, or Windows (fully supported natively!)
- **FFmpeg** installed and added to your system `PATH`:
  - Ubuntu/Debian: `sudo apt install ffmpeg`
  - macOS: `brew install ffmpeg`
  - Windows: `winget install ffmpeg`
- A [Hugging Face](https://huggingface.co/) access token (for pyannote.audio models)

## Installation

Because Surivoice relies on heavy Machine Learning libraries (PyTorch, Faster-Whisper, Pyannote), you must install it with the `[ml]` extra flag to pull these dependencies.

**Recommended (via pipx for isolated environments):**
```bash
pipx install "surivoice[ml]"
```

**Alternative (via pip in a virtual environment):**
```bash
pip install "surivoice[ml]"
```

For GPU support, ensure you have CUDA-enabled PyTorch installed.

> **Note (Linux only):** The GUI requires `tkinter`, which may not be installed by default.
> Install it with: `sudo apt install python3-tk` (Ubuntu/Debian) or your distro's equivalent.
> On Windows and macOS, tkinter is included with the standard Python installer.

### Development

```bash
git clone https://github.com/MozartSoares/surivoice.git
cd surivoice
pip install -e ".[dev,ml]"
```

## Usage

### Authentication

To use pyannote.audio for speaker diarization, you need to provide a Hugging Face access token:

```bash
surivoice save-token YOUR_HF_TOKEN
```
Or you can use the `--hf-token` flag when transcribing, or export the `HF_TOKEN` environment variable.

### GUI

Launch the graphical interface for a visual, point-and-click experience:

```bash
surivoice gui
```

The GUI provides:
- File pickers for input audio/video and output Markdown
- Model and device selection from dropdowns
- Language and speaker count options
- HF token management in the **Settings** tab
- Dark / Light theme toggle (defaults to dark)
- Real-time status log during transcription

### CLI Transcription

```bash
surivoice transcribe -i meeting.mp4 -o transcript.md
```

### CLI Options

| Flag | Description | Default |
|------|-------------|------------|
| `-i, --input` | Input video/audio file | *required* |
| `-o, --output` | Output Markdown file | *required* |
| `-m, --model` | Whisper model size | `large-v3` |
| `-d, --device` | Compute device (`auto`, `cpu`, `cuda`) | `auto` |
| `--compute-type` | Quantization algorithm | `int8` |
| `-l, --language` | Language code (auto-detect if omitted) | `None` |
| `--hf-token` | Hugging Face access token (or set `HF_TOKEN` env var) | `None` |
| `-s, --speakers` | Exact speaker count hint | `None` |
| `-v, --version` | Show version and exit | |

## Supported Formats

**Video:** `.mp4`, `.mkv`, `.avi`, `.mov`, `.webm`
**Audio:** `.mp3`, `.wav`, `.flac`, `.ogg`, `.m4a`, `.aac`, `.wma`

## Credits

Created by [Mozart Soares](https://github.com/MozartSoares).

## License

[MIT](LICENSE)
