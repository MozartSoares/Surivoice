# Surivoice

**Local-first speech transcription with speaker diarization.**

Surivoice is an open-source CLI tool that takes a video or audio file, transcribes speech to text using [Whisper](https://github.com/openai/whisper), identifies speakers using [pyannote.audio](https://github.com/pyannote/pyannote-audio), and outputs a structured Markdown transcript — all running locally on your machine.

## Features

- 🎙️ Speech-to-text transcription (via [faster-whisper](https://github.com/SYSTRAN/faster-whisper))
- 👥 Speaker diarization (via [pyannote.audio](https://github.com/pyannote/pyannote-audio))
- 📝 Structured Markdown output with timestamps
- 🔒 Fully local — no cloud APIs, your data stays on your machine
- ⚡ GPU-accelerated with CPU fallback

## Requirements

- Python 3.10+
- [FFmpeg](https://ffmpeg.org/) installed and on `PATH`
- A [Hugging Face](https://huggingface.co/) access token (for pyannote.audio models)

## Installation

```bash
pip install surivoice
```

For GPU support, ensure you have CUDA-enabled PyTorch installed.

### Development

```bash
git clone https://github.com/your-org/surivoice.git
cd surivoice
pip install -e ".[dev,ml]"
```

## Usage

```bash
surivoice transcribe -i meeting.mp4 -o transcript.md
```

### Options

| Flag | Description | Default |
|------|-------------|---------|
| `-i, --input` | Input video/audio file | *required* |
| `-o, --output` | Output Markdown file | *required* |
| `-m, --model` | Whisper model size | `medium` |
| `-d, --device` | Compute device (`auto`, `cpu`, `cuda`) | `auto` |
| `-l, --language` | Language code (auto-detect if omitted) | `None` |
| `--hf-token` | Hugging Face access token (or set `HF_TOKEN` env var) | `None` |
| `--min-speakers` | Minimum speaker count hint | `None` |
| `--max-speakers` | Maximum speaker count hint | `None` |
| `-v, --version` | Show version and exit | |

## Supported Formats

**Video:** `.mp4`, `.mkv`, `.avi`, `.mov`, `.webm`
**Audio:** `.mp3`, `.wav`, `.flac`, `.ogg`, `.m4a`, `.aac`, `.wma`

## License

[MIT](LICENSE)
