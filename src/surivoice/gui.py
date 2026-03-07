"""Surivoice GUI — Simple Tkinter transcription interface.

A tabbed window providing:
  • **Transcribe** tab — file pickers, model/device selectors, and a log area.
  • **Settings** tab  — Hugging Face token management and theme toggle.

The pipeline runs in a background thread so the UI stays responsive.
"""

import logging
import threading
import tkinter as tk
from collections.abc import Callable, Sequence
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from surivoice import __version__
from surivoice.audio.validators import (
    SUPPORTED_EXTENSIONS,
    check_ffmpeg,
    validate_input_file,
)
from surivoice.config import DeviceType, PipelineConfig, WhisperModel
from surivoice.diarization.validators import (
    resolve_hf_token,
)
from surivoice.diarization.validators import (
    save_token as save_token_to_disk,
)
from surivoice.errors import SurivoiceError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_filetypes() -> list[tuple[str, str]]:
    """Build a list of file-type filters for the file picker dialog."""
    video_exts: list[str] = []
    audio_exts: list[str] = []
    video_set = {".mp4", ".mkv", ".avi", ".mov", ".webm"}
    for ext in sorted(SUPPORTED_EXTENSIONS):
        if ext in video_set:
            video_exts.append(f"*{ext}")
        else:
            audio_exts.append(f"*{ext}")

    all_exts = [f"*{e}" for e in sorted(SUPPORTED_EXTENSIONS)]
    return [
        ("All supported", " ".join(all_exts)),
        ("Video files", " ".join(video_exts)),
        ("Audio files", " ".join(audio_exts)),
        ("All files", "*.*"),
    ]


# ---------------------------------------------------------------------------
# Theme configuration
# ---------------------------------------------------------------------------

_DARK = {
    "bg": "#1e1e2e",
    "fg": "#cdd6f4",
    "bg_alt": "#313244",
    "accent": "#89b4fa",
    "accent_hover": "#74c7ec",
    "border": "#45475a",
    "input_bg": "#313244",
    "input_fg": "#cdd6f4",
    "btn_bg": "#89b4fa",
    "btn_fg": "#1e1e2e",
    "success": "#a6e3a1",
    "error": "#f38ba8",
    "log_bg": "#11111b",
    "log_fg": "#a6adc8",
    "tab_bg": "#181825",
    "tab_selected": "#1e1e2e",
}

_LIGHT = {
    "bg": "#eff1f5",
    "fg": "#4c4f69",
    "bg_alt": "#e6e9ef",
    "accent": "#1e66f5",
    "accent_hover": "#2a6ef5",
    "border": "#ccd0da",
    "input_bg": "#ffffff",
    "input_fg": "#4c4f69",
    "btn_bg": "#1e66f5",
    "btn_fg": "#ffffff",
    "success": "#40a02b",
    "error": "#d20f39",
    "log_bg": "#ffffff",
    "log_fg": "#5c5f77",
    "tab_bg": "#dce0e8",
    "tab_selected": "#eff1f5",
}


def _apply_theme(root: tk.Tk, colors: dict[str, str]) -> None:
    """Apply a colour palette to the ttk style and the root window."""
    style = ttk.Style(root)
    style.theme_use("clam")

    root.configure(bg=colors["bg"])

    # General
    style.configure(".", background=colors["bg"], foreground=colors["fg"])
    style.configure("TFrame", background=colors["bg"])
    style.configure(
        "TLabelframe",
        background=colors["bg"],
        foreground=colors["fg"],
        bordercolor=colors["border"],
    )
    style.configure(
        "TLabelframe.Label",
        background=colors["bg"],
        foreground=colors["accent"],
        font=("", 10, "bold"),
    )
    style.configure("TLabel", background=colors["bg"], foreground=colors["fg"])

    # Notebook (tabs)
    style.configure(
        "TNotebook",
        background=colors["tab_bg"],
        bordercolor=colors["border"],
    )
    style.configure(
        "TNotebook.Tab",
        background=colors["tab_bg"],
        foreground=colors["fg"],
        padding=[12, 4],
    )
    style.map(
        "TNotebook.Tab",
        background=[("selected", colors["tab_selected"])],
        foreground=[("selected", colors["accent"])],
    )

    # Entry / Combobox / Spinbox
    style.configure(
        "TEntry",
        fieldbackground=colors["input_bg"],
        foreground=colors["input_fg"],
        bordercolor=colors["border"],
        insertcolor=colors["fg"],
    )
    style.configure(
        "TCombobox",
        fieldbackground=colors["input_bg"],
        foreground=colors["input_fg"],
        bordercolor=colors["border"],
        arrowcolor=colors["fg"],
    )
    style.map(
        "TCombobox",
        fieldbackground=[("readonly", colors["input_bg"])],
        foreground=[("readonly", colors["input_fg"])],
    )
    style.configure(
        "TSpinbox",
        fieldbackground=colors["input_bg"],
        foreground=colors["input_fg"],
        bordercolor=colors["border"],
        arrowcolor=colors["fg"],
    )

    # Buttons
    style.configure(
        "Accent.TButton",
        background=colors["btn_bg"],
        foreground=colors["btn_fg"],
        bordercolor=colors["btn_bg"],
        font=("", 10, "bold"),
        padding=[16, 6],
    )
    style.map(
        "Accent.TButton",
        background=[
            ("active", colors["accent_hover"]),
            ("disabled", colors["border"]),
        ],
    )
    style.configure(
        "TButton",
        background=colors["bg_alt"],
        foreground=colors["fg"],
        bordercolor=colors["border"],
        padding=[8, 4],
    )
    style.map(
        "TButton",
        background=[("active", colors["border"])],
    )


# ---------------------------------------------------------------------------
# Custom logging handler → Tkinter Text widget
# ---------------------------------------------------------------------------


class _TextHandler(logging.Handler):
    """Route log records into a Tkinter ``Text`` widget (thread-safe)."""

    def __init__(self, text_widget: tk.Text) -> None:
        super().__init__()
        self._widget = text_widget

    def emit(self, record: logging.LogRecord) -> None:
        msg = self.format(record) + "\n"
        self._widget.after(0, self._append, msg)

    def _append(self, msg: str) -> None:
        self._widget.configure(state=tk.NORMAL)
        self._widget.insert(tk.END, msg)
        self._widget.see(tk.END)
        self._widget.configure(state=tk.DISABLED)


# ---------------------------------------------------------------------------
# Main application window
# ---------------------------------------------------------------------------

_PAD = 10


class SurivoiceApp:
    """Tkinter application wrapping the Surivoice pipeline."""

    def __init__(self, root: tk.Tk) -> None:
        self._root = root
        self._root.title(f"Surivoice v{__version__}")
        self._root.minsize(640, 540)
        self._root.resizable(True, True)

        # Theme state (default: dark)
        self._is_dark = True

        # StringVars for form fields
        self._input_var = tk.StringVar()
        self._output_var = tk.StringVar()
        self._model_var = tk.StringVar(value=WhisperModel.LARGE_V3.value)
        self._device_var = tk.StringVar(value=DeviceType.AUTO.value)
        self._language_var = tk.StringVar()
        self._speakers_var = tk.StringVar()
        self._token_var = tk.StringVar()

        # Pre-fill HF token from env / saved file
        existing_token = resolve_hf_token(None)
        if existing_token:
            self._token_var.set(existing_token)

        # Apply initial theme
        _apply_theme(self._root, _DARK)

        self._build_ui()

    # ------------------------------------------------------------------ UI

    def _build_ui(self) -> None:
        """Construct all widgets."""
        # Notebook (tabs)
        self._notebook = ttk.Notebook(self._root)
        self._notebook.pack(fill=tk.BOTH, expand=True, padx=_PAD, pady=(_PAD, 0))

        # --- Transcribe tab ---
        transcribe_frame = ttk.Frame(self._notebook, padding=_PAD)
        self._notebook.add(transcribe_frame, text="  Transcribe  ")
        self._build_transcribe_tab(transcribe_frame)

        # --- Settings tab ---
        settings_frame = ttk.Frame(self._notebook, padding=_PAD)
        self._notebook.add(settings_frame, text="  Settings  ")
        self._build_settings_tab(settings_frame)

    def _build_transcribe_tab(self, parent: ttk.Frame) -> None:
        """Build the main transcription form."""
        # --- File selection ---
        file_frame = ttk.LabelFrame(parent, text="Files", padding=_PAD)
        file_frame.pack(fill=tk.X, pady=(0, _PAD))

        self._add_file_row(
            file_frame, "Input file:", self._input_var, self._browse_input, row=0
        )
        self._add_file_row(
            file_frame, "Output file:", self._output_var, self._browse_output, row=1
        )

        # --- Options ---
        opts_frame = ttk.LabelFrame(parent, text="Options", padding=_PAD)
        opts_frame.pack(fill=tk.X, pady=(0, _PAD))

        model_values = [m.value for m in WhisperModel]
        device_values = [d.value for d in DeviceType]

        row = 0
        row = self._add_combo_row(
            opts_frame, "Model:", self._model_var, model_values, row
        )
        row = self._add_combo_row(
            opts_frame, "Device:", self._device_var, device_values, row
        )

        # Language
        ttk.Label(opts_frame, text="Language:").grid(
            row=row, column=0, sticky=tk.W, pady=4
        )
        ttk.Entry(opts_frame, textvariable=self._language_var, width=10).grid(
            row=row, column=1, sticky=tk.W, pady=4, padx=(4, 0)
        )
        ttk.Label(opts_frame, text="(e.g. en, pt — leave empty for auto)").grid(
            row=row, column=2, sticky=tk.W, padx=(8, 0)
        )
        row += 1

        # Speakers
        ttk.Label(opts_frame, text="Speakers:").grid(
            row=row, column=0, sticky=tk.W, pady=4
        )
        speakers_spin = ttk.Spinbox(
            opts_frame,
            from_=1,
            to=50,
            textvariable=self._speakers_var,
            width=5,
        )
        speakers_spin.grid(row=row, column=1, sticky=tk.W, pady=4, padx=(4, 0))
        speakers_spin.delete(0, tk.END)  # start blank (= auto)
        ttk.Label(opts_frame, text="(leave empty for auto)").grid(
            row=row, column=2, sticky=tk.W, padx=(8, 0)
        )

        opts_frame.columnconfigure(2, weight=1)

        # --- Action button ---
        btn_frame = ttk.Frame(parent)
        btn_frame.pack(fill=tk.X, pady=(0, _PAD))

        self._transcribe_btn = ttk.Button(
            btn_frame,
            text="⚡ Transcribe",
            command=self._on_transcribe,
            style="Accent.TButton",
        )
        self._transcribe_btn.pack(side=tk.LEFT)

        # --- Log area ---
        log_frame = ttk.LabelFrame(parent, text="Status", padding=_PAD)
        log_frame.pack(fill=tk.BOTH, expand=True)

        colors = _DARK if self._is_dark else _LIGHT
        self._log_text = tk.Text(
            log_frame,
            height=8,
            state=tk.DISABLED,
            wrap=tk.WORD,
            bg=colors["log_bg"],
            fg=colors["log_fg"],
            insertbackground=colors["fg"],
            relief=tk.FLAT,
            font=("monospace", 9),
        )
        scrollbar = ttk.Scrollbar(
            log_frame, orient=tk.VERTICAL, command=self._log_text.yview
        )
        self._log_text.configure(yscrollcommand=scrollbar.set)
        self._log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Attach logging handler
        handler = _TextHandler(self._log_text)
        handler.setFormatter(
            logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")
        )
        logging.getLogger("surivoice").addHandler(handler)
        logging.getLogger("surivoice").setLevel(logging.INFO)

    def _build_settings_tab(self, parent: ttk.Frame) -> None:
        """Build the settings panel (token + theme)."""
        # --- HF Token ---
        token_frame = ttk.LabelFrame(
            parent, text="Hugging Face Token", padding=_PAD
        )
        token_frame.pack(fill=tk.X, pady=(0, _PAD))

        ttk.Label(
            token_frame,
            text="Required for speaker diarization (pyannote.audio).",
        ).grid(row=0, column=0, columnspan=3, sticky=tk.W, pady=(0, 8))

        ttk.Label(token_frame, text="Token:").grid(
            row=1, column=0, sticky=tk.W, pady=4
        )
        ttk.Entry(
            token_frame, textvariable=self._token_var, width=45, show="•"
        ).grid(row=1, column=1, sticky=tk.EW, padx=4, pady=4)

        ttk.Button(
            token_frame,
            text="Save",
            command=self._on_save_token,
            style="Accent.TButton",
        ).grid(row=1, column=2, pady=4, padx=(4, 0))

        token_frame.columnconfigure(1, weight=1)

        ttk.Label(
            token_frame,
            text=(
                "You can also set the HF_TOKEN environment variable "
                "or run: surivoice save-token YOUR_TOKEN"
            ),
        ).grid(row=2, column=0, columnspan=3, sticky=tk.W, pady=(4, 0))

        # --- Appearance ---
        theme_frame = ttk.LabelFrame(parent, text="Appearance", padding=_PAD)
        theme_frame.pack(fill=tk.X, pady=(0, _PAD))

        self._theme_btn = ttk.Button(
            theme_frame,
            text="☀️  Switch to Light Theme",
            command=self._toggle_theme,
        )
        self._theme_btn.pack(anchor=tk.W)

    # ------------------------------------------------------------------ layout helpers

    @staticmethod
    def _add_file_row(
        parent: ttk.LabelFrame,
        label: str,
        var: tk.StringVar,
        browse_cmd: Callable[[], object],
        *,
        row: int,
    ) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky=tk.W, pady=4)
        ttk.Entry(parent, textvariable=var).grid(
            row=row, column=1, sticky=tk.EW, padx=4, pady=4
        )
        ttk.Button(parent, text="Browse…", command=browse_cmd).grid(
            row=row, column=2, pady=4
        )
        parent.columnconfigure(1, weight=1)

    @staticmethod
    def _add_combo_row(
        parent: ttk.LabelFrame,
        label: str,
        var: tk.StringVar,
        values: Sequence[str],
        row: int,
    ) -> int:
        ttk.Label(parent, text=label).grid(
            row=row, column=0, sticky=tk.W, pady=4
        )
        ttk.Combobox(
            parent, textvariable=var, values=list(values), state="readonly"
        ).grid(row=row, column=1, sticky=tk.W, pady=4, padx=(4, 0))
        return row + 1

    # ------------------------------------------------------------------ browse

    def _browse_input(self) -> None:
        path = filedialog.askopenfilename(
            title="Select audio/video file",
            filetypes=_build_filetypes(),
        )
        if path:
            self._input_var.set(path)

    def _browse_output(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Save transcript as",
            defaultextension=".md",
            filetypes=[("Markdown", "*.md"), ("All files", "*.*")],
        )
        if path:
            self._output_var.set(path)

    # ------------------------------------------------------------------ settings actions

    def _on_save_token(self) -> None:
        """Persist the token entered in the Settings tab."""
        token = self._token_var.get().strip()
        if not token:
            messagebox.showwarning(
                "No Token", "Please paste your Hugging Face token first."
            )
            return
        saved_path = save_token_to_disk(token)
        messagebox.showinfo("Saved", f"Token saved to {saved_path}")

    def _toggle_theme(self) -> None:
        """Switch between dark and light themes."""
        self._is_dark = not self._is_dark
        colors = _DARK if self._is_dark else _LIGHT
        _apply_theme(self._root, colors)

        # Update the log Text widget (not managed by ttk)
        self._log_text.configure(
            bg=colors["log_bg"],
            fg=colors["log_fg"],
            insertbackground=colors["fg"],
        )

        # Update button label
        if self._is_dark:
            self._theme_btn.configure(text="☀️  Switch to Light Theme")
        else:
            self._theme_btn.configure(text="🌙  Switch to Dark Theme")

    # ------------------------------------------------------------------ log

    def _log(self, msg: str) -> None:
        """Append a message to the status text widget (main-thread safe)."""
        self._log_text.configure(state=tk.NORMAL)
        self._log_text.insert(tk.END, msg + "\n")
        self._log_text.see(tk.END)
        self._log_text.configure(state=tk.DISABLED)

    def _log_from_thread(self, msg: str) -> None:
        """Schedule a log append from a background thread."""
        self._root.after(0, self._log, msg)

    # ------------------------------------------------------------------ transcribe

    def _on_transcribe(self) -> None:
        """Validate inputs, build config, and launch the pipeline in a thread."""
        # --- Validate input file ---
        input_path_str = self._input_var.get().strip()
        if not input_path_str:
            messagebox.showerror("Error", "Please select an input file.")
            return
        input_path = Path(input_path_str)
        file_error = validate_input_file(input_path)
        if file_error is not None:
            messagebox.showerror("Input Error", file_error)
            return

        # --- Validate output file ---
        output_path_str = self._output_var.get().strip()
        if not output_path_str:
            messagebox.showerror("Error", "Please specify an output file path.")
            return
        output_path = Path(output_path_str)

        # --- FFmpeg check ---
        ffmpeg_error = check_ffmpeg()
        if ffmpeg_error is not None:
            messagebox.showerror("FFmpeg Error", ffmpeg_error)
            return

        # --- HF Token ---
        token_str = self._token_var.get().strip() or None
        resolved_token = resolve_hf_token(token_str)
        if resolved_token is None:
            messagebox.showerror(
                "Token Error",
                "Hugging Face token is required for speaker diarization.\n\n"
                "Go to the Settings tab to enter and save your token.",
            )
            return

        # --- Speakers ---
        speakers_str = self._speakers_var.get().strip()
        num_speakers: int | None = None
        if speakers_str:
            try:
                num_speakers = int(speakers_str)
                if num_speakers < 1:
                    raise ValueError("Must be >= 1")
            except ValueError:
                messagebox.showerror(
                    "Speakers Error",
                    "Speaker count must be a positive integer.",
                )
                return

        # --- Language ---
        language: str | None = self._language_var.get().strip() or None

        # --- Build config ---
        config = PipelineConfig(
            model=WhisperModel(self._model_var.get()),
            device=DeviceType(self._device_var.get()),
            language=language,
            hf_token=resolved_token,
            num_speakers=num_speakers,
        )

        # --- Disable button & run ---
        self._transcribe_btn.configure(state=tk.DISABLED)
        self._log(
            f"Starting transcription — model={config.model.value}, "
            f"device={config.device.value}"
        )
        self._log(f"  Input:  {input_path}")
        self._log(f"  Output: {output_path}")

        thread = threading.Thread(
            target=self._run_pipeline,
            args=(input_path, output_path, config),
            daemon=True,
        )
        thread.start()

    def _run_pipeline(
        self,
        input_path: Path,
        output_path: Path,
        config: PipelineConfig,
    ) -> None:
        """Execute the pipeline in a background thread."""
        try:
            from surivoice.pipeline import run

            result = run(input_path, output_path, config)
            self._log_from_thread(
                f"✅ Done! {len(result.segments)} segments, "
                f"{result.speakers_count} speakers, "
                f"language={result.detected_language}"
            )
            self._root.after(
                0,
                lambda: messagebox.showinfo(
                    "Done",
                    f"Transcription complete!\n"
                    f"{len(result.segments)} segments, "
                    f"{result.speakers_count} speakers.\n\n"
                    f"Saved to: {output_path}",
                ),
            )
        except SurivoiceError as exc:
            error_msg = str(exc)
            self._log_from_thread(f"❌ Error: {error_msg}")
            self._root.after(
                0, lambda m: messagebox.showerror("Pipeline Error", m), error_msg
            )
        except Exception as exc:
            error_msg = str(exc)
            self._log_from_thread(f"❌ Unexpected error: {error_msg}")
            self._root.after(
                0,
                lambda m: messagebox.showerror("Unexpected Error", m),
                error_msg,
            )
        finally:
            self._root.after(
                0, lambda: self._transcribe_btn.configure(state=tk.NORMAL)
            )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Launch the Surivoice GUI."""
    root = tk.Tk()
    SurivoiceApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
