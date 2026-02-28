"""Tests for the Surivoice CLI module."""

from pathlib import Path

import click
import typer
from typer.testing import CliRunner

from surivoice import __version__
from surivoice.cli import app
from surivoice.errors import InputFileError


class TestCliHelp:
    """Test CLI help and version output."""

    def test_no_args_shows_help(self, cli_runner: CliRunner) -> None:
        """Running with no args should show help text (but exit 2 due to missing required opts)."""
        result = cli_runner.invoke(app, [])
        assert result.exit_code == 2
        
        click_app = typer.main.get_command(app)
        assert click_app.name in result.output

    def test_help_flag(self, cli_runner: CliRunner) -> None:
        """--help should show usage information dynamically."""
        result = cli_runner.invoke(app, ["--help"])
        assert result.exit_code == 0

        click_app = typer.main.get_command(app)
        assert click_app.name in result.output
        assert click_app.help in result.output

    def test_help_shows_all_options(self, cli_runner: CliRunner) -> None:
        """--help should show command options dynamically."""
        result = cli_runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        
        # Extract all registered options dynamically
        click_app = typer.main.get_command(app)
        expected_options = [
            opt.opts[0]
            for opt in click_app.params
            if isinstance(opt, click.Option) and opt.opts
        ]
        
        for option in expected_options:
            assert option in result.output, f"Expected option {option} missing in help"

    def test_version_flag(self, cli_runner: CliRunner) -> None:
        """--version should print the version string."""
        result = cli_runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert __version__ in result.output


class TestCliInputValidation:
    """Test CLI input file validation."""

    def test_missing_input_file(
        self, cli_runner: CliRunner, tmp_output_file: Path
    ) -> None:
        """Non-existent input file should fail with exit code 1."""
        result = cli_runner.invoke(
            app,
            ["-i", "/nonexistent/file.mp4", "-o", str(tmp_output_file)],
        )
        assert result.exit_code == 1
        assert InputFileError.NOT_FOUND.lower() in result.output.lower()

    def test_unsupported_format(
        self, cli_runner: CliRunner, tmp_path: Path, tmp_output_file: Path
    ) -> None:
        """Unsupported file extension should fail with exit code 1."""
        bad_file = tmp_path / "document.pdf"
        bad_file.write_bytes(b"fake pdf")

        result = cli_runner.invoke(
            app,
            ["-i", str(bad_file), "-o", str(tmp_output_file)],
        )
        assert result.exit_code == 1
        assert InputFileError.UNSUPPORTED_FORMAT.lower() in result.output.lower()

    def test_input_is_directory(
        self, cli_runner: CliRunner, tmp_path: Path, tmp_output_file: Path
    ) -> None:
        """Passing a directory as input should fail with exit code 1."""
        result = cli_runner.invoke(
            app,
            ["-i", str(tmp_path), "-o", str(tmp_output_file)],
        )
        assert result.exit_code == 1
        assert InputFileError.NOT_A_FILE.lower() in result.output.lower()


class TestCliRequiredOptions:
    """Test that required options are enforced."""

    def test_missing_input_option(self, cli_runner: CliRunner) -> None:
        """Missing --input should fail."""
        result = cli_runner.invoke(app, ["-o", "output.md"])
        assert result.exit_code != 0

    def test_missing_output_option(
        self, cli_runner: CliRunner, tmp_audio_file: Path
    ) -> None:
        """Missing --output should fail."""
        result = cli_runner.invoke(app, ["-i", str(tmp_audio_file)])
        assert result.exit_code != 0
