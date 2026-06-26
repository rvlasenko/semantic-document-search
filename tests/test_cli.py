import os
import subprocess
import sys
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def make_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    return env


def run_cli(*args: str) -> "subprocess.CompletedProcess[str]":
    return subprocess.run(
        [sys.executable, "-m", "semantic_search", *args],
        capture_output=True,
        text=True,
        env=make_env(),
    )


@pytest.fixture
def index_dir(tmp_path: Path) -> Path:
    index_path = tmp_path / "index"
    result = run_cli(
        "index",
        "--data-dir",
        str(FIXTURES_DIR),
        "--index-dir",
        str(index_path),
    )
    assert result.returncode == 0, result.stderr
    return index_path


def test_index_command_exits_successfully(tmp_path: Path) -> None:
    result = run_cli(
        "index",
        "--data-dir",
        str(FIXTURES_DIR),
        "--index-dir",
        str(tmp_path / "index"),
    )
    assert result.returncode == 0, result.stderr


def test_search_command_exits_successfully(index_dir: Path) -> None:
    result = run_cli(
        "search",
        "How does the refund policy work?",
        "--index-dir",
        str(index_dir),
    )
    assert result.returncode == 0, result.stderr


def test_search_command_prints_results(index_dir: Path) -> None:
    result = run_cli(
        "search",
        "How does the refund policy work?",
        "--index-dir",
        str(index_dir),
    )
    assert "#1" in result.stdout
    assert "score=" in result.stdout


def test_search_top_k_flag(index_dir: Path) -> None:
    result = run_cli(
        "search",
        "refund",
        "--top-k",
        "1",
        "--index-dir",
        str(index_dir),
    )
    assert result.returncode == 0, result.stderr
    assert "#1" in result.stdout
    assert "#2" not in result.stdout


def test_search_command_without_query_exits_with_error() -> None:
    result = run_cli("search")
    assert result.returncode != 0


def test_unknown_command_exits_with_error() -> None:
    result = run_cli("unknown")
    assert result.returncode != 0


def test_help_exits_successfully() -> None:
    result = run_cli("--help")
    assert result.returncode == 0
    assert "index" in result.stdout
    assert "search" in result.stdout
