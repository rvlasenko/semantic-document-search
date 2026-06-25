import subprocess
import sys


def test_index_command_exits_successfully() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "semantic_search", "index"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0


def test_search_command_exits_successfully() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "semantic_search",
            "search",
            "How does the refund policy work?",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0


def test_search_command_prints_results() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "semantic_search",
            "search",
            "How does the refund policy work?",
        ],
        capture_output=True,
        text=True,
    )
    assert "#1" in result.stdout
    assert "score=" in result.stdout


def test_search_command_without_query_exits_with_error() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "semantic_search", "search"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "Usage:" in result.stdout


def test_unknown_command_exits_with_error() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "semantic_search", "unknown"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
