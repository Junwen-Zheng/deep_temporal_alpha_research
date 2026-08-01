from pathlib import Path

import deep_alpha


def test_package_imports() -> None:
    assert deep_alpha.__name__ == "deep_alpha"


def test_expected_directories_exist() -> None:
    expected = [
        Path("configs"),
        Path("docs"),
        Path("reports"),
        Path("src/deep_alpha/data"),
        Path("src/deep_alpha/models"),
        Path("src/deep_alpha/training"),
        Path("src/deep_alpha/evaluation"),
        Path("tests"),
    ]

    for path in expected:
        assert path.exists(), f"Missing expected path: {path}"
