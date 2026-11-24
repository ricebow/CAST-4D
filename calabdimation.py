"""Compatibility entry point for the crack skeleton analysis.

The implementation is maintained in ``skeleton.py`` to avoid duplicated code.
"""

from pathlib import Path

from skeleton import run_analysis


def main() -> None:
    input_dir = Path(__file__).resolve().parent / "序列图片"
    run_analysis(input_dir)


if __name__ == "__main__":
    main()
