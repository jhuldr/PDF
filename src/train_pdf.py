"""Unified training entry point for the PDF method."""
import argparse
import runpy
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Train or test PDF morph and intensity stages.",
        add_help=True,
    )
    parser.add_argument(
        "--stage",
        choices=("morph", "intensity"),
        default="morph",
        help="PDF stage to run. Use morph for lesion shape evolution or intensity for texture editing.",
    )
    args, remaining = parser.parse_known_args()

    script_name = {
        "morph": "pdf_stages/morph.py",
        "intensity": "pdf_stages/intensity.py",
    }[args.stage]
    script_path = Path(__file__).resolve().parent / script_name

    sys.argv = [str(script_path), *remaining]
    runpy.run_path(str(script_path), run_name="__main__")


if __name__ == "__main__":
    main()
