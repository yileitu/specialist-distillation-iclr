#!/usr/bin/env python3
"""Download the NLTK resources used by the postprocessing scripts."""

import argparse
import os

import nltk


def setup_nltk_data(custom_dir: str = "./nltk_data") -> None:
    """Configure and download NLTK data to the requested directory."""
    os.makedirs(custom_dir, exist_ok=True)
    if custom_dir not in nltk.data.path:
        nltk.data.path.insert(0, custom_dir)

    resources = ["punkt", "punkt_tab", "wordnet", "omw-1.4"]
    for resource in resources:
        lookup_path = f"tokenizers/{resource}" if resource.startswith("punkt") else f"corpora/{resource}"
        try:
            nltk.data.find(lookup_path)
            print(f"{resource} already exists")
        except LookupError:
            print(f"Downloading {resource}...")
            nltk.download(resource, download_dir=custom_dir, quiet=True)
            print(f"{resource} downloaded")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="./nltk_data", help="NLTK data directory")
    args = parser.parse_args()
    setup_nltk_data(args.data_dir)


if __name__ == "__main__":
    main()
