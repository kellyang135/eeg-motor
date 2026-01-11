#!/usr/bin/env python3
"""
Download BCI Competition IV Dataset 2a

This script downloads the 9-subject motor imagery dataset from BNCI Horizon 2020.
The data is in GDF format and includes:
- Training sessions (with labels)
- Evaluation sessions (with labels in separate file)

Usage:
    python scripts/download_data.py
    python scripts/download_data.py --subjects 1 2 3
"""

import argparse
import os
from pathlib import Path
import requests
from tqdm import tqdm


# BNCI Horizon 2020 hosts the BCI Competition datasets
BASE_URL = "https://bnci-horizon-2020.eu/database/data-sets"
DATASET_URL = "https://www.bbci.de/competition/iv/download"

# Direct download links for GDF files (BNCI mirror)
DATA_URLS = {
    # Training data
    "A01T.gdf": "https://lampx.tugraz.at/~bci/database/001-2014/A01T.gdf",
    "A02T.gdf": "https://lampx.tugraz.at/~bci/database/001-2014/A02T.gdf",
    "A03T.gdf": "https://lampx.tugraz.at/~bci/database/001-2014/A03T.gdf",
    "A04T.gdf": "https://lampx.tugraz.at/~bci/database/001-2014/A04T.gdf",
    "A05T.gdf": "https://lampx.tugraz.at/~bci/database/001-2014/A05T.gdf",
    "A06T.gdf": "https://lampx.tugraz.at/~bci/database/001-2014/A06T.gdf",
    "A07T.gdf": "https://lampx.tugraz.at/~bci/database/001-2014/A07T.gdf",
    "A08T.gdf": "https://lampx.tugraz.at/~bci/database/001-2014/A08T.gdf",
    "A09T.gdf": "https://lampx.tugraz.at/~bci/database/001-2014/A09T.gdf",
    # Evaluation data
    "A01E.gdf": "https://lampx.tugraz.at/~bci/database/001-2014/A01E.gdf",
    "A02E.gdf": "https://lampx.tugraz.at/~bci/database/001-2014/A02E.gdf",
    "A03E.gdf": "https://lampx.tugraz.at/~bci/database/001-2014/A03E.gdf",
    "A04E.gdf": "https://lampx.tugraz.at/~bci/database/001-2014/A04E.gdf",
    "A05E.gdf": "https://lampx.tugraz.at/~bci/database/001-2014/A05E.gdf",
    "A06E.gdf": "https://lampx.tugraz.at/~bci/database/001-2014/A06E.gdf",
    "A07E.gdf": "https://lampx.tugraz.at/~bci/database/001-2014/A07E.gdf",
    "A08E.gdf": "https://lampx.tugraz.at/~bci/database/001-2014/A08E.gdf",
    "A09E.gdf": "https://lampx.tugraz.at/~bci/database/001-2014/A09E.gdf",
    # Labels for evaluation data
    "A01E.mat": "https://lampx.tugraz.at/~bci/database/001-2014/A01E.mat",
    "A02E.mat": "https://lampx.tugraz.at/~bci/database/001-2014/A02E.mat",
    "A03E.mat": "https://lampx.tugraz.at/~bci/database/001-2014/A03E.mat",
    "A04E.mat": "https://lampx.tugraz.at/~bci/database/001-2014/A04E.mat",
    "A05E.mat": "https://lampx.tugraz.at/~bci/database/001-2014/A05E.mat",
    "A06E.mat": "https://lampx.tugraz.at/~bci/database/001-2014/A06E.mat",
    "A07E.mat": "https://lampx.tugraz.at/~bci/database/001-2014/A07E.mat",
    "A08E.mat": "https://lampx.tugraz.at/~bci/database/001-2014/A08E.mat",
    "A09E.mat": "https://lampx.tugraz.at/~bci/database/001-2014/A09E.mat",
}


def download_file(url: str, dest_path: Path, chunk_size: int = 8192) -> bool:
    """Download a file with progress bar."""
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))

        with open(dest_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True,
                     desc=dest_path.name, leave=False) as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        return True
    except Exception as e:
        print(f"  Error downloading {url}: {e}")
        return False


def download_dataset(data_dir: Path, subjects: list[int] | None = None):
    """Download BCI Competition IV Dataset 2a."""

    data_dir.mkdir(parents=True, exist_ok=True)

    # Filter files by subject if specified
    files_to_download = DATA_URLS.copy()
    if subjects:
        files_to_download = {
            fname: url for fname, url in DATA_URLS.items()
            if any(f"A0{s}" in fname for s in subjects)
        }

    print(f"Downloading {len(files_to_download)} files to {data_dir}")
    print("-" * 50)

    success = 0
    failed = []

    for filename, url in files_to_download.items():
        dest_path = data_dir / filename

        if dest_path.exists():
            print(f"  {filename} already exists, skipping")
            success += 1
            continue

        if download_file(url, dest_path):
            success += 1
        else:
            failed.append(filename)

    print("-" * 50)
    print(f"Downloaded: {success}/{len(files_to_download)} files")

    if failed:
        print(f"Failed: {failed}")
        print("\nNote: If downloads fail, you may need to manually download from:")
        print("  https://bnci-horizon-2020.eu/database/data-sets")
        print("  Search for 'Four class motor imagery (001-2014)'")

    return len(failed) == 0


def main():
    parser = argparse.ArgumentParser(
        description="Download BCI Competition IV Dataset 2a"
    )
    parser.add_argument(
        "--subjects",
        type=int,
        nargs="+",
        choices=range(1, 10),
        help="Specific subjects to download (1-9). Downloads all if not specified."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).parent.parent / "data" / "raw",
        help="Directory to save data files"
    )

    args = parser.parse_args()

    print("=" * 50)
    print("BCI Competition IV Dataset 2a Downloader")
    print("=" * 50)
    print(f"\nDataset info:")
    print("  - 9 subjects, 22 EEG channels, 250 Hz")
    print("  - 4 classes: left hand, right hand, feet, tongue")
    print("  - ~288 trials per subject")
    print()

    success = download_dataset(args.data_dir, args.subjects)

    if success:
        print("\nData download complete!")
        print(f"Files saved to: {args.data_dir.resolve()}")
        print("\nNext step: Run notebook 01_data_exploration.ipynb")

    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
