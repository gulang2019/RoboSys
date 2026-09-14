"""Download the JAX pi05_libero checkpoint using OpenPI's resumable cache."""
import argparse
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, default=Path('checkpoints/pi05_libero'))
    args = parser.parse_args()
    target = args.destination.expanduser().resolve()
    marker = target / '.download_complete'
    if marker.exists():
        print(f'Checkpoint already downloaded: {target}')
        return
    if target.exists():
        # Preserve manually downloaded checkpoints and avoid merging partial trees.
        if (target/'params/_METADATA').is_file() and (target/'assets/physical-intelligence/libero/norm_stats.json').is_file():
            print(f'Using existing checkpoint: {target} (metadata found; inference validates weights)')
            return
        raise SystemExit(f'{target} exists but is incomplete; move it aside before retrying.')
    from openpi.shared import download
    source = download.maybe_download('gs://openpi-assets/checkpoints/pi05_libero')
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target)
    marker.touch()
    print(f'Downloaded checkpoint: {target}')


if __name__ == '__main__':
    main()
