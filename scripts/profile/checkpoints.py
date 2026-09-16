"""Prepare the pi05_libero PyTorch checkpoint, preserving existing checkpoints."""
import argparse
from pathlib import Path
import subprocess
import sys
import tempfile

NORM_STATS = Path('assets/physical-intelligence/libero/norm_stats.json')


def prepare_checkpoint(checkpoint_dir: Path, output_dir: Path):
    checkpoint_dir = checkpoint_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if (output_dir / 'model.safetensors').is_file() and (output_dir / NORM_STATS).is_file():
        print(f'Using existing PyTorch checkpoint: {output_dir}')
        return
    if output_dir.exists():
        raise RuntimeError(f'{output_dir} exists but is incomplete; move it aside before retrying.')
    root = Path(__file__).resolve().parents[2]
    subprocess.run([sys.executable, str(root / 'download_openpi05.py'),
                    '--destination', str(checkpoint_dir)], check=True)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    # Publish only a complete conversion; failed runs leave the destination absent.
    with tempfile.TemporaryDirectory(prefix=f'.{output_dir.name}-', dir=output_dir.parent) as temporary:
        staged = Path(temporary) / 'checkpoint'
        subprocess.run([
            sys.executable, str(root / 'scripts/convert_jax_model_to_pytorch.py'),
            '--checkpoint_dir', str(checkpoint_dir), '--config_name', 'pi05_libero',
            '--output_path', str(staged), '--precision', 'bfloat16',
        ], check=True)
        if not (staged / 'model.safetensors').is_file() or not (staged / NORM_STATS).is_file():
            raise RuntimeError('Conversion did not produce model.safetensors and normalization assets')
        staged.rename(output_dir)
    print(f'PyTorch checkpoint ready: {output_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    prepare_checkpoint(args.checkpoint_dir, args.output_dir)
