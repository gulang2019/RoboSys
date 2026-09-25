"""The required FlashRT hook must be installable without a private Git commit."""
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = 'flash_rt/models/pi05/pipeline_rtx_batched.py'
PATCH = ROOT / 'patches/flashrt-pi05-decoder-hook.patch'
SCRIPT = ROOT / 'scripts/profile/apply-flashrt-patch.sh'


@pytest.fixture
def checkout(tmp_path):
    source = subprocess.check_output(
        ['git', '-C', str(ROOT / '3rdparty/FlashRT'), 'show', f'eaf90192:{SOURCE}'])
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
    target = tmp_path / SOURCE
    target.parent.mkdir(parents=True)
    target.write_bytes(source)
    return tmp_path, target, source


def test_patch_applies_once_and_preserves_upstream(checkout):
    directory, target, original = checkout
    subprocess.run(['bash', str(SCRIPT), str(directory)], check=True)
    patched = target.read_bytes()
    assert b'def _decoder_qkv_rope_batched(' in patched
    assert b'self._decoder_qkv_rope_batched(i, enc_seq, ds, stream)' in patched
    subprocess.run(['bash', str(SCRIPT), str(directory)], check=True)
    assert target.read_bytes() == patched
    subprocess.run(['git', '-C', str(directory), 'apply', '--reverse', str(PATCH)], check=True)
    assert target.read_bytes() == original


def test_patch_conflict_preserves_local_edits(checkout):
    directory, target, original = checkout
    changed = original.replace(b'# C2: QKV split + RoPE', b'# Local incompatible decoder edits')
    target.write_bytes(changed)
    result = subprocess.run(['bash', str(SCRIPT), str(directory)], capture_output=True, text=True)
    assert result.returncode == 1
    assert 'incompatible edits' in result.stderr
    assert target.read_bytes() == changed
