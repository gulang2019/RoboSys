import importlib.util
from pathlib import Path
import subprocess

import pytest

spec = importlib.util.spec_from_file_location(
    'checkpoint_setup', Path(__file__).resolve().parents[1] / 'scripts/profile/checkpoints.py')
checkpoints = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checkpoints)


def complete(path):
    path.mkdir(parents=True, exist_ok=True)
    (path / 'model.safetensors').write_bytes(b'weights')
    stats = path / checkpoints.NORM_STATS
    stats.parent.mkdir(parents=True, exist_ok=True)
    stats.write_text('{}')


def test_prepare_and_reuse_checkpoint(tmp_path, monkeypatch):
    calls = []
    def run(args, *, check):
        assert check
        calls.append(args)
        if '--output_path' in args:
            complete(Path(args[args.index('--output_path') + 1]))
        else:
            assert args[-2:] == ['--destination', str(tmp_path / 'jax')]
    monkeypatch.setattr(checkpoints.subprocess, 'run', run)
    output = tmp_path / 'torch'
    checkpoints.prepare_checkpoint(tmp_path / 'jax', output)
    assert len(calls) == 2
    assert (output / checkpoints.NORM_STATS).exists()
    checkpoints.prepare_checkpoint(tmp_path / 'jax', output)
    assert len(calls) == 2
    assert not list(tmp_path.glob('.torch-*'))


@pytest.mark.parametrize('failure', ['exception', 'missing_assets'])
def test_failed_conversion_is_retryable(tmp_path, monkeypatch, failure):
    def run(args, *, check):
        if '--output_path' not in args:
            return
        staged = Path(args[args.index('--output_path') + 1])
        staged.mkdir()
        (staged / 'model.safetensors').touch()
        if failure == 'exception':
            raise subprocess.CalledProcessError(1, args)
    monkeypatch.setattr(checkpoints.subprocess, 'run', run)
    with pytest.raises((RuntimeError, subprocess.CalledProcessError)):
        checkpoints.prepare_checkpoint(tmp_path / 'jax', tmp_path / 'torch')
    assert not (tmp_path / 'torch').exists()
    assert not list(tmp_path.glob('.torch-*'))


def test_preserve_incomplete_checkpoint(tmp_path, monkeypatch):
    output = tmp_path / 'torch'
    output.mkdir()
    weights = output / 'model.safetensors'
    weights.write_bytes(b'preserve me')
    monkeypatch.setattr(checkpoints.subprocess, 'run', lambda *a, **k: pytest.fail('should not run'))
    with pytest.raises(RuntimeError, match='incomplete'):
        checkpoints.prepare_checkpoint(tmp_path / 'jax', output)
    assert weights.read_bytes() == b'preserve me'
