"""Check setup orchestration in an empty checkout without installing packages."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize('skip_checkpoint', ['0', '1'])
def test_profile_setup_defaults_and_pinned_checkout(tmp_path, skip_checkpoint):
    source = Path(__file__).resolve().parents[1] / 'scripts/profile/setup.sh'
    script = tmp_path / 'scripts/profile/setup.sh'
    script.parent.mkdir(parents=True)
    script.write_text(source.read_text().replace(
        'export PATH="$HOME/.local/bin:$PATH"', ': # retain command stubs'))
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    stub = bin_dir / 'stub'
    stub.write_text(f'#!{sys.executable}\n' + '''
import json, os, pathlib, sys
root = pathlib.Path(os.environ['SETUP_TEST_ROOT'])
name, args = pathlib.Path(sys.argv[0]).name, sys.argv[1:]
with (root/'calls.jsonl').open('a') as log:
    log.write(json.dumps([name, args]) + '\\n')
def environment(path):
    binary = path/'bin'
    binary.mkdir(parents=True, exist_ok=True)
    for tool in ['python', 'cmake', 'ninja']:
        target = binary/tool
        if not target.exists(): target.symlink_to(root/'bin/stub')
if name == 'uv':
    if args[0] == 'venv': environment(pathlib.Path(args[1]))
    if args[0] == 'sync': environment(pathlib.Path(os.environ['UV_PROJECT_ENVIRONMENT']))
elif name == 'git':
    if 'clone' in args: pathlib.Path(args[-1]).mkdir(parents=True)
    if 'rev-parse' in args: print(os.environ.get('TEST_OPENPI_REV', '215abfb217dbac7d5f1273282331b9b1866c0479'))
    if 'submodule' in args:
        checkout = root/args[-1]
        checkout.mkdir(parents=True)
        (checkout/'pyproject.toml').touch()
elif name == 'nvcc': print('Cuda compilation tools, release 12.8, V12.8.61')
elif name == 'python':
    if args and args[0] == '-': compile(sys.stdin.read(), '<setup helper>', 'exec')
    if args and args[0].endswith('.py'):
        assert pathlib.Path(args[0]).exists(), args
''')
    stub.chmod(0o755)
    for tool in ['uv', 'git', 'nvcc', 'c++']:
        (bin_dir / tool).symlink_to(stub)
    (tmp_path / 'scripts/convert_jax_model_to_pytorch.py').touch()
    (tmp_path / 'scripts/profile/checkpoints.py').touch()
    env = dict(os.environ, PATH=f'{bin_dir}:{os.environ["PATH"]}',
               SETUP_TEST_ROOT=str(tmp_path), CUDA_HOME=str(tmp_path))
    env['SKIP_CHECKPOINT'] = skip_checkpoint
    env.pop('INSTALL_OPENPI', None)
    env.pop('BUILD_FLASHRT', None)
    for _ in range(2):
        result = subprocess.run(['bash', str(script)], env=env,
                                capture_output=True, text=True, timeout=10)
        assert result.returncode == 0, result.stdout + result.stderr
    calls = [json.loads(line) for line in (tmp_path / 'calls.jsonl').read_text().splitlines()]
    assert sum(name == 'python' and args and args[0] == str(tmp_path / 'scripts/profile/checkpoints.py')
               for name, args in calls) == (0 if skip_checkpoint == '1' else 2)
    assert sum(name == 'uv' and args[0] == 'sync' for name, args in calls) == 2
    assert sum(name == 'cmake' and args[0] == '--build' for name, args in calls) == 2
    assert sum(name == 'git' and 'clone' in args for name, args in calls) == 2
    assert any(name == 'python' and args == [str(tmp_path / 'scripts/convert_jax_model_to_pytorch.py'), '--help']
               for name, args in calls)
    result = subprocess.run(['bash', str(script)], env=dict(env, TEST_OPENPI_REV='wrong'),
                            capture_output=True, text=True, timeout=10)
    assert result.returncode != 0
    assert 'leaving it untouched' in result.stderr
