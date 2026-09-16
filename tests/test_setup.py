"""Exercise setup orchestration without network, package installs, or a GPU."""
import os
from pathlib import Path
import subprocess
import sys


def test_setup_is_repeatable_and_preserves_incompatible_checkout(tmp_path):
    root = Path(__file__).resolve().parents[1]
    script = (root / 'setup.sh').read_text().replace('export PATH="$HOME/.local/bin:$PATH"', ': # retain test command stubs')
    (tmp_path / 'setup.sh').write_text(script)
    profile_script = tmp_path / 'scripts/profile/setup.sh'
    profile_script.parent.mkdir(parents=True)
    profile_script.write_text('echo profile-setup >> "${SETUP_TEST_ROOT}/profile-calls"\n')
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    stub = bin_dir / 'stub'
    stub.write_text(f'#!{sys.executable}\n' + '''
import json, os, pathlib, sys
root = pathlib.Path(os.environ['SETUP_TEST_ROOT'])
name = pathlib.Path(sys.argv[0]).name
args = sys.argv[1:]
with (root/'calls.jsonl').open('a') as f:
    f.write(json.dumps([name, args]) + '\\n')
if name == 'git':
    if 'clone' in args:
        pathlib.Path(args[-1]).mkdir(parents=True)
    elif 'rev-parse' in args:
        print(os.environ.get('TEST_LIBERO_REV', '8f1084e3132a39270c3a13ebe37270a43ece2a01'))
    elif 'apply' in args:
        marker = root / pathlib.Path(args[-1]).name
        if '--reverse' in args:
            sys.exit(0 if marker.exists() else 1)
        if '--check' in args:
            sys.exit(1 if marker.exists() else 0)
        marker.touch()
    elif 'submodule' in args:
        (root/args[-1]).mkdir(parents=True, exist_ok=True)
elif name == 'uv' and 'sync' in args:
    python = root/'3rdparty/armory/.venv/bin/python'
    python.parent.mkdir(parents=True, exist_ok=True)
    python.symlink_to(root/'bin/stub') if not python.exists() else None
elif name == 'python' and args == ['-']:
    compile(sys.stdin.read(), '<setup helper>', 'exec')
''')
    stub.chmod(0o755)
    for name in ('git', 'uv'):
        (bin_dir/name).symlink_to(stub)
    env = dict(os.environ, PATH=f'{bin_dir}:{os.environ["PATH"]}',
               SETUP_TEST_ROOT=str(tmp_path), SKIP_SYSTEM_DEPS='1',
               SKIP_GPU_CHECK='1', SKIP_CHECKPOINT='1')
    for _ in range(2):
        result = subprocess.run(['bash', str(tmp_path/'setup.sh')], env=env,
                                capture_output=True, text=True, timeout=10)
        assert result.returncode == 0, result.stdout + result.stderr
        assert 'Setup complete' in result.stdout
    assert (tmp_path / 'profile-calls').read_text().splitlines() == ['profile-setup'] * 2
    result = subprocess.run(['bash', str(tmp_path/'setup.sh')],
                            env=dict(env, TEST_LIBERO_REV='wrong-revision'),
                            capture_output=True, text=True, timeout=10)
    assert result.returncode != 0
    assert 'leaving it untouched' in result.stderr
