"""Static checks for the Thor setup entry point; installation needs Thor hardware."""
from pathlib import Path
import subprocess


def test_thor_setup_help_and_platform_contract():
    root = Path(__file__).resolve().parents[1]
    script = root / 'scripts/profile/setup-thor.sh'
    result = subprocess.run(['bash', str(script), '--help'], capture_output=True,
                            text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert 'Linux aarch64' in result.stdout
    assert 'CUDA 13.x' in result.stdout
    assert 'FlashRT is built' in result.stdout

    source = script.read_text()
    assert 'Linux-aarch64' in source
    assert 'targets/sbsa-linux/lib' in source
    assert 'torch==2.9.0+cu130' in source
    assert 'capability == (11, 0)' in source
    assert '--no-deps' in source
    assert '-DGPU_ARCH=110' in source
