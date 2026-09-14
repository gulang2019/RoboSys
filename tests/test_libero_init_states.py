import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("task_id", [0, 9])
def test_bundled_numpy_initial_states_load(tmp_path, task_id):
    # LIBERO imports alter asyncio globally; isolate them from server unit tests.
    root = Path(__file__).resolve().parents[1]
    package = root / "3rdparty/AutoHorizon/third_party/libero"
    (tmp_path / "config.yaml").write_text("{}\n")
    result = subprocess.run(
        [sys.executable, "-c", """
import sys
import numpy as np
from libero.libero import benchmark
benchmark.get_libero_path = lambda key: sys.argv[1]
suite = benchmark.get_benchmark_dict()['libero_spatial']()
states = suite.get_task_init_states(int(sys.argv[2]))
assert len(states) >= 25
assert isinstance(states[0], np.ndarray)
assert states[0].ndim == 1
assert np.isfinite(states).all()
""", str(package / 'libero/libero/init_files'), str(task_id)],
        env=dict(os.environ, PYTHONPATH=str(package), LIBERO_CONFIG_PATH=str(tmp_path)),
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
