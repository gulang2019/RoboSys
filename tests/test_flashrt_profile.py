from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock
import math

import pytest

from robort.profile.runner import Runner
from robort.profile.schemas import PolicyConfig, RunnerConfig


@pytest.mark.parametrize('failure', [False, True])
def test_native_targets_are_repeated_and_closed(failure):
    runner = Runner(RunnerConfig(num_warmup=1, num_iter=2))
    targets = {stage: Mock() for stage in ('embed', 'encode', 'decode')}
    if failure:
        targets['encode'].side_effect = RuntimeError('native failure')
    policy = SimpleNamespace(
        config=PolicyConfig(backend='flash_rt', batch_sizes={s: [1] for s in targets}),
        profile_functions=Mock(return_value=targets))
    measurements = SimpleNamespace(latencies=[1., 2.], energies=[3., 4.], memories=[0., 0.], tick=Mock())
    runner.profiler = SimpleNamespace(measure=lambda **kw: nullcontext(measurements))
    stream = object()
    if failure:
        with pytest.raises(RuntimeError, match='native failure'):
            runner._profile_batch(policy, SimpleNamespace(), stream)
    else:
        result = runner._profile_batch(policy, SimpleNamespace(), stream)
        for stage, target in targets.items():
            assert target.call_count == 3
            assert result.stages[stage].lat[1] == (1.5, 0.5)
            assert all(math.isnan(v) for v in result.stages[stage].mem_fp_activation_gb[1])
    policy.profile_functions.assert_called_once_with(stream)
    for target in targets.values():
        target.close.assert_called_once()
