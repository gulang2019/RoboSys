import csv
from dataclasses import fields
from itertools import product
from unittest.mock import Mock

import pytest

from robort.profile import main as sweep
from robort.profile.policies import FlashRTBackend
from robort.profile.schemas import HardwareConfig, HardwareProfile, PolicyConfig, PolicyProfile, RunnerConfig, StageProfile


def test_sweep_initializes_each_backend_once(tmp_path, monkeypatch):
    def initialize(config):
        config.hardware_name = 'test'
        config.num_sms = 16
        config.mem_cap_gb = 8
        config.max_power_w = 100

    monkeypatch.setattr(HardwareConfig, '__post_init__', initialize)
    runner = Mock()
    runner.profile_hardware.side_effect = RuntimeError('skip hardware')
    runner.profile_policy.return_value = {}
    monkeypatch.setattr(sweep, 'Runner', lambda _: runner)
    policy = PolicyConfig(batch_sizes=[1])
    choices = {field.name: [getattr(policy, field.name)] for field in fields(PolicyConfig)}
    choices.update(backend=['openpi', 'flash_rt'], batch_sizes=[[1], [2]])
    sweep.main(RunnerConfig(output_dir=str(tmp_path)),
               {'power_perc': [1.0, 0.8], 'sm_perc': [0.5, 1.0]}, choices)
    assert runner.init_backend.call_count == 2
    assert [call.args[0].backend for call in runner.init_backend.call_args_list] == ['openpi', 'flash_rt']
    for call in runner.init_backend.call_args_list:
        assert len(call.args[1]) == 4
        assert call.args[2] == [1, 2]
    assert runner.profile_policy.call_count == 16
    calls = [call[0] for call in runner.mock_calls]
    assert max(i for i, name in enumerate(calls) if name == 'init_backend') < calls.index('profile_policy')
    runner.close.assert_called_once()


def test_default_policies_are_supported():
    names, values = zip(*sweep.POLICY_CHOICES)
    for choices in product(*values):
        config = PolicyConfig(**dict(zip(names, choices)))
        assert config.backend in {'openpi', 'flash_rt'}
        assert config.batch_sizes and all(1 <= n <= config.max_batch_size for n in config.batch_sizes)


@pytest.mark.parametrize('phase', ['hardware', 'policy'])
def test_sweep_persists_before_interrupt_and_merges(tmp_path, monkeypatch, phase):
    def initialize(config):
        config.hardware_name = 'test'
        config.num_sms = 16
        config.mem_cap_gb = 8
        config.max_power_w = 100

    monkeypatch.setattr(HardwareConfig, '__post_init__', initialize)
    hardware = HardwareConfig(1.0, 1.0)
    policy = PolicyConfig(batch_sizes=[1])
    hardware_result = HardwareProfile(hardware, **{
        field.name: 1.0 for field in fields(HardwareProfile) if field.name != 'hardware_config'})
    policy_result = PolicyProfile(hardware, policy, {
        'vis': StageProfile(1, 2, 3, 4, 5, 6, 7, 8)})
    runner = Mock()
    runner.profile_hardware.return_value = hardware_result
    runner.profile_policy.return_value = {1: policy_result}
    target = getattr(runner, f'profile_{phase}')
    result = hardware_result if phase == 'hardware' else policy_result
    target.side_effect = [result if phase == 'hardware' else {1: result}, RuntimeError('unsupported'), KeyboardInterrupt()]
    monkeypatch.setattr(sweep, 'Runner', lambda _: runner)
    hw_choices = {'power_perc': [1.0], 'sm_perc': [0.6, 0.8, 1.0]}
    policy_choices = {field.name: [getattr(policy, field.name)] for field in fields(PolicyConfig)}
    config = RunnerConfig(output_dir=str(tmp_path))

    with pytest.raises(KeyboardInterrupt):
        sweep.main(config, hw_choices, policy_choices)
    runner.close.assert_called_once()
    path = tmp_path / f'{phase}_profile.csv'
    with path.open(newline='') as source:
        rows = list(csv.DictReader(source))
    assert len(rows) == 1
    assert (tmp_path / 'hardware_profile.csv').exists()

    target.side_effect = None
    if phase == 'hardware':
        result.idle_power_w = 42
        metric = 'idle_power_w'
    else:
        result.stages['vis'].lat_mean = 42
        metric = 'lat_mean'
    sweep.main(config, hw_choices, policy_choices)
    with path.open(newline='') as source:
        rows = list(csv.DictReader(source))
    assert len(rows) == 1
    assert float(rows[0][metric]) == 42


@pytest.mark.parametrize('legacy_sizes', [None, '[1]', '[1, 2]'])
def test_batch_size_rows_and_legacy_migration(tmp_path, monkeypatch, legacy_sizes):
    def initialize(config):
        config.hardware_name = 'test'
        config.num_sms = 16
        config.mem_cap_gb = 8
        config.max_power_w = 100

    monkeypatch.setattr(HardwareConfig, '__post_init__', initialize)
    hardware = HardwareConfig(1.0, 1.0)
    runner = Mock()
    runner.profile_hardware.side_effect = RuntimeError('skip hardware')
    runner.profile_policy.return_value = {
        size: PolicyProfile(hardware, PolicyConfig(batch_sizes=[size]), {
            'decode': StageProfile(1, 2, 3, 4, 5, 6, 7, 8)})
        for size in (1, 2)
    }
    monkeypatch.setattr(sweep, 'Runner', lambda _: runner)
    config = RunnerConfig(output_dir=str(tmp_path))
    choices = {'sm_perc': [1.0], 'power_perc': [1.0]}
    sweep.main(config, choices, {'batch_sizes': [[1, 2]]})
    path = tmp_path / 'policy_profile.csv'
    with path.open(newline='') as source:
        rows = list(csv.DictReader(source))
    assert [row['batch_size'] for row in rows] == ['1', '2']
    assert all('batch_sizes' not in row for row in rows)
    if legacy_sizes is not None:
        rows[0]['batch_sizes'] = legacy_sizes
        rows[0].pop('batch_size')
        with path.open('w', newline='') as target:
            writer = csv.DictWriter(target, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerow(rows[0])
    before = path.read_text()
    if legacy_sizes == '[1, 2]':
        with pytest.raises(ValueError, match='ambiguous batch_sizes'):
            sweep.main(config, choices, {'batch_sizes': [[1, 2]]})
        assert path.read_text() == before
    else:
        sweep.main(config, choices, {'batch_sizes': [[1, 2]]})
        with path.open(newline='') as source:
            rows = list(csv.DictReader(source))
        assert [row['batch_size'] for row in rows] == ['1', '2']
        assert all('batch_sizes' not in row for row in rows)
