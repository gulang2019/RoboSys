import importlib.util
import json
from pathlib import Path
import sys
import threading
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from robort.schemas import InferenceResponse


@pytest.fixture
def libero(monkeypatch):
    # Keep these unit tests independent of the simulator and its rendering setup.
    package = ModuleType("libero")
    library = ModuleType("libero.libero")
    library.benchmark = SimpleNamespace(get_benchmark_dict=Mock())
    library.get_libero_path = Mock()
    envs = ModuleType("libero.libero.envs")
    envs.OffScreenRenderEnv = Mock()
    package.libero = library
    library.envs = envs
    for name, module in [("libero", package), ("libero.libero", library),
                         ("libero.libero.envs", envs)]:
        monkeypatch.setitem(sys.modules, name, module)
    path = Path(__file__).resolve().parents[1] / "benchmark" / "libero.py"
    spec = importlib.util.spec_from_file_location("libero_benchmark_test", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.tqdm, "tqdm", lambda items, **kwargs: items)
    return module


@pytest.mark.parametrize("workers", [1, 2])
def test_parallel_tasks_and_aggregate_results(libero, monkeypatch, tmp_path, workers):
    suite = SimpleNamespace(n_tasks=4)
    libero.benchmark.get_benchmark_dict.return_value = {"libero_spatial": lambda: suite}
    args = libero.Args(max_parallel_tasks=workers, inference_delay=2, use_rtc=True,
                       video_out_path=str(tmp_path / "videos"),
                       results_path=str(tmp_path / "results" / "eval.json"))
    barrier = threading.Barrier(workers)
    lock = threading.Lock()
    active = peak = 0
    thread_ids = set()

    def evaluate(actual_args, actual_suite, max_steps, task_id):
        nonlocal active, peak
        assert actual_args is args
        assert actual_suite is suite
        assert max_steps == 220
        with lock:
            active += 1
            peak = max(peak, active)
            thread_ids.add(threading.get_ident())
        barrier.wait(timeout=5)
        with lock:
            active -= 1
        return {"task_id": task_id, "episodes": 4, "successes": task_id}

    monkeypatch.setattr(libero, "eval_one_task", evaluate)
    libero.eval_libero(args)

    result = json.loads(Path(args.results_path).read_text())
    assert result["inference_delay"] == 2
    assert result["use_rtc"] is True
    assert [task["task_id"] for task in result["tasks"]] == [0, 1, 2, 3]
    assert (result["episodes"], result["successes"], result["success_rate"]) == (16, 6, 6 / 16)
    assert peak == workers
    assert threading.get_ident() not in thread_ids
    assert Path(args.video_out_path).is_dir()


@pytest.mark.parametrize("task_id", [0, 2])
def test_selected_task_runs_without_executor(libero, monkeypatch, tmp_path, task_id):
    suite = SimpleNamespace(n_tasks=4)
    libero.benchmark.get_benchmark_dict.return_value = {"libero_spatial": lambda: suite}
    args = libero.Args(task_id=task_id, video_out_path=str(tmp_path))
    evaluate = Mock(return_value={"episodes": 2, "successes": 1})
    monkeypatch.setattr(libero, "eval_one_task", evaluate)
    executor = Mock(side_effect=AssertionError("single task should run directly"))
    monkeypatch.setattr(libero, "ThreadPoolExecutor", executor)

    libero.eval_libero(args)

    evaluate.assert_called_once_with(args, suite, 220, task_id)
    executor.assert_not_called()


def test_parallel_task_error_propagates(libero, monkeypatch, tmp_path):
    libero.benchmark.get_benchmark_dict.return_value = {
        "libero_spatial": lambda: SimpleNamespace(n_tasks=2)
    }
    monkeypatch.setattr(libero, "eval_one_task", Mock(side_effect=RuntimeError("inference failed")))
    with pytest.raises(RuntimeError, match="inference failed"):
        libero.eval_libero(libero.Args(video_out_path=str(tmp_path), fail_on_error=True))


@pytest.mark.parametrize("use_rtc", [False, True])
@pytest.mark.parametrize("local", [False, True])
def test_episode_preprocessing_and_action_chunks(libero, monkeypatch, tmp_path, use_rtc, local):
    img = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    obs = {"agentview_image": img, "robot0_eye_in_hand_image": img + 20,
           "robot0_eef_pos": np.array([1., 2., 3.]),
           "robot0_eef_quat": np.array([0., 0., 0., 1.]),
           "robot0_gripper_qpos": np.array([0.1, 0.2])}
    env = Mock()
    env.set_init_state.return_value = obs
    env.step.side_effect = [(obs, 0, False, {})] * 3 + [(obs, 1, True, {})]
    monkeypatch.setattr(libero, "_get_libero_env", Mock(return_value=(env, "test task")))
    suite = Mock()
    suite.get_task_init_states.return_value = ["initial state"]
    actions = np.arange(21).reshape(3, 7)
    client = Mock()
    client.infer.return_value = InferenceResponse(actions, rtc_prev_actions=actions)
    factory = Mock(return_value=client)
    monkeypatch.setattr(libero, "WebsocketClientPolicy", factory)
    video = Mock()
    monkeypatch.setattr(libero.imageio, "mimwrite", video)
    args = libero.Args(host="localhost", port=9000, num_trials_per_task=1,
                       num_steps_wait=1, replan_steps=2, inference_delay=0,
                       resize_size=2, use_rtc=use_rtc, video_out_path=str(tmp_path))

    result = libero.eval_one_task(args, suite, 5, 0, **({"client": client} if local else {}))

    if local:
        factory.assert_not_called()
    else:
        factory.assert_called_once_with("localhost", 9000)
    assert client.infer.call_count == 2
    request = client.infer.call_args_list[0].args[0]
    np.testing.assert_array_equal(request.observation["observation/image"], img[::-1, ::-1])
    np.testing.assert_array_equal(request.observation["observation/wrist_image"], (img + 20)[::-1, ::-1])
    np.testing.assert_allclose(request.observation["observation/state"], [1, 2, 3, 0, 0, 0, .1, .2])
    assert request.observation["prompt"] == "test task"
    assert request.inference_type == ("rtc" if use_rtc else "sync")
    assert (request.rtc_s_param, request.rtc_d_param) == (2, 0)
    assert [call.args[0] for call in env.step.call_args_list] == [
        libero.LIBERO_DUMMY_ACTION, actions[0].tolist(), actions[1].tolist(), actions[0].tolist()
    ]
    assert (result["episodes"], result["successes"], result["success_rate"]) == (1, 1, 1)
    assert video.call_args.args[0].name == "task0_ep0_test_task_success.mp4"
    env.close.assert_called_once_with()
    if local:
        client.close.assert_not_called()  # The evaluation owns the shared client.
    else:
        client.close.assert_called_once_with()


@pytest.mark.parametrize("stage", ["connect", "reset", "infer", "video", "short_chunk", "missing_rtc"])
def test_environment_closes_on_error(libero, monkeypatch, tmp_path, stage):
    obs = {"agentview_image": np.zeros((2, 2, 3), dtype=np.uint8),
           "robot0_eye_in_hand_image": np.zeros((2, 2, 3), dtype=np.uint8),
           "robot0_eef_pos": np.zeros(3), "robot0_eef_quat": np.array([0., 0., 0., 1.]),
           "robot0_gripper_qpos": np.zeros(2)}
    env = Mock()
    env.set_init_state.return_value = obs
    env.step.return_value = (obs, 1, True, {})
    monkeypatch.setattr(libero, "_get_libero_env", Mock(return_value=(env, "test task")))
    suite = Mock()
    suite.get_task_init_states.return_value = ["initial state"]
    client = Mock()
    client.infer.return_value = InferenceResponse(np.zeros((5, 7)))
    factory = Mock(return_value=client)
    video = Mock()
    monkeypatch.setattr(libero, "WebsocketClientPolicy", factory)
    monkeypatch.setattr(libero.imageio, "mimwrite", video)
    if stage in ("short_chunk", "missing_rtc"):
        error = "need at least 6" if stage == "short_chunk" else "RTC requires model-space"
    else:
        failing_call = {"connect": factory, "reset": env.reset,
                        "infer": client.infer, "video": video}[stage]
        failing_call.side_effect = RuntimeError(stage)
        error = stage
    args = libero.Args(num_trials_per_task=1, num_steps_wait=0, resize_size=2,
                       inference_delay=1 if stage == "short_chunk" else 0,
                       use_rtc=stage == "missing_rtc",
                       video_out_path=str(tmp_path), fail_on_error=True)

    with pytest.raises((RuntimeError, ValueError), match=error):
        libero.eval_one_task(args, suite, 5, 0)

    env.close.assert_called_once_with()
    if stage != "connect":
        client.close.assert_called_once_with()


@pytest.mark.parametrize("delay", [0, 1, 3])
@pytest.mark.parametrize("use_rtc", [False, True])
def test_simulated_delay_and_rtc_history(libero, monkeypatch, tmp_path, delay, use_rtc):
    applied, requests = [], []
    episode_step = 0
    env = Mock()

    def observation():
        return {"agentview_image": np.zeros((2, 2, 3), dtype=np.uint8),
                "robot0_eye_in_hand_image": np.zeros((2, 2, 3), dtype=np.uint8),
                "robot0_eef_pos": np.full(3, episode_step),
                "robot0_eef_quat": np.array([0., 0., 0., 1.]),
                "robot0_gripper_qpos": np.zeros(2)}

    def reset():
        nonlocal episode_step
        episode_step = 0

    def step(action):
        nonlocal episode_step
        applied.append(action)
        episode_step += 1
        return observation(), 0, False, {}

    env.reset.side_effect = reset
    env.set_init_state.side_effect = lambda state: observation()
    env.step.side_effect = step
    monkeypatch.setattr(libero, "_get_libero_env", Mock(return_value=(env, "test task")))
    suite = Mock()
    suite.get_task_init_states.return_value = [0, 1]
    chunks = [np.repeat((100 * i + np.arange(8))[:, None], 7, axis=1) for i in range(8)]
    # RTC uses raw model-space actions, whose dimension differs from robot actions.
    raw_chunks = [np.full((8, 32), i) for i in range(8)]

    def infer(request):
        i = len(requests)
        requests.append(request)
        return InferenceResponse(chunks[i], raw_chunks[i])

    client = Mock()
    client.infer.side_effect = infer
    monkeypatch.setattr(libero, "WebsocketClientPolicy", Mock(return_value=client))
    monkeypatch.setattr(libero.imageio, "mimwrite", Mock())
    args = libero.Args(num_trials_per_task=2, num_steps_wait=0, replan_steps=2,
                       inference_delay=delay, resize_size=2, use_rtc=use_rtc,
                       video_out_path=str(tmp_path), fail_on_error=True)

    result = libero.eval_one_task(args, suite, 7, 0)

    assert result["episodes"] == 2
    assert len(requests) == 8
    for episode in range(2):
        for t in range(7):
            expected = libero.LIBERO_DUMMY_ACTION if t < delay else chunks[
                episode * 4 + (t - delay) // 2
            ][delay + (t - delay) % 2].tolist()
            assert applied[episode * 7 + t] == expected
        for i in range(4):
            request = requests[episode * 4 + i]
            np.testing.assert_array_equal(request.observation["observation/state"][:3], [i * 2] * 3)
            assert request.inference_type == ("rtc" if use_rtc else "sync")
            assert (request.rtc_s_param, request.rtc_d_param) == (2, delay)
            assert request.max_execution_horizon == 2 + delay
            if use_rtc and i:
                assert request.previous_action is raw_chunks[episode * 4 + i - 1]
            else:
                assert request.previous_action is None


@pytest.mark.parametrize("steps,delay", [(0, 0), (1, -1)])
def test_invalid_schedule_fails_before_environment_creation(libero, steps, delay):
    suite = Mock()
    with pytest.raises(ValueError, match="replan_steps must be positive"):
        libero.eval_one_task(libero.Args(replan_steps=steps, inference_delay=delay), suite, 5, 0)
    suite.get_task.assert_not_called()


@pytest.mark.parametrize("fail", [False, True])
def test_local_cli_reuses_and_closes_one_client(libero, monkeypatch, tmp_path, fail):
    suite = SimpleNamespace(n_tasks=3)
    libero.benchmark.get_benchmark_dict.return_value = {"libero_spatial": lambda: suite}
    client = Mock()
    factory = Mock(return_value=client)
    monkeypatch.setattr(libero, "LocalClientPolicy", factory)
    executor = Mock(side_effect=AssertionError("local evaluation must be sequential"))
    monkeypatch.setattr(libero, "ThreadPoolExecutor", executor)
    seen = []

    def evaluate(args, actual_suite, max_steps, task_id, *, client):
        seen.append((task_id, client))
        client.close.assert_not_called()
        if fail:
            raise RuntimeError("inference failed")
        return {"episodes": 1, "successes": 1}

    monkeypatch.setattr(libero, "eval_one_task", evaluate)

    def run():
        libero.tyro.cli(libero.eval_libero, args=[
            "--client-args", '{"type": "local", "model_dir": "checkpoint", "device": "cpu"}',
            "--args.video-out-path", str(tmp_path),
        ])

    if fail:
        with pytest.raises(RuntimeError, match="inference failed"):
            run()
    else:
        run()
    factory.assert_called_once_with(model_dir="checkpoint", device="cpu")
    assert seen == [(i, client) for i in range(1 if fail else 3)]
    client.close.assert_called_once_with()
    executor.assert_not_called()


def test_local_rtc_rejected_before_loading(libero, monkeypatch):
    factory = Mock()
    monkeypatch.setattr(libero, "LocalClientPolicy", factory)
    with pytest.raises(ValueError, match="disable RTC"):
        libero.eval_libero(libero.Args(use_rtc=True), client_args='{"type": "local"}')
    factory.assert_not_called()
    libero.benchmark.get_benchmark_dict.assert_not_called()


def test_websocket_client_options_forwarded(libero, monkeypatch, tmp_path):
    suite = SimpleNamespace(n_tasks=1)
    libero.benchmark.get_benchmark_dict.return_value = {"libero_spatial": lambda: suite}
    evaluate = Mock(return_value={"episodes": 1, "successes": 1})
    monkeypatch.setattr(libero, "eval_one_task", evaluate)
    args = libero.Args(task_id=0, video_out_path=str(tmp_path))
    libero.eval_libero(args, client_args='{"type":"websocket","host":"localhost","port":9000}')
    evaluate.assert_called_once_with(
        args, suite, 220, 0, client_options={"host": "localhost", "port": 9000})
