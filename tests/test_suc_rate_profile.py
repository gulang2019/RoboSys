import csv
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "benchmark/suc_rate_profile.sh"


@pytest.fixture
def sweep_environment(tmp_path):
    # Exercise the real shell loops and CSV generator without loading a GPU model.
    interpreter = tmp_path / "fake_python"
    interpreter.write_text(f"#!{sys.executable}\n" + textwrap.dedent('''\
        import json
        import os
        from pathlib import Path
        import signal
        import sys
        import time

        root = Path(os.environ["TEST_SWEEP_ROOT"])
        args = sys.argv[1:]
        with (root / "calls.jsonl").open("a") as file:
            file.write(json.dumps(args) + "\\n")
        if args[0] == "-":
            source = sys.stdin.read()
            compile(source, "<sweep helper>", "exec")
            if "import csv" in source:
                sys.argv = args
                exec(source)
            elif "horizon_path =" in source:
                Path(args[3]).write_text(os.environ.get("TEST_HORIZON", "10") + "\\n")
            elif "deadline =" in source:
                deadline = time.monotonic() + 5
                while not (root / "started").exists():
                    if time.monotonic() > deadline:
                        raise SystemExit("fake server did not start")
                    time.sleep(.01)
            raise SystemExit(0)
        module = args[args.index("-m") + 1]
        if module == "robort.server":
            def stop(*_):
                (root / "stopped").touch()
                raise SystemExit(0)
            signal.signal(signal.SIGTERM, stop)
            (root / "started").touch()
            while True:
                time.sleep(1)
        assert module == "benchmark.libero"
        assert "--args.task-id" not in args
        if os.environ.get("TEST_FAIL_CLIENT"):
            raise SystemExit(23)
        def option(name):
            return args[args.index("--args." + name) + 1]
        suite = option("task-suite-name")
        trials = int(option("num-trials-per-task"))
        count = 90 if suite == "libero_90" else 10
        tasks = [{"task_id": i, "task": f"task {i}", "episodes": trials,
                  "successes": trials if i % 2 == 0 else 0,
                  "success_rate": 1. if i % 2 == 0 else 0.} for i in range(count)]
        data = {"suite": suite, "replan_steps": int(option("replan-steps")),
                "inference_delay": int(option("inference-delay")),
                "use_rtc": "--args.use-rtc" in args, "seed": int(option("seed")),
                "episodes": count * trials, "successes": count * trials // 2,
                "success_rate": .5, "tasks": tasks}
        Path(option("results-path")).write_text(json.dumps(data))
    '''))
    interpreter.chmod(0o755)
    env = dict(os.environ, POLICY_PYTHON=str(interpreter), SIM_PYTHON=str(interpreter),
               TEST_SWEEP_ROOT=str(tmp_path), OUTPUT_ROOT=str(tmp_path / "results"),
               GPU_ID="0", POLICY_PORT="8999", TRIALS="1", PLAN_STEPS="1 3",
               INFERENCE_DELAYS="0 2", USE_RTC="false true", SEED="9",
               MAX_PARALLEL_TASKS="2", MAX_BATCH_SIZE="2")
    env.pop("SUITES", None)
    env.pop("SERVER_LOG", None)
    return env


def run_sweep(env, tmp_path):
    return subprocess.run(["bash", str(SCRIPT)], cwd=tmp_path, env=env,
                          capture_output=True, text=True, timeout=30)


def test_sweep_cartesian_product_and_reports(sweep_environment, tmp_path):
    result = run_sweep(sweep_environment, tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    suites = ["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"]
    calls = [json.loads(line) for line in (tmp_path / "calls.jsonl").read_text().splitlines()]
    servers = [args for args in calls if "robort.server" in args]
    clients = [args for args in calls if "benchmark.libero" in args]
    assert len(servers) == 1
    assert len(clients) == 40
    assert servers[0][servers[0].index("--policy-config.max-batch-size") + 1] == "2"
    for args in clients:
        assert "--args.task-id" not in args
        assert args[args.index("--args.max-parallel-tasks") + 1] == "2"
        assert "--args.fail-on-error" in args
    assert ["-", "127.0.0.1", "8999", str(tmp_path / "results/action_horizon.txt"), "1 3", "0 2", "false", "true"] in calls
    output = tmp_path / "results"
    rows = list(csv.DictReader((output / "success_rate_profile.csv").open()))
    combinations = {(row["suite"], int(row["plan_steps"]), int(row["inference_delay"]),
                     row["use_rtc"]) for row in rows}
    assert combinations == set(itertools.product(suites, [1, 3], [0, 2], ["False", "True"]))
    assert all(row["seed"] == "9" and row["success_percent"] == "50.0" for row in rows)
    tasks = list(csv.DictReader((output / "per_task.csv").open()))
    assert len(tasks) == (4 * 10 + 90) * 8
    assert len(list(output.glob("*/steps_*/delay_*/rtc_*/seed_*/results.json"))) == 40
    assert (tmp_path / "stopped").exists()


def test_client_failure_stops_server_and_preserves_exit_status(sweep_environment, tmp_path):
    sweep_environment["TEST_FAIL_CLIENT"] = "1"
    result = run_sweep(sweep_environment, tmp_path)
    assert result.returncode == 23, result.stdout + result.stderr
    assert (tmp_path / "stopped").exists()
    assert not (tmp_path / "results/success_rate_profile.csv").exists()


def test_existing_run_is_not_overwritten(sweep_environment, tmp_path):
    run = tmp_path / "results/libero_spatial/steps_1/delay_0/rtc_false/seed_9"
    run.mkdir(parents=True)
    saved = run / "results.json"
    saved.write_text("preserve this result")
    result = run_sweep(sweep_environment, tmp_path)
    assert result.returncode == 1
    assert "Refusing to overwrite" in result.stderr
    assert saved.read_text() == "preserve this result"
    assert (tmp_path / "stopped").exists()


@pytest.mark.parametrize("key,value", [("PLAN_STEPS", "0"), ("INFERENCE_DELAYS", "-1"),
                                       ("USE_RTC", "maybe"), ("SUITES", "unknown")])
def test_invalid_sweep_fails_before_starting_server(sweep_environment, tmp_path, key, value):
    sweep_environment[key] = value
    result = run_sweep(sweep_environment, tmp_path)
    assert result.returncode != 0
    assert "Invalid" in result.stderr
    assert not (tmp_path / "started").exists()


def test_unsupported_pairs_are_skipped(sweep_environment, tmp_path):
    sweep_environment.update(PLAN_STEPS="1 10", INFERENCE_DELAYS="0 4", SUITES="libero_spatial")
    result = run_sweep(sweep_environment, tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Skipping plan_steps=10, inference_delay=4" in result.stdout
    rows = list(csv.DictReader((tmp_path / "results/success_rate_profile.csv").open()))
    assert len(rows) == 6
    assert "6/6 runs" in result.stdout


def test_all_pairs_unsupported(sweep_environment, tmp_path):
    sweep_environment.update(PLAN_STEPS="10", INFERENCE_DELAYS="4")
    result = run_sweep(sweep_environment, tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "No supported combinations" in result.stdout
    assert (tmp_path / "stopped").exists()
