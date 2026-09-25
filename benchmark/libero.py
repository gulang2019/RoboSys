import collections
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, nullcontext
import dataclasses
import logging
import json
import math
import pathlib
from typing import Optional

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
import tqdm
import tyro

from robort.schemas import InferenceRequest
from benchmark.client import LocalClientPolicy, WebsocketClientPolicy, parse_client_args

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data
ACTION_HORIZON=10


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5  # Request a new chunk every N simulation control steps.
    inference_delay: int = 0  # Delay each chunk's delivery by N control steps.
    use_rtc: bool = False

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_spatial"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    task_id: int | None = None
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 25  # Number of rollouts per task
    max_parallel_tasks: int = 20
    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/libero/videos"  # Path to save videos

    seed: int = 7  # Random Seed (for reproducibility)
    results_path: Optional[str] = None
    fail_on_error: bool = False


def eval_one_task(
    args: Args, task_suite, max_steps: int, task_id: int, client=None, client_options=None
):
    if args.replan_steps < 1 or args.inference_delay < 0:
        raise ValueError("replan_steps must be positive and inference_delay must be nonnegative")
    # Get task
    task = task_suite.get_task(task_id)

    # Get default LIBERO initial states
    initial_states = task_suite.get_task_init_states(task_id)

    # Initialize LIBERO environment and task description
    env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

    with closing(env), (
        nullcontext(client) if client is not None else closing(
            WebsocketClientPolicy(**{"host": args.host, "port": args.port, **client_options})
            if client_options else WebsocketClientPolicy(args.host, args.port)
        )
    ) as client:
        # Start episodes
        task_episodes, task_successes = 0, 0

        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):


            # Reset environment
            env.reset()
            client.reset()
            action_plan = collections.deque()
            pending_plans = collections.deque()
            previous_action = None

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            done = False
            replay_images = []

            while t < max_steps + args.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    # Get preprocessed image
                    # IMPORTANT: rotate 180 degrees to match train preprocessing
                    # Matches Armory's evaluation/envs/libero.py preprocessing.
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    )

                    # Save preprocessed image for replay video
                    replay_images.append(img)

                    control_step = t - args.num_steps_wait
                    if control_step % args.replan_steps == 0:
                        # Pause simulation during inference, then simulate its delivery delay.
                        # Prepare observations dict

                        req = InferenceRequest(
                            observation = {
                                "observation/image": img,
                                "observation/wrist_image": wrist_img,
                                "observation/state": np.concatenate(
                                    (
                                        obs["robot0_eef_pos"],
                                        _quat2axisangle(obs["robot0_eef_quat"]),
                                        obs["robot0_gripper_qpos"],
                                    )
                                ),
                                "prompt": str(task_description),
                                "episode": int(episode_idx),
                                "multi_actions": 0,
                            },
                            inference_type="rtc" if args.use_rtc else "sync",
                            previous_action=previous_action if args.use_rtc else None,
                            rtc_s_param=args.replan_steps,
                            rtc_d_param=args.inference_delay,
                            max_execution_horizon=args.replan_steps + args.inference_delay,
                        )

                        # Query model and execute a fixed-length action chunk.
                        server_outputs = client.infer(req)
                        action_chunk = server_outputs.actions
                        required = args.replan_steps + args.inference_delay
                        if len(action_chunk) < required:
                            raise ValueError(
                                f"Policy predicts {len(action_chunk)} steps; need at least {required} "
                                "(replan_steps + inference_delay)."
                            )
                        previous_action = server_outputs.rtc_prev_actions
                        if args.use_rtc and previous_action is None:
                            raise ValueError("RTC requires model-space rtc_prev_actions from the server")
                        pending_plans.append((
                            control_step + args.inference_delay,
                            action_chunk[args.inference_delay:required],
                        ))

                    if pending_plans and pending_plans[0][0] <= control_step:
                        _, ready_actions = pending_plans.popleft()
                        action_plan = collections.deque(ready_actions)
                    # Continue the old plan while waiting; idle before the first delivery.
                    action = action_plan.popleft() if action_plan else np.asarray(LIBERO_DUMMY_ACTION)

                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        break
                    t += 1

                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    if args.fail_on_error:
                        raise
                    break

            task_episodes += 1

            # Save a replay video of the episode
            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")
            video_basename = f"task{task_id}_ep{episode_idx}_{task_segment}_{suffix}"
            imageio.mimwrite(
                pathlib.Path(args.video_out_path) / f"{video_basename}.mp4",
                [np.asarray(x) for x in replay_images],
                fps=10,
            )

        return {
            "task_id": task_id,
            "task": task_description,
            "episodes": task_episodes,
            "successes": task_successes,
            "success_rate": task_successes / task_episodes,
        }


def eval_libero(args: Args, client_args: str = "{}") -> None:
    """Evaluate with --client-args '{"type": "local", "device": "cuda"}'."""
    _eval_libero(args, client_args)


def _eval_libero(args: Args, client_args: str, client=None) -> None:
    client_type, client_options = parse_client_args(client_args)
    if client_type == "local" and args.use_rtc:
        raise ValueError("Local OpenPI client only supports sync requests; disable RTC")
    if args.inference_delay > ACTION_HORIZON or \
        args.replan_steps > ACTION_HORIZON or \
        args.inference_delay + args.replan_steps > ACTION_HORIZON:
        logging.warning("invalid delay or replan steps")
        exit(0)
    # Set random seed
    np.random.seed(args.seed)
    if client_type == 'local':
        import torch

        torch.manual_seed(args.seed)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    if args.task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    # Start evaluation
    total_episodes, total_successes = 0, 0
    task_results = []

    borrowed_client = client is not None
    if client_type == "local":
        logging.info("Local OpenPI evaluation runs tasks sequentially with one shared model")
        task_ids = [args.task_id] if args.task_id is not None else range(num_tasks_in_suite)
        with (nullcontext(client) if client is not None else closing(LocalClientPolicy(**client_options))) as client:
            task_results = [
                eval_one_task(args, task_suite, max_steps, task_id, client=client)
                for task_id in tqdm.tqdm(task_ids)
            ]
            if borrowed_client:
                client.policy.export_debug_videos()
    else:
        kwargs = {"client_options": client_options} if client_options else {}
        if args.task_id is not None:
            task_results = [eval_one_task(args, task_suite, max_steps, args.task_id, **kwargs)]
        else:
            with ThreadPoolExecutor(max_workers=args.max_parallel_tasks) as executor:
                task_results = list(tqdm.tqdm(
                    executor.map(
                        lambda task_id: eval_one_task(args, task_suite, max_steps, task_id, **kwargs),
                        range(num_tasks_in_suite),
                    ),
                    total=num_tasks_in_suite,
                ))
    total_episodes = sum(result['episodes'] for result in task_results)
    total_successes = sum(result['successes'] for result in task_results)

    logging.info(f"{args.task_suite_name}, Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")
    if args.results_path:
        results_path = pathlib.Path(args.results_path)
        results_path.parent.mkdir(parents=True, exist_ok=True)
        results_path.write_text(json.dumps({
            "suite": args.task_suite_name,
            "replan_steps": args.replan_steps,
            "inference_delay": args.inference_delay,
            "use_rtc": args.use_rtc,
            "seed": args.seed,
            "episodes": total_episodes,
            "successes": total_successes,
            "success_rate": total_successes / total_episodes,
            "tasks": task_results,
        }, indent=2) + "\n")


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)
