from armory.backends.registry import OpenPiPolicyFactory, Gr00tPolicyFactory
from dataclasses import replace

from armory.backends.types import ServingPolicy, EnvMode, ModelFamily, warmup_request
from armory.serving.rtc import InferType, RTCParams
from armory.serving.schemas import SlotData

from robort.schemas import InferenceRequest, PolicyConfig, InferenceResponse


class Policy:
    '''
    Policy adapted from Armory's backend
    '''
    def __init__(self, policy_config: PolicyConfig):
        self.config = policy_config
        self._batch_sizes = sorted(set(policy_config.batch_sizes or [policy_config.max_batch_size]))
        if (policy_config.max_batch_size < 1
                or any(size < 1 or size > policy_config.max_batch_size for size in self._batch_sizes)
                or self._batch_sizes[-1] != policy_config.max_batch_size):
            raise ValueError("batch_sizes must be positive, not exceed max_batch_size, and include max_batch_size")
        env = EnvMode(policy_config.env_name)
        if policy_config.model_name == ModelFamily.GROOT_N17.value:
            factory = Gr00tPolicyFactory(
                policy_config.model_name, env, policy_config.model_dir
            )
        else:
            factory = OpenPiPolicyFactory(
                policy_config.model_name, policy_config.model_dir,
                policy_config.num_sample_steps, env,
            )
        self._policy: ServingPolicy = factory()
        if policy_config.model_name == ModelFamily.GROOT_N17.value:
            self._policy.warmup(self.config.max_batch_size)
        else:
            self._policy.warmup(self.config.max_batch_size, batch_sizes=self._batch_sizes)

    def _convert_to_armory(self, req: InferenceRequest) -> SlotData:
        infer_type = (
            InferType.INFERENCE_TIME_RTC
            if req.inference_type == "rtc" else InferType(req.inference_type)
        )
        params = None
        if infer_type == InferType.INFERENCE_TIME_RTC and any(
            getattr(self._policy, flag, False) is True
            for flag in ("_is_pytorch_model", "_is_triton_optimized")
        ):
            raise ValueError("RTC requires the OpenPI JAX backend; this backend ignores RTC conditioning")
        if infer_type == InferType.INFERENCE_TIME_RTC and req.previous_action is not None:
            params = RTCParams(req.previous_action, req.rtc_s_param, req.rtc_d_param)
        return replace(
            warmup_request(req.observation, infer_type, params),
            max_execution_horizon=req.max_execution_horizon,
        )

    def infer_batch(self, batch: list[InferenceRequest]) -> list[InferenceResponse]:
        if not batch:
            return []
        if len(batch) > self.config.max_batch_size:
            raise ValueError("Batch exceeds max_batch_size")
        armory_batch = [self._convert_to_armory(req) for req in batch]
        # Match Armory's split: an RTC request without previous actions uses SYNC.
        groups: dict[bool, list[int]] = {False: [], True: []}
        for index, req in enumerate(armory_batch):
            use_rtc = req.infer_type == InferType.INFERENCE_TIME_RTC and isinstance(req.params, RTCParams)
            groups[use_rtc].append(index)

        responses: list[InferenceResponse | None] = [None] * len(batch)
        for indices in groups.values():
            if not indices:
                continue
            size = next(size for size in self._batch_sizes if size >= len(indices))
            requests = [armory_batch[index] for index in indices]
            # Copy a compatible request, retaining model-space RTC shape and dtype.
            requests.extend(replace(requests[0]) for _ in range(size - len(requests)))
            results = self._policy.infer_batch(requests)
            if len(results) != size:
                raise ValueError(f"Backend returned {len(results)} responses for {size} requests")
            for index, result in zip(indices, results[:len(indices)], strict=True):
                responses[index] = InferenceResponse(
                    actions=result["actions"], rtc_prev_actions=result.get("rtc_prev_actions")
                )
        return [response for response in responses if response is not None]



def create_policy(config: PolicyConfig) -> Policy:
    return Policy(config)
