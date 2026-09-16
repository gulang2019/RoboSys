from time import perf_counter

from robort.profile.schemas import HardwareConfig, HardwareProfile
from robort.profile.environment import power_draw_w, prepare_hardware_env, validate_hardware_config


def profile_hardware(hardware_config: HardwareConfig) -> HardwareProfile:
    """Measure the current CUDA device using decimal GB and per-dtype compute throughput.

    Power control and a current green-context stream cover all benchmarks.
    Device metadata resides in hardware_config. Unsupported
    power telemetry and unsupported compute types are reported as NaN; peer
    bandwidth is None without a peer. INT8 reports GOPS in gflops_int8. FP8 uses
    E4M3 with FP16 output; INT8 uses INT32 output. Each multiply-add counts as
    two operations. FP32 disables TF32; the caller's setting is restored.
    """
    import torch

    validate_hardware_config(hardware_config)
    if not torch.cuda.is_available():
        raise RuntimeError("Hardware profiling requires a CUDA device")

    device = hardware_config._device_index
    def benchmark(operation, devices=(device,)):
        for _ in range(3):
            operation()
        stream.synchronize()
        for index in devices:
            if index != device:
                torch.cuda.synchronize(index)
        start = perf_counter()
        for _ in range(10):
            operation()
        stream.synchronize()
        for index in devices:
            if index != device:
                torch.cuda.synchronize(index)
        seconds = (perf_counter() - start) / 10
        return seconds

    with prepare_hardware_env(hardware_config) as stream, torch.inference_mode():
        stream.synchronize()
        idle_power = power_draw_w(torch, device)
        free_bytes, _ = torch.cuda.mem_get_info(device)
        # Bound temporary allocations on small or already occupied GPUs.
        size = min(64 * 1024**2, free_bytes // 8)
        if size < 4:
            raise RuntimeError("Insufficient free CUDA memory for profiling")
        source = torch.empty(size, dtype=torch.uint8, device=device)
        source.zero_()
        target = torch.empty_like(source)
        host = torch.empty(size, dtype=torch.uint8, pin_memory=True)
        host.zero_()
        mem_bw = 2 * size / benchmark(lambda: target.copy_(source)) / 1e9
        c2g_bw = size / benchmark(lambda: target.copy_(host, non_blocking=True)) / 1e9
        g2c_bw = size / benchmark(lambda: host.copy_(source, non_blocking=True)) / 1e9

        g2g_bw = None
        for peer in range(torch.cuda.device_count()):
            if peer == device or not torch.cuda.can_device_access_peer(device, peer):
                continue
            peer_free, _ = torch.cuda.mem_get_info(peer)
            peer_size = min(size, peer_free // 8)
            if peer_size < 1:
                continue
            peer_target = torch.empty(peer_size, dtype=torch.uint8, device=peer)
            peer_source = source[:peer_size]
            g2g_bw = peer_size / benchmark(
                lambda: peer_target.copy_(peer_source, non_blocking=True),
                (device, peer),
            ) / 1e9
            del peer_target, peer_source
            break
        del source, target, host

        # Keep dimensions aligned for integer and FP8 CUDA kernels.
        dimension = min(2048, int((free_bytes // 48) ** 0.5)) // 16 * 16
        if dimension < 16:
            raise RuntimeError("Insufficient free CUDA memory for profiling")
        shape = (dimension, dimension)
        throughput = {}
        allow_tf32 = torch.backends.cuda.matmul.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            with torch.autocast(device_type="cuda", enabled=False):
                for name, dtype in (("fp32", torch.float32),
                                    ("fp16", torch.float16),
                                    ("bf16", torch.bfloat16)):
                    if name == "bf16" and not torch.cuda.is_bf16_supported():
                        throughput[name] = float("nan")
                        continue
                    a = torch.randn(shape, device=device, dtype=dtype)
                    b = torch.randn_like(a)
                    out = torch.empty_like(a)
                    seconds = benchmark(lambda: torch.mm(a, b, out=out))
                    throughput[name] = 2 * dimension**3 / seconds / 1e9
                    del a, b, out

                for name in ("int8", "fp8"):
                    throughput[name] = float("nan")
                    helper = getattr(torch, "_int_mm" if name == "int8" else "_scaled_mm", None)
                    if helper is None:
                        continue
                    if name == "fp8" and (
                        not hasattr(torch, "float8_e4m3fn")
                        or torch.cuda.get_device_capability(device) < (8, 9)
                    ):
                        continue
                    if name == "int8":
                        a = torch.randint(-8, 8, shape, device=device, dtype=torch.int8)
                        b = torch.randint(-8, 8, shape, device=device, dtype=torch.int8)
                        out = torch.empty(shape, device=device, dtype=torch.int32)
                        operation = lambda: helper(a, b, out=out)
                    else:
                        a = torch.randn(shape, device=device, dtype=torch.float16).to(torch.float8_e4m3fn)
                        # cuBLAS FP8 requires a column-major second operand.
                        b = torch.randn(shape, device=device, dtype=torch.float16).to(torch.float8_e4m3fn).t()
                        scale = torch.ones((), device=device, dtype=torch.float32)
                        out = torch.empty(shape, device=device, dtype=torch.float16)
                        operation = lambda: helper(a, b, scale, scale, out_dtype=torch.float16, out=out)
                    try:
                        seconds = benchmark(operation)
                        throughput[name] = 2 * dimension**3 / seconds / 1e9
                    except NotImplementedError:
                        pass
                    except RuntimeError as error:
                        if not any(message in str(error).lower() for message in
                                   ("not supported", "unsupported", "not implemented")):
                            raise
                    finally:
                        del a, b, out, operation
        finally:
            torch.backends.cuda.matmul.allow_tf32 = allow_tf32

    return HardwareProfile(
        hardware_config=hardware_config,
        idle_power_w=idle_power,
        mem_bw_gbs=mem_bw,
        **{f"gflops_{name}": value for name, value in throughput.items()},
        c2g_bw_gbs=c2g_bw,
        g2c_bw_gbs=g2c_bw,
        g2g_bw_gbs=g2g_bw,
    )
