1. Serve the policy module 

```bash
PYTHONPATH=src:. XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=0 \
.venv-openpi/bin/python -m benchmark.serve \
  --port 8020 \
  --config-name pi05_conveyor_joint \
  --model-dir checkpoints/pi05_libero_pytorch
```

If successful, you will see `pi05 ready on ws://127.0.0.1:8020`

2. Run the inference benchmark 

```bash
PYTHONPATH=src:. CUDA_VISIBLE_DEVICES=1 \
  ~/.local/miniconda3/envs/reflexbench/bin/python -u -m benchmark.bench \
  --config benchmark/configs/conveyor.yaml
```



