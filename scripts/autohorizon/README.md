# Native AutoHorizon setup

From a clone of RoboSys, run as a regular user:

```bash
bash setup_autohorizon.sh
```

The setup targets Ubuntu 24.04 with an installed NVIDIA driver. It uses sudo
for system build/rendering dependencies, initializes AutoHorizon, clones LIBERO
at a pinned revision, and installs uv if unavailable. It creates separate
Python 3.11 (policy) and Python 3.8 (simulator) environments.

It applies the evaluator results patch, aligns robosuite versions, constrains
Python 3.8 build-time setuptools, and patches Transformers before converting
the checkpoint. It downloads public JAX weights and the tokenizer, converts to
`checkpoints/pi05_libero_pytorch_fixed`, copies normalization assets, checks
initial states, and performs one model inference. Downloads and conversion
require substantial disk space and memory. No demonstration dataset is needed.

The final output gives a command for the sweep, which starts its own server
and stops it on completion or failure. Stop any existing server on the selected
port first. Keep the printed `LIBERO_CONFIG_PATH` for the sweep: it
uses a local config and avoids LIBERO's interactive first-import prompt.

Defaults preserve the current experiment: prediction horizon 10, execution
horizons 1, 4, 7, 10, four suites, 10 trials per task, and one seed (7).
This is not the full 50-step Figure 1 experiment. The sweep verifies the actual
server output before evaluation. See `bash setup_autohorizon.sh --help` for
setup overrides, and the sweep script for `TRIALS`, `SEED`, `SUITES`, and
`PLAN_STEPS` overrides.

Rerunning setup syncs dependencies and reapplies the Transformers patch. It
preserves an existing sweep script and reuses validated checkpoint files.
It stops on an incompatible evaluator patch or converted checkpoint instead
of silently overwriting them. Stop the policy server before rerunning setup
so its environment is not changed while it is running.

Keep this directory and the root setup script in the RoboSys repository:
the AutoHorizon submodule alone does not include these sweep/result changes.
