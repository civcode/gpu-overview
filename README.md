# System + GPU Overview

A lightweight, dependency-free Linux command-line app that combines CPU and RAM data from `top` with `nvidia-smi` and `amd-smi` into a single view. Designed around **NVIDIA-SMI 580.178.04** and **AMD-SMI 27.1.0** output, as provided in the sample.

## Requirements

- Linux, Python 3.10+.
- `top` (procps/procps-ng) for CPU and RAM statistics.
- NVIDIA drivers with `nvidia-smi` and/or AMD-SMI with `amd-smi` available in `PATH`.
- No `pip install` required. One GPU vendor can be absent; the available vendor is shown.

## Run

```bash
python3 gpu_overview.py
```

Refresh the dashboard every two seconds:

```bash
python3 gpu_overview.py --watch 2
```

Show five snapshots then exit:

```bash
python3 gpu_overview.py --watch 2 --count 5
```

Export a machine-readable snapshot:

```bash
python3 gpu_overview.py --json > snapshot.json
```

Stream newline-separated JSON snapshots:

```bash
python3 gpu_overview.py --watch 2 --json >> gpu-history.jsonl
```

## Try the included data without GPUs

```bash
python3 gpu_overview.py --demo
```

The `--demo` mode includes **illustrative** CPU and RAM values in `examples/top.txt`; these were not measured on your host.

You can test your own captured raw outputs:

```bash
nvidia-smi > nvidia.txt
amd-smi > amd.txt
top -b -n 2 -d 0.2 > top.txt
python3 gpu_overview.py --nvidia-file nvidia.txt --amd-file amd.txt --top-file top.txt
```

To test a layout where one vendor isn't available, use `--nvidia-file` or `--amd-file` for just that vendor. The other will be run live or reported as missing.

## Metrics shown

- **CPU non-idle %** from the second `top` snapshot (`100 - %id`) with a utilization bar. This is not Linux's load average and includes I/O wait.
- **RAM used / total** and percent bar from the `top` physical-memory summary (excluding the `buff/cache` portion reported by `top`; RAM is separate from GPU VRAM).
- GPU index is **vendor-qualified**, e.g. `NV0`, `AMD0`, `AMD1`.
- Name, utilization, temperature, fan, power use / cap.
- Per-GPU memory used / total, relative utilization and bar chart.
- Overall dedicated VRAM allocated, **excluding an integrated Radeon Graphics shared-memory pool**.
- Processes grouped by PID, with per-device memory allocations. A `llama-server` PID appearing on three GPUs is grouped as one process.
- `--json` output for automation.

**Important interpretation notes:**

- `top -b -n 2 -d 0.2` samples twice with a 0.2-second interval. This avoids relying on the first CPU reading, which can reflect time since boot. Each refresh invokes the tool again.
- CPU or RAM statistics show `unavailable` if `top` cannot run or its summary is unrecognized; the GPU overview still works, with a warning.
- For locales with decimal commas, the application sets `LC_ALL=C` for command outputs so its parsers see decimal points.


- The displayed VRAM sum is an **accounting total** across discrete devices, *not* one memory pool. Actual model offload limits depend on the distribution of tensors/weights across GPUs.
- NVIDIA's MiB and AMD's MB/GB columns are normalized to MiB using the byte-scale convention commonly used in CLI GPU reporting (`1 GB = 1024 MiB`). As tools evolve, confirm units for your installed release if byte-level accuracy is essential.
- The snapshot process memory may not add up exactly to the GPU-level memory allocations; driver reservations, graphics workloads and accounting differ.
- NVIDIA and AMD utilization percentages come from separate tools, so they are **not synchronized** to precisely the same sampling instant.
- The application deliberately parses the *human-readable* output matching your example. New versions or alternative table layouts may require parser adjustments. The native tool JSON interfaces or direct AMD-SMI/NVML libraries are a natural next step for broader compatibility.
- `amd-smi` may return incomplete process names due to permissions; try running each vendor tool directly to diagnose access problems.

## Run tests

```bash
python3 -m unittest discover -s tests -v
```

## Files

- `gpu_overview.py` — application; no extra dependencies.
- `examples/nvidia-smi.txt`, `examples/amd-smi.txt`, `examples/top.txt` — captured GPU data and illustrative system data.
- `tests/test_gpu_overview.py` — parser and output tests.


## Rendering note

This version uses bracketed ASCII bars (`[====....]`) instead of Unicode block characters to improve compatibility with terminal fonts and line-height rendering.
