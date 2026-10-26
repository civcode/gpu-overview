#!/usr/bin/env python3
"""Unified CPU, RAM, NVIDIA and AMD GPU status; Python standard library only."""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time


@dataclass
class GPU:
    vendor: str
    index: int
    name: str
    bus_id: str = ""
    gpu_util_pct: float | None = None
    temperature_c: float | None = None
    fan_pct: float | None = None
    power_w: float | None = None
    power_cap_w: float | None = None
    vram_used_mib: float | None = None
    vram_total_mib: float | None = None
    shared_memory: bool = False

    @property
    def label(self) -> str:
        return f"{'NV' if self.vendor == 'NVIDIA' else 'AMD'}{self.index}"


@dataclass
class Process:
    pid: int
    name: str
    gpu: str
    vram_mib: float | None


def num(value: str | None) -> float | None:
    if value is None or str(value).strip().upper() in {"N/A", "NA", "", "-"}:
        return None
    try:
        return float(str(value).replace(",", ""))
    except ValueError:
        return None


def memory_to_mib(value: str, unit: str) -> float:
    # AMD-SMI labels byte-scaled quantities MB/GB even when they are
    # traditionally displayed in powers of 1024. Normalize to MiB.
    return float(value) * {"B": 1 / 1048576, "KB": 1 / 1024,
                           "MB": 1, "GB": 1024, "MiB": 1, "GiB": 1024}[unit]


def parse_nvidia(output: str) -> tuple[list[GPU], list[Process], str | None]:
    gpus: list[GPU] = []
    procs: list[Process] = []
    driver_match = re.search(r"Driver Version:\s*([\d.]+)", output)
    driver = driver_match.group(1) if driver_match else None
    in_processes = False
    lines = output.splitlines()
    bdf = r"[\da-fA-F]{4,8}:[\da-fA-F]{2}:[\da-fA-F]{2}\.[0-7]"
    for i, line in enumerate(lines):
        if "| Processes:" in line:
            in_processes = True
        if not in_processes:
            # GPU identity line: '| 0 NVIDIA ... Off | 00000000:04:00.0 Off | ... |'
            head = re.match(r"^\|\s*(\d+)\s+(.+?)\s{2,}(?:On|Off|N/A)\s*\|", line)
            bus_match = re.search(bdf, line)
            if head and bus_match:
                gpu = GPU("NVIDIA", int(head.group(1)), head.group(2).strip(), bus_match.group())
                for metrics in lines[i + 1:i + 4]:
                    mem_match = re.search(r"([\d.]+)\s*MiB\s*/\s*([\d.]+)\s*MiB", metrics)
                    if mem_match:
                        gpu.vram_used_mib, gpu.vram_total_mib = map(float, mem_match.groups())
                        parts = metrics.split("|")
                        left = parts[1] if len(parts) > 1 else metrics
                        right = parts[3] if len(parts) > 3 else ""
                        temp = re.search(r"([\d.]+)\s*C\b", left)
                        fan = re.search(r"([\d.]+)\s*%", left)
                        watts = re.search(r"([\d.]+)\s*W\s*/\s*([\d.]+)\s*W", left)
                        util = re.search(r"([\d.]+)\s*%", right)
                        gpu.temperature_c = num(temp.group(1)) if temp else None
                        gpu.fan_pct = num(fan.group(1)) if fan else None
                        gpu.gpu_util_pct = num(util.group(1)) if util else None
                        if watts:
                            gpu.power_w, gpu.power_cap_w = map(float, watts.groups())
                        break
                gpus.append(gpu)
        else:
            # No assumptions about GI/CI values or whether this is a compute/graphics process.
            proc = re.match(r"^\|\s*(\d+)\s+(?:N/A|\d+)\s+(?:N/A|\d+)\s+(\d+)\s+\S+\s+(.+?)\s+([\d.]+)\s*MiB\s*\|", line)
            if proc:
                gpu, pid, name, mem = proc.groups()
                procs.append(Process(int(pid), name.strip(), f"NV{gpu}", float(mem)))
    return gpus, procs, driver


def parse_amd(output: str) -> tuple[list[GPU], list[Process], str | None]:
    gpus: list[GPU] = []
    procs: list[Process] = []
    rocm_match = re.search(r"ROCm Version:\s*([\d.]+)", output)
    rocm = rocm_match.group(1) if rocm_match else None
    in_processes = False
    bdf = r"[\da-fA-F]{4,8}:[\da-fA-F]{2}:[\da-fA-F]{2}\.[0-7]"
    pending: GPU | None = None
    for line in output.splitlines():
        if "| Processes:" in line:
            in_processes = True
        if not in_processes:
            first = re.match(rf"^\|\s*({bdf})\s+(.+?)\s*\|\s*(.*?)\s*\|", line)
            if first:
                bus, name, metrics = first.groups()
                pending = GPU("AMD", -1, name.strip(), bus)
                temperature = re.search(r"([\d.]+)\s*°?\s*C", metrics)
                watts = re.search(r"([\d.]+)\s*/\s*([\d.]+)\s*W", metrics)
                pending.temperature_c = num(temperature.group(1)) if temperature else None
                if watts:
                    pending.power_w, pending.power_cap_w = map(float, watts.groups())
                continue
            if pending:
                second = re.match(r"^\|\s*(\d+)\s+(\S+)\s+(\S+)\s+(.+?)\s*\|\s*(.*?)\s*\|", line)
                if second:
                    index, _hip, _oam, _part, metrics = second.groups()
                    pending.index = int(index)
                    usages = re.findall(r"([\d.]+)\s*%", metrics)
                    if usages:
                        pending.gpu_util_pct = num(usages[0])
                    if len(usages) > 1:
                        pending.fan_pct = num(usages[1])
                    elif usages:
                        # Some AMD-SMI releases omit the second percent sign.
                        fan = re.search(r"^\s*[\d.]+\s*%\s+([\d.]+)(?:\s*%|(?=\s+[\d.]+\s*/))", metrics)
                        if fan:
                            pending.fan_pct = num(fan.group(1))
                    mem = re.search(r"([\d.]+)\s*/\s*([\d.]+)\s*(MB|GB|MiB|GiB)", metrics)
                    if mem:
                        used, total, unit = mem.groups()
                        pending.vram_used_mib = memory_to_mib(used, unit)
                        pending.vram_total_mib = memory_to_mib(total, unit)
                    # The integrated Radeon Graphics reports a shared memory pool.
                    pending.shared_memory = bool(re.search(r"\bRadeon Graphics\b", pending.name, re.I))
                    gpus.append(pending)
                    pending = None
        else:
            # GPU PID process-name GTT memory VRAM memory MEM_USAGE CU% SDMA
            proc = re.match(
                r"^\|\s*(\d+)\s+(\d+)\s+(\S+)\s+"
                r"[\d.]+\s+(?:B|KB|MB|GB|MiB|GiB)\s+"
                r"([\d.]+)\s+(B|KB|MB|GB|MiB|GiB)\s+"
                r"[\d.]+\s+(?:B|KB|MB|GB|MiB|GiB)\s+", line,
            )
            if proc:
                gpu, pid, name, vram, unit = proc.groups()
                procs.append(Process(int(pid), name, f"AMD{gpu}", memory_to_mib(vram, unit)))
    return gpus, procs, rocm



def parse_top(output: str) -> dict[str, float | None]:
    """Read overall CPU non-idle percentage and physical RAM from procps top.

    top is run with two iterations, because the first CPU summary can represent
    activity since boot instead of an interval. The last matching values win.
    The 'used' RAM field excludes the displayed free and buff/cache amounts.
    """
    metrics: dict[str, float | None] = {
        "cpu_busy_pct": None,
        "ram_used_mib": None,
        "ram_total_mib": None,
    }
    factors = {"KiB": 1 / 1024, "MiB": 1, "GiB": 1024, "TiB": 1024 * 1024,
               "kB": 1000 / (1024 * 1024), "MB": 1000 * 1000 / (1024 * 1024),
               "GB": 1000 ** 3 / (1024 ** 2)}
    for line in output.splitlines():
        if re.match(r"^\s*%?Cpu(?:\(s\))?\s*:", line, re.IGNORECASE):
            idle = re.search(r"([\d.]+)\s*%?\s*id\b", line, re.IGNORECASE)
            if idle:
                metrics["cpu_busy_pct"] = min(100.0, max(0.0, 100.0 - float(idle.group(1))))
        mem = re.match(r"^\s*(KiB|MiB|GiB|TiB|kB|MB|GB)\s+Mem\s*:\s*(.*)", line)
        if mem:
            unit, values = mem.groups()
            total = re.search(r"([\d.]+)\s+total\b", values)
            used = re.search(r"([\d.]+)\s+used\b", values)
            if total and used:
                metrics["ram_total_mib"] = float(total.group(1)) * factors[unit]
                metrics["ram_used_mib"] = float(used.group(1)) * factors[unit]
    return metrics


def run_command(command: list[str]) -> tuple[str, str | None]:
    if shutil.which(command[0]) is None:
        return "", f"{command[0]} not found in PATH"
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=15,
                                check=False, env={**os.environ, "LC_ALL": "C"})
    except (OSError, subprocess.TimeoutExpired) as exc:
        return "", f"{' '.join(command)}: {exc}"
    if result.returncode:
        return "", f"{' '.join(command)}: {result.stderr.strip() or 'failed with exit '+str(result.returncode)}"
    return result.stdout, None


def collect(nvidia_file: Path | None, amd_file: Path | None,
            top_file: Path | None = None) -> dict:
    warnings: list[str] = []
    outputs: dict[str, str] = {}
    for vendor, fixture, command in (("NVIDIA", nvidia_file, ["nvidia-smi"]),
                                      ("AMD", amd_file, ["amd-smi"])):
        if fixture is not None:
            try:
                outputs[vendor] = fixture.read_text(encoding="utf-8")
            except OSError as exc:
                outputs[vendor] = ""
                warnings.append(f"{fixture}: {exc}")
        else:
            outputs[vendor], error = run_command(command)
            if error:
                warnings.append(error)
    if top_file is not None:
        try:
            top_output = top_file.read_text(encoding="utf-8")
        except OSError as exc:
            top_output = ""
            warnings.append(f"{top_file}: {exc}")
    else:
        top_output, error = run_command(["top", "-b", "-n", "2", "-d", "0.2"])
        if error:
            warnings.append(error)
    system = parse_top(top_output)
    if top_output and all(value is None for value in system.values()):
        warnings.append("Could not parse top system summary (format may differ)")

    nvidia, nv_processes, nv_driver = parse_nvidia(outputs["NVIDIA"])
    amd, amd_processes, rocm = parse_amd(outputs["AMD"])
    if outputs["NVIDIA"] and not nvidia:
        warnings.append("Could not parse NVIDIA GPU table (format may differ)")
    if outputs["AMD"] and not amd:
        warnings.append("Could not parse AMD GPU table (format may differ)")
    gpus = nvidia + amd
    procs = nv_processes + amd_processes
    physical = [g for g in gpus if not g.shared_memory and g.vram_total_mib]
    return {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "drivers": {"nvidia": nv_driver, "rocm": rocm},
        "system": system,
        "gpus": [dict(asdict(g), label=g.label) for g in gpus],
        "processes": [asdict(p) for p in procs],
        "summary": {
            "discrete_gpu_count": len(physical),
            "dedicated_vram_used_mib": sum(g.vram_used_mib or 0 for g in physical),
            "dedicated_vram_total_mib": sum(g.vram_total_mib or 0 for g in physical),
        },
        "warnings": warnings,
    }


def fmt_num(value: float | None, suffix: str = "", decimals: int = 0) -> str:
    return "—" if value is None else f"{value:.{decimals}f}{suffix}"


def bar(used: float | None, total: float | None, length: int = 14) -> str:
    """Render a terminal-safe ASCII progress bar.

    Plain ASCII avoids font/rendering issues seen with Unicode block glyphs,
    which can appear taller than a single row in some terminals.
    """
    if used is None or not total or total <= 0:
        return "." * length
    ratio = min(1, max(0, used / total))
    filled = round(ratio * length)
    return "=" * filled + "." * (length - filled)


def draw(snapshot: dict) -> str:
    gpu_rows = snapshot["gpus"]
    summary = snapshot["summary"]
    total, used = summary["dedicated_vram_total_mib"], summary["dedicated_vram_used_mib"]
    version = snapshot["drivers"]
    header = f"SYSTEM + GPU OVERVIEW  •  {snapshot['timestamp']}  •  {len(gpu_rows)} GPU devices"
    if version["nvidia"] or version["rocm"]:
        header += f"  •  NV {version['nvidia'] or '—'} / ROCm {version['rocm'] or '—'}"
    system = snapshot.get("system", {})
    cpu = system.get("cpu_busy_pct")
    ram_used = system.get("ram_used_mib")
    ram_total = system.get("ram_total_mib")
    cpu_text = (f"{cpu:.1f}%" if cpu is not None else "unavailable")
    ram_text = (f"{ram_used / 1024:.1f} / {ram_total / 1024:.1f} GiB  "
                f"{ram_used / ram_total * 100:.1f}%"
                if ram_used is not None and ram_total else "unavailable")
    output = [header, "═" * min(120, max(80, len(header))),
              "SYSTEM (from top)",
              f"CPU non-idle: {cpu_text:>23}  [{bar(cpu, 100, 24)}]",
              f"RAM used:     {ram_text:>23}  [{bar(ram_used, ram_total, 24)}]",
              "", "GPU MEMORY",
              f"Dedicated VRAM allocated: {used / 1024:.1f} / {total / 1024:.1f} GiB  [{bar(used, total, 20)}]  "
              f"{used / total * 100:.0f}%" if total else "Dedicated VRAM: unavailable",
              "(Combined allocations, not a single addressable memory pool)", "",
              f"{'GPU':<5} {'MODEL':<27} {'GPU%':>5} {'TEMP':>6} {'POWER':>11} {'FAN':>5}  {'VRAM USED / TOTAL':<23} {'VRAM%':>5}  MEMORY",
              "─" * 118]
    for g in gpu_rows:
        name = g["name"].lstrip(". ")
        if len(name) > 27:
            name = name[:26] + "…"
        u, t = g["vram_used_mib"], g["vram_total_mib"]
        power = f"{g['power_w']:.0f}/{g['power_cap_w']:.0f}W" if g["power_w"] is not None and g["power_cap_w"] is not None else "—"
        memory = f"{u / 1024:.1f} / {t / 1024:.1f} GiB" if u is not None and t else "—"
        percentage = f"{100 * u / t:.0f}%" if u is not None and t else "—"
        suffix = "shared" if g["shared_memory"] else f"[{bar(u, t, 12)}]"
        output.append(f"{g['label']:<5} {name:<27} {fmt_num(g['gpu_util_pct'], '%'):>5} "
                      f"{fmt_num(g['temperature_c'], '°C'):>6} {power:>11} "
                      f"{fmt_num(g['fan_pct'], '%'):>5}  {memory:<23} {percentage:>5}  {suffix}")
    output.extend(["", "GPU PROCESSES (VRAM reported per GPU)", "─" * 118])
    per_pid: dict[int, list[dict]] = defaultdict(list)
    for proc in snapshot["processes"]:
        per_pid[proc["pid"]].append(proc)
    if not per_pid:
        output.append("No GPU processes reported")
    for pid, processes in sorted(per_pid.items()):
        name = next((p["name"] for p in processes if p["name"]), "?")
        output.append(f"{pid:<7} {name}")
        for p in processes:
            allocated = "—" if p["vram_mib"] is None else f"{p['vram_mib'] / 1024:.2f} GiB"
            output.append(f"{'':7}  └─ {p['gpu']:<5} {allocated:>11}")
    if any(g["shared_memory"] for g in gpu_rows):
        output.extend(["", "* AMD Radeon Graphics uses a shared-memory pool; excluded from dedicated VRAM totals."])
    if snapshot["warnings"]:
        output.extend(["", "WARNINGS", *["- " + msg for msg in snapshot["warnings"]]])
    return "\n".join(output)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Combined top + nvidia-smi + amd-smi overview")
    parser.add_argument("--watch", type=float, default=0, metavar="SECONDS", help="refresh interval; Ctrl+C to quit")
    parser.add_argument("--count", type=int, default=0, help="number of snapshots in watch mode (0=infinite)")
    parser.add_argument("--json", action="store_true", help="JSON snapshot (JSON-lines in watch mode)")
    parser.add_argument("--nvidia-file", type=Path, help="read saved nvidia-smi output instead of running it")
    parser.add_argument("--amd-file", type=Path, help="read saved amd-smi output instead of running it")
    parser.add_argument("--top-file", type=Path, help="read saved top output instead of running it")
    parser.add_argument("--demo", action="store_true", help="use included example from a mixed-GPU host")
    args = parser.parse_args(argv)
    if args.watch < 0 or args.count < 0:
        parser.error("--watch and --count must be >= 0")
    if args.demo:
        base = Path(__file__).resolve().parent
        candidates = [base / "examples", base / "gpu-overview" / "examples"]
        folder = next((p for p in candidates if p.exists()), candidates[0])
        args.nvidia_file = folder / "nvidia-smi.txt"
        args.amd_file = folder / "amd-smi.txt"
        args.top_file = folder / "top.txt"
    count = 0
    try:
        while True:
            snapshot = collect(args.nvidia_file, args.amd_file, args.top_file)
            if not snapshot["gpus"]:
                print("No GPU data parsed. " + "; ".join(snapshot["warnings"]), file=sys.stderr)
                return 1
            if args.watch and not args.json and sys.stdout.isatty():
                print("\033[2J\033[H", end="")
            print(json.dumps(snapshot, ensure_ascii=False) if args.json else draw(snapshot), flush=True)
            count += 1
            if args.watch == 0 or (args.count and count >= args.count):
                break
            if not args.json and not sys.stdout.isatty():
                print("\n" + "=" * 80, flush=True)
            time.sleep(args.watch)
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
