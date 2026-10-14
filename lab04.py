import csv
import statistics
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from lab04_starter import RecallNet, make_recall_batch, train_recall

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
SEEDS = [0, 1, 2]
LENGTHS = [10, 30, 60]
CONFIGS = [
    ("RNN", "rnn", None),
    ("LSTM-default", "lstm", None),
    ("GRU-default", "gru", None),
    ("LSTM-forget-bias-1", "lstm", 1.0),
    ("GRU-update-bias-1", "gru", 1.0),
]


def prepare_folders():
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    torch.set_num_threads(1)


def run_benchmark():
    prepare_folders()
    rows = []
    fields = ["config", "kind", "gate_bias", "seq_len", "seed", "accuracy", "above_chance"]
    path = RESULTS / "benchmark.csv"
    for length in LENGTHS:
        for label, kind, bias in CONFIGS:
            for seed in SEEDS:
                accuracy, _ = train_recall(kind, length, seed=seed, gate_bias=bias, steps=600, batch_size=64, lr=3e-3, clip=1.0)
                row = {"config": label, "kind": kind, "gate_bias": bias, "seq_len": length, "seed": seed, "accuracy": accuracy, "above_chance": accuracy - 0.25}
                rows.append(row)
                print(f"{label} length={length} seed={seed} accuracy={accuracy:.4f} above_chance={accuracy - 0.25:+.4f}", flush=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print("\nBenchmark means")
    for length in LENGTHS:
        for label, _, _ in CONFIGS:
            values = [row["accuracy"] for row in rows if row["seq_len"] == length and row["config"] == label]
            print(f"{label} length={length} seeds={values} mean={statistics.mean(values):.4f} above_chance={statistics.mean(values) - 0.25:+.4f}")
    print(f"Saved {path}")
    return rows


def run_gradients():
    prepare_folders()
    seeds = [0, 1, 2, 3, 4]
    seq_len = 60
    rows = []
    for seed in seeds:
        torch.manual_seed(seed)
        model = RecallNet("rnn")
        x, y = make_recall_batch(128, seq_len, torch.Generator().manual_seed(seed + 100))
        x.requires_grad_(True)
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        norms = x.grad.norm(dim=2).median(dim=0).values
        for timestep, value in enumerate(norms.tolist()):
            rows.append({"seed": seed, "input_timestep": timestep, "steps_back_from_final": seq_len - 1 - timestep, "gradient_norm_median_over_batch": value})
    path = RESULTS / "gradients.csv"
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    offsets = [0, 9, 29, 59]
    summary = []
    for back in offsets:
        values = [row["gradient_norm_median_over_batch"] for row in rows if row["steps_back_from_final"] == back]
        summary.append(statistics.median(values))
        per_seed = [next(row["gradient_norm_median_over_batch"] for row in rows if row["seed"] == seed and row["steps_back_from_final"] == back) for seed in seeds]
        print(f"steps_back={back} seed_values={per_seed} across_seed_median={statistics.median(values):.8e}")
    plt.figure(figsize=(7, 4.5))
    for seed in seeds:
        seed_rows = sorted([row for row in rows if row["seed"] == seed], key=lambda row: row["steps_back_from_final"])
        plt.plot([row["steps_back_from_final"] for row in seed_rows], [max(row["gradient_norm_median_over_batch"], 1e-30) for row in seed_rows], alpha=0.25)
    medians = []
    for back in range(seq_len):
        values = [row["gradient_norm_median_over_batch"] for row in rows if row["steps_back_from_final"] == back]
        medians.append(max(statistics.median(values), 1e-30))
    plt.plot(range(seq_len), medians, marker="", linewidth=2.5, label="Median across five seeds")
    plt.yscale("log")
    plt.xlabel("Steps back from the final input step")
    plt.ylabel("Median input-gradient norm (log scale)")
    plt.title("RNN gradient through time: last-step loss, sequence length 60")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    image_path = FIGURES / "gradient_through_time.png"
    plt.savefig(image_path, dpi=180)
    plt.close()
    print(f"Saved {path} and {image_path}")
    return rows


def verify_gate(kind, gate_bias, hidden=32):
    model = RecallNet(kind, hidden=hidden, gate_bias=gate_bias)
    if kind == "lstm":
        gate_name = "forget"
    elif kind == "gru":
        gate_name = "update"
    else:
        raise ValueError("Gate verification applies to LSTM or GRU")
    ih = model.rnn.bias_ih_l0[hidden:2 * hidden]
    hh = model.rnn.bias_hh_l0[hidden:2 * hidden]
    effective = ih + hh
    print(f"{kind.upper()} {gate_name} gate slice [H:2H] = [{hidden}:{2 * hidden}]")
    print(f"bias_ih first={ih[0].item():.4f}; bias_hh first={hh[0].item():.4f}; effective first={effective[0].item():.4f}")
    assert torch.allclose(effective, torch.full_like(effective, gate_bias), atol=1e-6)
    print("Gate-bias verification passed.")


def run_fixes():
    prepare_folders()
    verify_gate("lstm", 1.0)
    verify_gate("gru", 1.0)
    rows = []
    for bias in [0.0, 1.0, 2.0, 4.0]:
        for seed in SEEDS:
            accuracy, _ = train_recall("lstm", 60, seed=seed, gate_bias=bias, steps=600, batch_size=64, lr=3e-3, clip=1.0)
            rows.append({"experiment": "forget_bias_sweep", "bias": bias, "seed": seed, "seq_len": 60, "clip": 1.0, "accuracy": accuracy})
            print(f"forget_bias={bias:.1f} seed={seed} accuracy={accuracy:.4f}", flush=True)
    for clip in [1.0, None]:
        for seed in SEEDS:
            accuracy, _ = train_recall("rnn", 30, seed=seed, gate_bias=None, steps=600, batch_size=64, lr=3e-3, clip=clip)
            rows.append({"experiment": "clipping_comparison", "bias": "none", "seed": seed, "seq_len": 30, "clip": "off" if clip is None else clip, "accuracy": accuracy})
            print(f"RNN length=30 clip={'off' if clip is None else clip} seed={seed} accuracy={accuracy:.4f}", flush=True)
    path = RESULTS / "fixes.csv"
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    groups = {}
    for row in rows:
        key = (row["experiment"], str(row["bias"]), str(row["clip"]))
        groups.setdefault(key, []).append(row["accuracy"])
    print("\nFixes means")
    for key, values in groups.items():
        print(f"{key}: seeds={values} mean={statistics.mean(values):.4f}")
    print(f"Saved {path}")
    return rows


def measure_timing(seq_len, repeats=3):
    model = nn.LSTM(6, 32, batch_first=True)
    x = torch.randn(64, seq_len, 6, requires_grad=True)
    forward_times = []
    backward_times = []
    for _ in range(repeats):
        model.zero_grad(set_to_none=True)
        x.grad = None
        start = time.perf_counter()
        output, _ = model(x)
        loss = output[:, -1, :].square().mean()
        forward_times.append(time.perf_counter() - start)
        start = time.perf_counter()
        loss.backward()
        backward_times.append(time.perf_counter() - start)
    return min(forward_times), min(backward_times)


def run_timing():
    prepare_folders()
    rows = []
    for length in [50, 100, 200, 400]:
        forward, backward = measure_timing(length)
        row = {"seq_len": length, "batch_size": 64, "hidden_size": 32, "best_forward_seconds": forward, "best_backward_seconds": backward, "best_combined_seconds": forward + backward, "repeats": 3}
        rows.append(row)
        print(f"length={length} forward={forward:.6f}s backward={backward:.6f}s combined={forward + backward:.6f}s")
    path = RESULTS / "timing.csv"
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {path}")
    return rows


def run_all():
    run_benchmark()
    run_gradients()
    run_fixes()
    run_timing()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("part", choices=["all", "benchmark", "gradients", "fixes", "timing"], nargs="?", default="all")
    part = parser.parse_args().part
    {"all": run_all, "benchmark": run_benchmark, "gradients": run_gradients, "fixes": run_fixes, "timing": run_timing}[part]()
