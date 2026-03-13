#!/usr/bin/env python3
"""
Auto-generate SESSION_SUMMARY.md from saved experiment artifacts.

This script is intentionally conservative:
- It reads metrics JSON files if they exist.
- It includes plot links only when the files exist.
- It reports trainable parameter counts from metrics when available.
- It falls back to checkpoint-derived counts for older notebooks.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parent
OUT_PATH = ROOT / "SESSION_SUMMARY.md"


# Non-trainable tensors commonly exported by snntorch modules into state_dict.
# We exclude these when inferring trainable parameter counts from checkpoints.
NON_TRAINABLE_SUFFIXES = (
    ".threshold",
    ".graded_spikes_factor",
    ".reset_mechanism_val",
    ".beta",
)


@dataclass
class NotebookSpec:
    notebook: str
    title: str
    change: str
    metrics_path: Path | None = None
    checkpoint_path: Path | None = None
    fallback_plots: dict[str, str] = field(default_factory=dict)
    parameter_note: str | None = None
    # Optional static Mermaid diagram block.
    mermaid: str | None = None


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def discover_state_dict(ckpt_obj: Any) -> dict[str, torch.Tensor] | None:
    if isinstance(ckpt_obj, dict):
        if "model_state_dict" in ckpt_obj and isinstance(ckpt_obj["model_state_dict"], dict):
            return ckpt_obj["model_state_dict"]
        if "state_dict" in ckpt_obj and isinstance(ckpt_obj["state_dict"], dict):
            return ckpt_obj["state_dict"]
        if all(hasattr(v, "numel") for v in ckpt_obj.values()):
            return ckpt_obj  # plain state_dict saved directly
    return None


def is_trainable_key(key: str) -> bool:
    # Fixed delay buffers are named "*.delays" in the older notebooks.
    if key.endswith(".delays"):
        return False
    if any(key.endswith(sfx) for sfx in NON_TRAINABLE_SUFFIXES):
        return False
    return True


def trainable_from_checkpoint(path: Path) -> int | None:
    if not path.exists():
        return None
    ckpt = torch.load(path, map_location="cpu")
    state = discover_state_dict(ckpt)
    if state is None:
        return None
    return int(
        sum(
            tensor.numel()
            for key, tensor in state.items()
            if hasattr(tensor, "numel") and is_trainable_key(key)
        )
    )


def fmt_pct(val: Any) -> str:
    if val is None:
        return "N/A"
    try:
        return f"{100.0 * float(val):.1f}%"
    except Exception:
        return "N/A"


def fmt_float(val: Any, nd: int = 4) -> str:
    if val is None:
        return "N/A"
    try:
        return f"{float(val):.{nd}f}"
    except Exception:
        return "N/A"


def existing_plot_lines(metrics: dict[str, Any], fallback: dict[str, str]) -> list[str]:
    lines: list[str] = []
    artifacts = metrics.get("artifacts", {}) if isinstance(metrics.get("artifacts"), dict) else {}

    # Prefer plot paths explicitly recorded in metrics.
    for key, p in artifacts.items():
        plot_path = ROOT / p
        if plot_path.exists():
            label = key.replace("_", " ").title()
            lines.append(f"- ![{label}]({p})")

    # Add fallback plots for older runs that did not save an "artifacts" dict.
    for key, p in fallback.items():
        if artifacts.get(key):
            continue
        plot_path = ROOT / p
        if plot_path.exists():
            label = key.replace("_", " ").title()
            lines.append(f"- ![{label}]({p})")

    return lines


def model_metric_lines(metrics: dict[str, Any]) -> list[str]:
    out: list[str] = []
    # Single-task classifiers
    if "final_train_acc" in metrics:
        out.append(f"- Final train accuracy: `{fmt_pct(metrics.get('final_train_acc'))}`")
    if "final_test_acc" in metrics:
        out.append(f"- Final test accuracy: `{fmt_pct(metrics.get('final_test_acc'))}`")

    # Multi-task range+azimuth variants
    if "final_range_train_acc" in metrics or "final_az_train_acc" in metrics:
        out.append(
            "- Final range train/test accuracy: "
            f"`{fmt_pct(metrics.get('final_range_train_acc'))}` / `{fmt_pct(metrics.get('final_range_test_acc'))}`"
        )
        out.append(
            "- Final azimuth train/test accuracy: "
            f"`{fmt_pct(metrics.get('final_az_train_acc'))}` / `{fmt_pct(metrics.get('final_az_test_acc'))}`"
        )

    if "best_test_loss" in metrics:
        out.append(f"- Best test loss: `{fmt_float(metrics.get('best_test_loss'))}`")
    if "ece" in metrics:
        out.append(f"- ECE: `{fmt_float(metrics.get('ece'))}`")
    if "ece_range" in metrics or "ece_azimuth" in metrics:
        out.append(
            f"- ECE (range / azimuth): `{fmt_float(metrics.get('ece_range'))}` / `{fmt_float(metrics.get('ece_azimuth'))}`"
        )
    if "learned_delay_min" in metrics or "learned_delay_max" in metrics:
        out.append(
            "- Learned delay range: "
            f"`{fmt_float(metrics.get('learned_delay_min'), 2)}` to `{fmt_float(metrics.get('learned_delay_max'), 2)}` steps"
        )
    if "learned_delay_range_min" in metrics or "learned_delay_range_max" in metrics:
        out.append(
            "- Learned range-delay span: "
            f"`{fmt_float(metrics.get('learned_delay_range_min'), 2)}` to "
            f"`{fmt_float(metrics.get('learned_delay_range_max'), 2)}` steps"
        )
        out.append(
            "- Learned azimuth-delay span: "
            f"`{fmt_float(metrics.get('learned_delay_az_min'), 2)}` to "
            f"`{fmt_float(metrics.get('learned_delay_az_max'), 2)}` steps"
        )
    return out


def main() -> None:
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    specs: list[NotebookSpec] = [
        NotebookSpec(
            notebook="network_v1.ipynb",
            title="Baseline Pulse+Echo Classifier",
            change="Initial baseline model with fixed delay bank and 1 hidden LIF layer.",
            # No direct metrics/checkpoint was saved for v1 in artifacts. We compute trainable params
            # from architecture constants (delay=59, hidden=96, classes=20): 59*96 + 96 + 96*20 + 20 = 7700.
            parameter_note="7700 (derived from architecture constants in notebook code)",
            mermaid="""```mermaid
flowchart LR
    A[Gaussian pulse + delayed echo] --> B[Fixed delay bank]
    B --> C[Linear + LIF hidden]
    C --> D[Linear + LIF output]
    D --> E[Spike-count readout]
    E --> F[Distance class]
```""",
        ),
        NotebookSpec(
            notebook="network_v2.ipynb / notebook_v2.ipynb",
            title="Baseline + Analysis/Tuning Infrastructure",
            change="Added Optuna, checkpoint/JSON persistence, and richer diagnostics.",
            checkpoint_path=ROOT / "artifacts_v2/checkpoints/network_v2_baseline_best.pt",
            fallback_plots={
                "training_curves": "artifacts_v2/plots/training_curves.png",
                "confusion_matrix": "artifacts_v2/plots/confusion_matrix.png",
                "weights_distribution_and_sorted": "artifacts_v2/plots/weights_distribution_and_sorted.png",
            },
            mermaid="""```mermaid
flowchart TD
    A[Baseline SNN train/eval] --> B[Checkpoint .pt save/load]
    A --> C[Best params JSON save/load]
    A --> D[Optuna SQLite study]
    A --> E[Diagnostics plots]
```""",
        ),
        NotebookSpec(
            notebook="network_step1_tx_delay_coincidence.ipynb",
            title="Step 1: TX-Only Delay Coincidence Bank",
            change="Delayed only TX stream before coincidence with RX stream.",
            metrics_path=ROOT / "artifacts_steps/step1_tx_delay_coincidence/metrics_step1.json",
            checkpoint_path=ROOT / "artifacts_steps/step1_tx_delay_coincidence/model_step1.pt",
            fallback_plots={
                "curves": "artifacts_steps/step1_tx_delay_coincidence/plots/curves_step1.png",
                "confusion": "artifacts_steps/step1_tx_delay_coincidence/plots/confusion_step1.png",
            },
        ),
        NotebookSpec(
            notebook="network_step1_retuned_optuna.ipynb",
            title="Step 1 (Retuned): Optuna + Calibration/Error Diagnostics",
            change="Retuned Step 1 with Optuna and added calibration/per-bin error analysis.",
            metrics_path=ROOT / "artifacts_steps/step1_retuned_optuna/metrics_step1_retuned.json",
            checkpoint_path=ROOT / "artifacts_steps/step1_retuned_optuna/checkpoints/model_step1_retuned.pt",
        ),
        NotebookSpec(
            notebook="network_step2_continuous_binned.ipynb",
            title="Step 2: Continuous Simulation, Binned Labels",
            change="Distance sampled continuously, then supervised by class bins.",
            metrics_path=ROOT / "artifacts_steps/step2_continuous_binned/metrics_step2.json",
            checkpoint_path=ROOT / "artifacts_steps/step2_continuous_binned/model_step2.pt",
            fallback_plots={
                "curves": "artifacts_steps/step2_continuous_binned/plots/curves_step2.png",
                "confusion": "artifacts_steps/step2_continuous_binned/plots/confusion_step2.png",
            },
        ),
        NotebookSpec(
            notebook="network_step3_trainable_delays.ipynb",
            title="Step 3: Trainable Delay Taps",
            change="Converted fixed delay taps to trainable soft/interpolated delays.",
            metrics_path=ROOT / "artifacts_steps/step3_trainable_delays/metrics_step3.json",
            checkpoint_path=ROOT / "artifacts_steps/step3_trainable_delays/model_step3.pt",
            fallback_plots={
                "curves": "artifacts_steps/step3_trainable_delays/plots/curves_step3.png",
                "learned_delays": "artifacts_steps/step3_trainable_delays/plots/learned_delays_step3.png",
            },
        ),
        NotebookSpec(
            notebook="network_step4_fm_filterbank.ipynb",
            title="Step 4: FM Sweep + Filterbank Front-End",
            change="Replaced simple pulse with chirp and added frequency filterbank channels.",
            metrics_path=ROOT / "artifacts_steps/step4_fm_filterbank/metrics_step4.json",
            checkpoint_path=ROOT / "artifacts_steps/step4_fm_filterbank/model_step4.pt",
            fallback_plots={
                "curves": "artifacts_steps/step4_fm_filterbank/plots/curves_step4.png",
                "confusion": "artifacts_steps/step4_fm_filterbank/plots/confusion_step4.png",
            },
        ),
        NotebookSpec(
            notebook="network_step5_parallel_range_azimuth.ipynb",
            title="Step 5: Parallel Range + Azimuth Branches",
            change="Introduced parallel coincidence branches for multitask range/azimuth prediction.",
            metrics_path=ROOT / "artifacts_steps/step5_parallel_range_azimuth/metrics_step5.json",
            checkpoint_path=ROOT / "artifacts_steps/step5_parallel_range_azimuth/model_step5.pt",
            fallback_plots={
                "curves": "artifacts_steps/step5_parallel_range_azimuth/plots/curves_step5.png",
                "range_confusion": "artifacts_steps/step5_parallel_range_azimuth/plots/confusion_range_step5.png",
                "azimuth_confusion": "artifacts_steps/step5_parallel_range_azimuth/plots/confusion_azimuth_step5.png",
            },
        ),
        NotebookSpec(
            notebook="network_step5_retuned_optuna_realistic.ipynb",
            title="Step 5 (Retuned): Realistic Azimuth + Loss Schedule",
            change=(
                "Added ITD + frequency-dependent ILD approximation + noise/reverb simulation, "
                "Optuna retuning, and per-task loss-weight scheduling."
            ),
            metrics_path=ROOT / "artifacts_steps/step5_retuned_optuna_realistic/metrics_step5_retuned.json",
            checkpoint_path=ROOT / "artifacts_steps/step5_retuned_optuna_realistic/checkpoints/model_step5_retuned.pt",
        ),
        NotebookSpec(
            notebook="network_step6_combined_all_complexities.ipynb",
            title="Step 6: Combined All Complexities",
            change=(
                "Integrated FM/filterbank front-end, continuous range+azimuth simulation, "
                "parallel branches, trainable delays, loss schedule, and full diagnostics."
            ),
            metrics_path=ROOT / "artifacts_steps/step6_combined_all/metrics_step6_combined.json",
            checkpoint_path=ROOT / "artifacts_steps/step6_combined_all/checkpoints/model_step6_combined.pt",
            # Derived from default architecture when metrics/checkpoint are not yet available:
            # conv(8x1x33)=264, delay params=24+24, fc_r1/fc_a1=(192->128), fc_r2=(128->20), fc_a2=(128->9)
            parameter_note="53,461 (derived from Step 6 default architecture)",
            fallback_plots={
                "signal_example": "artifacts_steps/step6_combined_all/plots/signal_example_step6_combined.png",
            },
        ),
    ]

    # Preload metrics and parameter counts so we can render both per-section and a single summary table.
    loaded_metrics: dict[str, dict[str, Any]] = {}
    trainable_map: dict[str, str] = {}

    for spec in specs:
        metrics = load_json(spec.metrics_path) if spec.metrics_path else {}
        loaded_metrics[spec.notebook] = metrics

        trainable: int | None = None
        if "trainable_params" in metrics:
            try:
                trainable = int(metrics["trainable_params"])
            except Exception:
                trainable = None

        if trainable is None and spec.checkpoint_path is not None:
            trainable = trainable_from_checkpoint(spec.checkpoint_path)

        if trainable is not None:
            trainable_map[spec.notebook] = f"{trainable:,}"
        elif spec.parameter_note:
            trainable_map[spec.notebook] = spec.parameter_note
        else:
            trainable_map[spec.notebook] = "N/A (artifact not available yet)"

    lines: list[str] = []
    lines.append("# Radar SNN Incremental Complexity Summary")
    lines.append("")
    lines.append(f"_Auto-generated by `update_session_summary.py` on {generated_at}_")
    lines.append("")
    lines.append("## Notebook Order (Increasing Complexity)")
    for idx, spec in enumerate(specs, start=1):
        lines.append(f"{idx}. `{spec.notebook}`")
    lines.append("")

    lines.append("## Trainable Parameters by Network")
    lines.append("")
    lines.append("| Network | Trainable Parameters | Source |")
    lines.append("|---|---:|---|")
    for spec in specs:
        metrics = loaded_metrics[spec.notebook]
        if "trainable_params" in metrics:
            source = "metrics JSON"
        elif spec.parameter_note:
            source = "derived from notebook architecture"
        elif spec.checkpoint_path and spec.checkpoint_path.exists():
            source = "checkpoint-derived"
        else:
            source = "not available yet"
        lines.append(f"| `{spec.notebook}` | {trainable_map[spec.notebook]} | {source} |")
    lines.append("")

    lines.append("## Per-Notebook Results")
    for idx, spec in enumerate(specs, start=1):
        metrics = loaded_metrics[spec.notebook]
        lines.append("")
        lines.append(f"### {idx}) `{spec.notebook}`")
        lines.append(f"**{spec.title}**")
        lines.append("")
        lines.append(f"- Change from previous: {spec.change}")
        lines.append(f"- Trainable parameters: `{trainable_map[spec.notebook]}`")

        metric_lines = model_metric_lines(metrics)
        if metric_lines:
            lines.append("- Metrics:")
            lines.extend(metric_lines)
        else:
            lines.append("- Metrics: `N/A` (run artifacts not found yet).")

        if spec.mermaid:
            lines.append("")
            lines.append(spec.mermaid)

        plot_lines = existing_plot_lines(metrics, spec.fallback_plots)
        if plot_lines:
            lines.append("")
            lines.append("- Key plots:")
            lines.extend(plot_lines)

        # Include tuned hyperparameters when present.
        params = metrics.get("params")
        if isinstance(params, dict) and params:
            lines.append("")
            lines.append("- Best/used parameters:")
            for k, v in params.items():
                lines.append(f"  - `{k}`: `{v}`")

    lines.append("")
    lines.append("## Update Route")
    lines.append("")
    lines.append(
        "Rebuild this report any time new notebook artifacts are generated:"
    )
    lines.append("")
    lines.append("```bash")
    lines.append(".venv/bin/python update_session_summary.py")
    lines.append("```")
    lines.append("")
    lines.append(
        "The generator reads `metrics_*.json` files and referenced plot paths under "
        "`artifacts_steps/` and `artifacts_v2/`."
    )

    OUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Updated summary: {OUT_PATH}")


if __name__ == "__main__":
    main()
