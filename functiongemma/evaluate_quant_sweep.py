#!/usr/bin/env python3
"""Evaluate several router GGUFs sequentially on one machine.

The same-runner constraint is intentional: route-eval decisions can drift by a
few cases across CPU kernels, which is large enough to obscure the effect of a
nearby quantisation level. Each model is measured by route-eval with its raw
confidence dump, its floor is derived by derive_floor.py, and the generated-set
gate metrics are then computed from that dump using derive_floor.metrics (the
same definitions as route-eval).
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import derive_floor  # noqa: E402


FLOOR_RE = re.compile(
    r"floor: min_confidence=(?P<floor>-?[0-9.]+) source=(?P<source>\w+)"
)


def parse_model(value: str) -> tuple[str, Path]:
    label, separator, path = value.partition("=")
    if not separator or not label or not path:
        raise argparse.ArgumentTypeError("--model must be LABEL=PATH")
    if not re.fullmatch(r"[a-z0-9_]+", label):
        raise argparse.ArgumentTypeError(
            f"model label must contain only lowercase letters, digits and _: {label!r}"
        )
    return label, Path(path)


def measure(
    router: Path,
    model: Path,
    eval_path: Path,
    locale: str,
    skills_dir: Path,
    stdout_path: Path,
    stderr_path: Path,
) -> None:
    command = [
        str(router),
        "--locale",
        locale,
        "--skills-dir",
        str(skills_dir),
        str(model),
        str(eval_path),
    ]
    env = dict(os.environ, ROUTE_EVAL_VERBOSE="1")
    with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
        result = subprocess.run(command, stdout=stdout, stderr=stderr, env=env)
    if result.returncode >= 2:
        raise RuntimeError(
            f"route-eval failed structurally for {model.name} / {eval_path.name} "
            f"(exit {result.returncode}); see {stderr_path}"
        )


def derive(spine_log: Path, generated_log: Path, output_path: Path) -> tuple[float, str]:
    result = subprocess.run(
        [
            sys.executable,
            str(HERE / "derive_floor.py"),
            "--spine",
            str(spine_log),
            "--generated",
            str(generated_log),
        ],
        capture_output=True,
        text=True,
        env={k: v for k, v in os.environ.items() if k != "GITHUB_OUTPUT"},
    )
    output_path.write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError(
            f"derive_floor.py failed (exit {result.returncode}); see {output_path}"
        )
    match = FLOOR_RE.search(result.stdout)
    if not match:
        raise RuntimeError(f"derive_floor.py emitted no floor; see {output_path}")
    return float(match.group("floor")), match.group("source")


def render_summary(results: list[dict]) -> str:
    lines = [
        "# FunctionGemma quant sweep",
        "",
        "All variants were evaluated sequentially on the same runner.",
        "",
        "| Variant | Size (MiB) | Floor | Source | Abstention | Precision | Recall | Gate |",
        "|---|---:|---:|---|---:|---:|---:|---|",
    ]
    for result in results:
        metrics = result["metrics"]
        lines.append(
            f"| {result['label']} | {result['bytes'] / 1024 / 1024:.1f} "
            f"| {result['floor']:.4f} | {result['floor_source']} "
            f"| {metrics['abst_pass']}/{metrics['abst_total']} "
            f"({metrics['abstention']:.0%}) "
            f"| {metrics['correct']}/{metrics['fired']} "
            f"({metrics['precision']:.0%}) "
            f"| {metrics['correct']}/{metrics['pos_total']} "
            f"({metrics['recall']:.0%}) "
            f"| {'PASS' if result['passed'] else 'FAIL'} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--router", required=True, type=Path)
    parser.add_argument("--model", required=True, action="append", type=parse_model)
    parser.add_argument("--locale", required=True)
    parser.add_argument("--skills-dir", required=True, type=Path)
    parser.add_argument("--spine", required=True, type=Path)
    parser.add_argument("--generated", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--precision-min", type=float, default=0.85)
    parser.add_argument("--abstain-min", type=float, default=0.90)
    args = parser.parse_args()

    for path in [args.router, args.skills_dir, args.spine, args.generated]:
        if not path.exists():
            parser.error(f"not found: {path}")
    for _label, model in args.model:
        if not model.is_file():
            parser.error(f"model not found: {model}")

    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    for label, model in args.model:
        print(f"Measuring {label}: {model}", flush=True)
        model_output = args.output / label
        model_output.mkdir(exist_ok=True)
        spine_log = model_output / "route-eval.spine.log"
        generated_log = model_output / "route-eval.generated.log"
        measure(
            args.router,
            model,
            args.spine,
            args.locale,
            args.skills_dir,
            spine_log,
            model_output / "route-eval.spine.stderr.log",
        )
        measure(
            args.router,
            model,
            args.generated,
            args.locale,
            args.skills_dir,
            generated_log,
            model_output / "route-eval.generated.stderr.log",
        )
        floor, floor_source = derive(
            spine_log, generated_log, model_output / "derive-floor.log"
        )
        generated_cases, _compiled = derive_floor.parse_dump(generated_log)
        metrics = derive_floor.metrics(generated_cases, floor)
        passed = (
            metrics["precision"] >= args.precision_min
            and metrics["abstention"] >= args.abstain_min
        )
        result = {
            "label": label,
            "model": str(model),
            "bytes": model.stat().st_size,
            "floor": floor,
            "floor_source": floor_source,
            "metrics": metrics,
            "passed": passed,
        }
        results.append(result)
        print(
            f"  {label}: precision={metrics['precision']:.1%} "
            f"abstention={metrics['abstention']:.1%} "
            f"recall={metrics['recall']:.1%} "
            f"{'PASS' if passed else 'FAIL'}",
            flush=True,
        )

    (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    summary = render_summary(results)
    (args.output / "summary.md").write_text(summary)
    print(summary)


if __name__ == "__main__":
    main()
