"""Command-line interface.

Subcommands:

- ``generate``  — build a synthetic dataset (PNG + ground-truth JSON).
- ``detect``    — run the vision pipeline on one image, print detections.
- ``classify``  — evaluate a whole dataset directory (confusion matrix,
  precision/recall/F1, detection metrics).
- ``sort``      — simulate the conveyor for N items, print lane stats.
- ``report``    — evaluate + simulate + write Markdown/JSON/HTML outputs.
- ``demo``      — one small end-to-end run with printed summary.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import config
from .conveyor import run_conveyor
from .dashboard import build_dashboard
from .generator import generate_dataset
from .pipeline import detect, evaluate_dataset, load_image
from .report import build_report


def _cmd_generate(args: argparse.Namespace) -> int:
    paths = generate_dataset(Path(args.output), n_frames=args.frames,
                             seed=args.seed, n_objects=args.objects,
                             defect_rate=args.defect_rate)
    print(f"generated {len(paths)} frames in {args.output}")
    return 0


def _cmd_detect(args: argparse.Namespace) -> int:
    image = load_image(Path(args.image))
    detections = detect(image)
    print(f"{args.image}: {len(detections)} object(s)")
    for det in detections:
        print(f"  #{det['label_id']:>2} {det['label']:<9} "
              f"centroid=({det['centroid'][0]:.0f},{det['centroid'][1]:.0f}) "
              f"area={det['area']:>5} circ={det['circularity']:.3f} "
              f"extent={det['extent']:.3f} aspect={det['aspect_ratio']:.2f} "
              f"solidity={det['solidity']:.3f} conf={det['confidence']:.2f}")
    if args.output:
        Path(args.output).write_text(json.dumps(detections, indent=2, default=str))
        print(f"detections written to {args.output}")
    return 0


def _print_evaluation(summary: dict) -> None:
    print(f"frames: {summary['frames']}  matched objects: {summary['total']}  "
          f"mean frame time: {summary['mean_frame_ms']} ms")
    print(f"accuracy: {summary['accuracy']:.4f} "
          f"({summary['correct']}/{summary['total']})  "
          f"macro P/R/F1: {summary['macro_precision']:.3f} / "
          f"{summary['macro_recall']:.3f} / {summary['macro_f1']:.3f}")
    det = summary["detection"]
    print(f"detection: precision {det['precision']:.3f}  recall {det['recall']:.3f}  "
          f"F1 {det['f1']:.3f}  ({det['matched']}/{det['truth']} matched)")
    print("confusion matrix (rows=true, cols=pred):")
    header = "true\\pred | " + " | ".join(config.CLASSES)
    print(header)
    for true_label in config.CLASSES:
        row = summary["confusion_matrix"][true_label]
        print(f"{true_label:>9} | " + " | ".join(f"{row[p]:>9}" for p in config.CLASSES))
    print(f"{'class':<10}{'precision':>10}{'recall':>8}{'F1':>8}{'support':>9}")
    for label in config.CLASSES:
        m = summary["per_class"][label]
        print(f"{label:<10}{m['precision']:>10.3f}{m['recall']:>8.3f}"
              f"{m['f1']:>8.3f}{m['support']:>9}")


def _cmd_classify(args: argparse.Namespace) -> int:
    summary = evaluate_dataset(Path(args.dataset))
    _print_evaluation(summary)
    if args.output:
        Path(args.output).write_text(json.dumps(summary, indent=2, default=str))
        print(f"evaluation written to {args.output}")
    return 0


def _print_conveyor(run) -> None:
    print(f"items: {run.total}  duration: {run.duration_s:.0f} s  "
          f"throughput: {run.items_per_min:.1f} items/min")
    print(f"lanes: {run.lane_counts}  "
          f"false accepts: {run.false_accepts}  "
          f"false rejects: {run.false_rejects}  unknown: {run.unknown}")
    print(f"sort accuracy: {run.accuracy:.4f}  error rate: {run.error_rate:.4f}  "
          f"mean inspect: {run.mean_inspect_ms:.1f} ms/item")


def _cmd_sort(args: argparse.Namespace) -> int:
    run = run_conveyor(n_items=args.items, seed=args.seed,
                       defect_rate=args.defect_rate,
                       items_per_min=args.rate)
    _print_conveyor(run)
    if args.output:
        Path(args.output).write_text(json.dumps(run.to_dict(), indent=2))
        print(f"conveyor run written to {args.output}")
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    dataset = Path(args.dataset)
    summary = evaluate_dataset(dataset)
    run = run_conveyor(n_items=args.items, seed=args.seed,
                       defect_rate=args.defect_rate, items_per_min=args.rate)
    out = Path(args.output)
    md_path, json_path = build_report(summary, run.to_dict(),
                                      dataset.name, out)
    html_path = build_dashboard(summary, run, dataset.name,
                                out / "dashboard" / "index.html")
    print(f"report: {md_path}")
    print(f"summary: {json_path}")
    print(f"dashboard: {html_path}")
    _print_evaluation(summary)
    _print_conveyor(run)
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    summary = evaluate_dataset(Path(args.dataset))
    run = run_conveyor(n_items=args.items, seed=args.seed,
                       defect_rate=args.defect_rate)
    _print_evaluation(summary)
    _print_conveyor(run)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vision-sorter-sim",
        description="End-to-end machine-vision conveyor sorting simulator.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("generate", help="generate a synthetic dataset")
    p.add_argument("--output", "-o", default="data/dataset")
    p.add_argument("--frames", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--objects", type=int, default=5)
    p.add_argument("--defect-rate", type=float, default=0.12)
    p.set_defaults(func=_cmd_generate)

    p = sub.add_parser("detect", help="detect + classify objects in one image")
    p.add_argument("image")
    p.add_argument("--output", "-o", default=None)
    p.set_defaults(func=_cmd_detect)

    p = sub.add_parser("classify", help="evaluate a dataset directory")
    p.add_argument("dataset")
    p.add_argument("--output", "-o", default=None)
    p.set_defaults(func=_cmd_classify)

    p = sub.add_parser("sort", help="simulate the conveyor sorter")
    p.add_argument("--items", type=int, default=120)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--defect-rate", type=float, default=config.DEFAULT_DEFECT_RATE)
    p.add_argument("--rate", type=float, default=config.DEFAULT_ITEMS_PER_MIN)
    p.add_argument("--output", "-o", default=None)
    p.set_defaults(func=_cmd_sort)

    p = sub.add_parser("report", help="evaluate + simulate + write outputs")
    p.add_argument("dataset")
    p.add_argument("--output", "-o", default="examples/output")
    p.add_argument("--items", type=int, default=120)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--defect-rate", type=float, default=config.DEFAULT_DEFECT_RATE)
    p.add_argument("--rate", type=float, default=config.DEFAULT_ITEMS_PER_MIN)
    p.set_defaults(func=_cmd_report)

    p = sub.add_parser("demo", help="quick end-to-end demo on a dataset")
    p.add_argument("dataset")
    p.add_argument("--items", type=int, default=60)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--defect-rate", type=float, default=config.DEFAULT_DEFECT_RATE)
    p.set_defaults(func=_cmd_demo)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
