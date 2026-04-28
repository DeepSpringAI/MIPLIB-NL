#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def format_files_block(files: Dict[str, Any]) -> str:
    if not files:
        return ""

    lines = ["", "Associated data files:"]
    for key in sorted(files.keys()):
        value = files.get(key, {})
        if isinstance(value, dict):
            p = value.get("path", "")
            d = value.get("description", "")
            if p:
                lines.append(f"- {key}.path: {p}")
            if d:
                lines.append(f"- {key}.description: {d}")
        else:
            lines.append(f"- {key}: {value}")
    return "\n".join(lines)


def build_description(instance: Dict[str, Any]) -> str:
    abstract_problem = instance.get("abstract_problem", "").strip()
    parameters = instance.get("parameters", {})
    parameters_text = json.dumps(parameters, ensure_ascii=False, sort_keys=True)
    files_text = format_files_block(instance.get("files"))

    description = (
        f"{abstract_problem}\n\n"
        f"Parameters (JSON): {parameters_text}"
    )
    if files_text:
        description += f"{files_text}"
    return description.strip()


def convert_dataset(input_root: Path) -> List[Dict[str, Any]]:
    dataset_dir = input_root / "dataset"
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    tasks: List[Dict[str, Any]] = []
    for instance_file in sorted(dataset_dir.glob("*/instance.json")):
        task_id = instance_file.parent.name
        with instance_file.open("r", encoding="utf-8") as f:
            instance = json.load(f)

        optimal_value = instance.get("optimal_value", None)
        if optimal_value is None:
            continue

        try:
            ground_truth = float(optimal_value)
        except (TypeError, ValueError):
            continue

        task = {
            "task_id": task_id,
            "description": build_description(instance),
            "ground_truth": ground_truth,
            "tag": "miplib-nl",
        }
        tasks.append(task)

    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(description="Flatten MIPLIB-NL into AlphaOPT task schema.")
    parser.add_argument("--input-root", default=".", help="Root of MIPLIB-NL repository")
    parser.add_argument(
        "--output",
        default="data/miplib_nl_alphaopt_flattened.json",
        help="Output JSON path",
    )
    args = parser.parse_args()

    input_root = Path(args.input_root).resolve()
    output_path = (input_root / args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tasks = convert_dataset(input_root)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(tasks, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"Wrote {len(tasks)} tasks to {output_path}")


if __name__ == "__main__":
    main()
