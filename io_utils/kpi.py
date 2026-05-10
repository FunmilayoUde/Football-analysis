from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def write_kpi_summary(out_dir: str, metrics: dict[str, Any]) -> tuple[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    json_path = out / "kpi_summary.json"
    csv_path = out / "kpi_summary.csv"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        w.writeheader()
        w.writerow(metrics)

    return str(json_path), str(csv_path)
