#!/usr/bin/env python3
"""
Summarize protein–ligand H-bond occupancy reports across systems.

Inputs (per system):
  mds/aggregate_outputs/systems/<system>/data/HBond_PL_occupancy_summary.csv

Output (CSV):
  mds/aggregate_outputs/hbond_pl_occupancy_top_pairs_summary.csv

This script is dependency-free (stdlib only), so it can run in minimal
environments where numpy/pandas are unavailable.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional


@dataclass(frozen=True)
class Row:
    direction: str
    donor: str
    acceptor: str
    occupancy_pct: float
    avg_distance_a: Optional[float]
    avg_angle_deg: Optional[float]
    pair: str


def to_float(x: object) -> Optional[float]:
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    try:
        return float(s)
    except Exception:
        return None


def read_rows(path: Path) -> List[Row]:
    rows: List[Row] = []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            occ = to_float(r.get("Occupancy (%)"))
            if occ is None:
                continue
            rows.append(
                Row(
                    direction=str(r.get("Direction", "")).strip(),
                    donor=str(r.get("Donor", "")).strip(),
                    acceptor=str(r.get("Acceptor", "")).strip(),
                    occupancy_pct=float(occ),
                    avg_distance_a=to_float(r.get("Avg Distance (A)")),
                    avg_angle_deg=to_float(r.get("Avg Angle (deg)")),
                    pair=str(r.get("Pair", "")).strip(),
                )
            )
    return rows


def iter_system_files(root: Path) -> Iterable[tuple[str, Path]]:
    systems_dir = root / "systems"
    if not systems_dir.exists():
        return
    for sys_dir in sorted(systems_dir.iterdir()):
        if not sys_dir.is_dir():
            continue
        system = sys_dir.name
        path = sys_dir / "data" / "HBond_PL_occupancy_summary.csv"
        if path.exists():
            yield system, path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    agg_root = repo_root / "mds" / "aggregate_outputs"
    out_csv = agg_root / "hbond_pl_occupancy_top_pairs_summary.csv"

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    out_rows: List[dict] = []
    for system, path in iter_system_files(agg_root):
        rows = read_rows(path)
        rows.sort(key=lambda r: r.occupancy_pct, reverse=True)
        top = rows[:5]
        for rank, r in enumerate(top, start=1):
            out_rows.append(
                {
                    "system": system,
                    "rank": rank,
                    "direction": r.direction,
                    "donor": r.donor,
                    "acceptor": r.acceptor,
                    "occupancy_pct": r.occupancy_pct,
                    "avg_distance_a": r.avg_distance_a,
                    "avg_angle_deg": r.avg_angle_deg,
                    "pair": r.pair,
                    "source_csv": str(path.relative_to(repo_root)),
                }
            )

    fields = [
        "system",
        "rank",
        "direction",
        "donor",
        "acceptor",
        "occupancy_pct",
        "avg_distance_a",
        "avg_angle_deg",
        "pair",
        "source_csv",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)

    print(f"[OK] Wrote: {out_csv}")
    print(f"[OK] Rows: {len(out_rows)} (5 per system when available)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

