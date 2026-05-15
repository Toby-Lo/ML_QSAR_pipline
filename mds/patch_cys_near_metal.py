#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Atom:
    record: str
    serial: int
    name: str
    resname: str
    chain: str
    resseq: str  # includes optional insertion code
    x: float
    y: float
    z: float
    element: str


def parse_pdb_atoms(pdb_path: Path) -> list[Atom]:
    atoms: list[Atom] = []
    for raw in pdb_path.read_text(errors="ignore").splitlines():
        if not (raw.startswith("ATOM") or raw.startswith("HETATM")):
            continue
        record = raw[0:6].strip()
        try:
            serial = int(raw[6:11])
        except Exception:
            serial = -1
        name = raw[12:16].strip()
        resname = raw[17:20].strip()
        chain = raw[21:22].strip() or "?"
        resseq = raw[22:26].strip() + (raw[26:27].strip() or "")
        try:
            x = float(raw[30:38])
            y = float(raw[38:46])
            z = float(raw[46:54])
        except Exception:
            continue
        element = (raw[76:78].strip() if len(raw) >= 78 else "").strip()
        atoms.append(
            Atom(
                record=record,
                serial=serial,
                name=name,
                resname=resname,
                chain=chain,
                resseq=resseq,
                x=x,
                y=y,
                z=z,
                element=element,
            )
        )
    return atoms


def dist(a: Atom, b: Atom) -> float:
    dx = a.x - b.x
    dy = a.y - b.y
    dz = a.z - b.z
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def is_metal(atom: Atom, metal: str) -> bool:
    m = metal.upper()
    elem = (atom.element or "").upper()
    if elem.startswith(m):
        return True
    if atom.resname.upper().startswith(m):
        return True
    if atom.name.upper().startswith(m):
        return True
    return False


def patch_resname_in_pdb(
    pdb_in: Path,
    pdb_out: Path,
    targets: set[tuple[str, str]],
    from_resname: str,
    to_resname: str,
) -> int:
    from_name = from_resname.upper()
    to_name = to_resname.upper()
    changed = 0
    out_lines: list[str] = []
    for line in pdb_in.read_text(errors="ignore").splitlines(True):
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            out_lines.append(line)
            continue
        if len(line) < 27:
            out_lines.append(line)
            continue
        chain = line[21:22].strip() or "?"
        resseq = line[22:26].strip() + (line[26:27].strip() or "")
        if (chain, resseq) not in targets:
            out_lines.append(line)
            continue
        resname = line[17:20].strip().upper()
        if resname != from_name:
            out_lines.append(line)
            continue
        out_lines.append(f"{line[:17]}{to_name:>3s}{line[20:]}")
        changed += 1
    pdb_out.parent.mkdir(parents=True, exist_ok=True)
    pdb_out.write_text("".join(out_lines))
    return changed


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Patch CYS->CYM for cysteines whose SG is within cutoff of a metal (e.g., Zn) in a PDB."
    )
    ap.add_argument("inp", type=Path, help="Input PDB (e.g., protein_clean.pdb)")
    ap.add_argument("out", type=Path, help="Output PDB (e.g., protein_clean_ZnCYM.pdb)")
    ap.add_argument("--metal", default="ZN", help="Metal element symbol (default: ZN)")
    ap.add_argument("--cutoff", type=float, default=3.0, help="Distance cutoff in Å (default: 3.0)")
    ap.add_argument("--from", dest="from_resname", default="CYS", help="Residue name to patch (default: CYS)")
    ap.add_argument("--to", dest="to_resname", default="CYM", help="Residue name to write (default: CYM)")
    ap.add_argument("--dry-run", action="store_true", help="Only print detected targets; do not write output")
    args = ap.parse_args()

    atoms = parse_pdb_atoms(args.inp)
    metals = [a for a in atoms if is_metal(a, args.metal)]
    if not metals:
        print(f"No metal '{args.metal}' found in: {args.inp}")
        return 2

    targets: dict[tuple[str, str], float] = {}
    for m in metals:
        for a in atoms:
            if a.resname.upper() != args.from_resname.upper():
                continue
            if a.name.upper() != "SG":
                continue
            d = dist(m, a)
            if d <= args.cutoff:
                key = (a.chain, a.resseq)
                targets[key] = min(d, targets.get(key, 1e9))

    if not targets:
        print(f"No {args.from_resname} SG within {args.cutoff:.2f} Å of metal {args.metal}.")
        return 3

    print("Targets to patch (chain+resseq -> min_dist Å):")
    for (chain, resseq), d in sorted(targets.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        print(f"  {chain}{resseq}: {d:5.2f}")

    if args.dry_run:
        return 0

    changed = patch_resname_in_pdb(
        pdb_in=args.inp,
        pdb_out=args.out,
        targets=set(targets.keys()),
        from_resname=args.from_resname,
        to_resname=args.to_resname,
    )
    print(f"Patched lines: {changed}")
    print(f"Wrote: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

