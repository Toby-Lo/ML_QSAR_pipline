#!/usr/bin/env python3
'''
python find_metal_neighbors.py \
    ../docking/Final_candidates/mol2_for_mds_TOP100/9CVD_before_removing_zn/9CVD.pdb --metal ZN --cutoff 2.5

Metal:   ZN  ZN A1301  serial=3418  (  14.099, -10.454, -10.053)  elem='ZN'
   1.97 Å  ATOM     SG CYS A1198  serial=3347
   1.98 Å  ATOM     SG CYS A1144  serial=2559
   1.98 Å  ATOM     SG CYS A1193  serial=3295
   1.99 Å  ATOM     SG CYS A1191  serial=3261

Metal:   ZN  ZN A1302  serial=3419  ( -21.613,   1.658,  -4.327)  elem='ZN'
   2.15 Å  ATOM     SG CYS A1032  serial=762
   2.27 Å  ATOM     SG CYS A1018  serial=572
   2.46 Å  ATOM     SG CYS A1016  serial=548

Metal:   ZN  ZN A1303  serial=3420  ( -20.452,   2.960,  -0.896)  elem='ZN'
   2.19 Å  ATOM     SG CYS A1026  serial=687
   2.26 Å  ATOM     SG CYS A1041  serial=917
   2.33 Å  ATOM     SG CYS A1046  serial=991
   2.39 Å  ATOM     SG CYS A1052  serial=1067
'''
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Atom:
    record: str  # ATOM/HETATM
    serial: int
    name: str
    resname: str
    chain: str
    resseq: str  # keep as text (supports insertion codes)
    x: float
    y: float
    z: float
    element: str
    line: str


def parse_pdb_atoms(pdb_path: Path) -> list[Atom]:
    atoms: list[Atom] = []
    for raw in pdb_path.read_text(errors="ignore").splitlines():
        if not (raw.startswith("ATOM") or raw.startswith("HETATM")):
            continue
        # PDB fixed columns (1-based):
        #  1-6 record, 7-11 serial, 13-16 name, 18-20 resname, 22 chain,
        # 23-26 resseq, 31-38 x, 39-46 y, 47-54 z, 77-78 element
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
                line=raw,
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
    # fall back to residue name for poorly formatted PDBs
    if atom.resname.upper().startswith(m):
        return True
    if atom.name.upper().startswith(m):
        return True
    return False


def is_donor(atom: Atom) -> bool:
    # Common Zn donors in proteins: Cys SG, His ND1/NE2, Asp/Glu OD*/OE*, plus sometimes backbone O.
    # We keep it simple: consider N/O/S heavy atoms, exclude waters unless user wants them.
    e = (atom.element or "").upper()
    if e in {"N", "O", "S"}:
        return True
    # If element missing, infer from atom name first letter.
    if not e and atom.name:
        return atom.name[0].upper() in {"N", "O", "S"}
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description="List nearby donor atoms around metal ions in a PDB.")
    ap.add_argument("pdb", type=Path, help="Input PDB (e.g., protein_clean.pdb)")
    ap.add_argument("--metal", default="ZN", help="Metal element symbol (default: ZN)")
    ap.add_argument("--cutoff", type=float, default=3.0, help="Distance cutoff in Å (default: 3.0)")
    ap.add_argument("--include-water", action="store_true", help="Include HOH/WAT residues")
    ap.add_argument("--all-atoms", action="store_true", help="List all nearby atoms (not only N/O/S donors)")
    ap.add_argument("--max", type=int, default=24, help="Max neighbors to print per metal (default: 24)")
    args = ap.parse_args()

    atoms = parse_pdb_atoms(args.pdb)
    metals = [a for a in atoms if is_metal(a, args.metal)]
    if not metals:
        print(f"No metal '{args.metal}' found in: {args.pdb}")
        return 2

    for m in metals:
        print(
            f"\nMetal: {m.name:>4s} {m.resname:>3s} {m.chain}{m.resseq}  "
            f"serial={m.serial}  ({m.x:8.3f},{m.y:8.3f},{m.z:8.3f})  elem='{m.element}'"
        )
        neigh: list[tuple[float, Atom]] = []
        for a in atoms:
            if a is m:
                continue
            if not args.all_atoms and not is_donor(a):
                continue
            if not args.include_water and a.resname.upper() in {"HOH", "WAT"}:
                continue
            d = dist(m, a)
            if d <= args.cutoff:
                neigh.append((d, a))
        neigh.sort(key=lambda t: t[0])
        if not neigh:
            print(f"  No donor atoms within {args.cutoff:.2f} Å.")
            continue
        for d, a in neigh[: args.max]:
            elem = (a.element or "").strip()
            elem_disp = f" elem={elem!r}" if elem else ""
            print(
                f"  {d:5.2f} Å  {a.record:6s} {a.name:>4s} {a.resname:>3s} {a.chain}{a.resseq}  serial={a.serial}{elem_disp}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
