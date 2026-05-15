#!/usr/bin/env python3
'''
python3 ../../patch_pdb_resname.py protein_clean.pdb protein_clean_ZnCYM.pdb --chain A --res 215 161 210 208 49 35 33 43 58 63 69 --from CYS --to CYM
'''

from __future__ import annotations

import argparse
from pathlib import Path


def iter_pdb_lines(path: Path) -> list[str]:
    return path.read_text(errors="ignore").splitlines(True)


def parse_resseq_icode(line: str) -> tuple[str, str]:
    # PDB: resseq is columns 23-26 (22:26 in 0-based), insertion code is column 27 (26:27)
    resseq = line[22:26].strip()
    icode = line[26:27].strip()
    return resseq, icode


def main() -> int:
    ap = argparse.ArgumentParser(description="Patch residue names in a PDB by chain+resseq (+optional icode).")
    ap.add_argument("inp", type=Path, help="Input PDB")
    ap.add_argument("out", type=Path, help="Output PDB")
    ap.add_argument("--chain", required=True, help="Chain ID (e.g., A)")
    ap.add_argument(
        "--res",
        nargs="+",
        required=True,
        help="Residue numbers (e.g., 1016 1018 1026). Use format like 1026A for insertion code A.",
    )
    ap.add_argument("--from", dest="from_resname", default="CYS", help="Only patch residues with this name (default: CYS)")
    ap.add_argument("--to", dest="to_resname", default="CYM", help="New residue name (default: CYM)")
    args = ap.parse_args()

    chain = args.chain
    from_name = args.from_resname.upper()
    to_name = args.to_resname.upper()
    if len(to_name) != 3:
        raise SystemExit("--to must be a 3-letter residue name (e.g., CYM)")

    targets: set[tuple[str, str]] = set()
    for r in args.res:
        r = str(r).strip()
        if not r:
            continue
        # Accept "1026" or "1026A" (icode)
        if r[-1].isalpha():
            targets.add((r[:-1], r[-1].upper()))
        else:
            targets.add((r, ""))

    out_lines: list[str] = []
    changed = 0
    for line in iter_pdb_lines(args.inp):
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            out_lines.append(line)
            continue
        if len(line) < 27:
            out_lines.append(line)
            continue
        line_chain = line[21:22]
        if line_chain != chain:
            out_lines.append(line)
            continue
        resseq, icode = parse_resseq_icode(line)
        if (resseq, icode.upper()) not in targets:
            out_lines.append(line)
            continue
        resname = line[17:20].strip().upper()
        if resname != from_name:
            out_lines.append(line)
            continue
        # Patch residue name in fixed columns 18-20 (17:20 0-based)
        patched = f"{line[:17]}{to_name:>3s}{line[20:]}"
        out_lines.append(patched)
        changed += 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(out_lines))
    print(f"Patched lines: {changed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

