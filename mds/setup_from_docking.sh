#!/usr/bin/env bash
set -euo pipefail

VERSION="2026-04-25"

usage() {
  cat <<'EOF'
Usage:
  bash setup_from_docking.sh -p <receptor.pdb> -l <ligand.mol2> [-n <net_charge>] [-o <run_dir>] [--skip-tleap]
  bash setup_from_docking.sh -p <receptor.pdb> -l <ligand.mol2> [-n <net_charge>] [-o <run_dir>] --use-prebuilt-mcpb <mcpb_dir>

Example:
bash setup_from_docking.sh -p ~/TobyLo/Project/1.2QSAR-simple-ml/docking/Final_candidate/mol2_for_mds_TOP100/9CVD_before_removing_zn/9CVD.pdb -l ~/TobyLo/Project/1.2QSAR-simple-ml/docking/Final_candidate/mol2_for_mds_TOP100/a1a0m.mol2 --skip-tleap
bash setup_from_docking.sh -p ~/TobyLo/Project/1.2QSAR-simple-ml/docking/Final_candidate/mol2_for_mds_TOP100/9CVD_before_removing_zn/9CVD.pdb -l ~/TobyLo/Project/1.2QSAR-simple-ml/docking/Final_candidate/mol2_for_mds_TOP100/18198196.mol2 --use-prebuilt-mcpb ~/TobyLo/Project/1.2QSAR-simple-ml/mds/runs/a1a0m/mcpb_zn
bash setup_from_docking.sh -p ~/TobyLo/Project/1.2QSAR-simple-ml/docking/Final_candidate/mol2_for_mds_TOP100/9CVD_before_removing_zn/9CVD.pdb -l ~/TobyLo/Project/1.2QSAR-simple-ml/docking/Final_candidate/mol2_for_mds_TOP100/18198198.mol2 --use-prebuilt-mcpb  ~/TobyLo/Project/1.2QSAR-simple-ml/mds/runs/a1a0m/mcpb_zn
bash setup_from_docking.sh -p ~/TobyLo/Project/1.2QSAR-simple-ml/docking/Final_candidate/mol2_for_mds_TOP100/9CVD_before_removing_zn/9CVD.pdb -l ~/TobyLo/Project/1.2QSAR-simple-ml/docking/Final_candidate/mol2_for_mds_TOP100/18198200.mol2 --use-prebuilt-mcpb   ~/TobyLo/Project/1.2QSAR-simple-ml/mds/runs/a1a0m/mcpb_zn
bash setup_from_docking.sh -p ~/TobyLo/Project/1.2QSAR-simple-ml/docking/Final_candidate/mol2_for_mds_TOP100/9CVD_before_removing_zn/9CVD.pdb -l ~/TobyLo/Project/1.2QSAR-simple-ml/docking/Final_candidate/mol2_for_mds_TOP100/1857797734.mol2 --use-prebuilt-mcpb  ~/TobyLo/Project/1.2QSAR-simple-ml/mds/runs/a1a0m/mcpb_zn
bash setup_from_docking.sh -p ~/TobyLo/Project/1.2QSAR-simple-ml/docking/Final_candidate/mol2_for_mds_TOP100/9CVD_before_removing_zn/9CVD.pdb -l ~/TobyLo/Project/1.2QSAR-simple-ml/docking/Final_candidate/mol2_for_mds_TOP100/1857797811.mol2  --use-prebuilt-mcpb  ~/TobyLo/Project/1.2QSAR-simple-ml/mds/runs/a1a0m/mcpb_zn
bash setup_from_docking.sh -p ~/TobyLo/Project/1.2QSAR-simple-ml/docking/Final_candidate/mol2_for_mds_TOP100/9CVD_before_removing_zn/9CVD.pdb -l ~/TobyLo/Project/1.2QSAR-simple-ml/docking/Final_candidate/mol2_for_mds_TOP100/219210673.mol2 --use-prebuilt-mcpb  ~/TobyLo/Project/1.2QSAR-simple-ml/mds/runs/a1a0m/mcpb_zn
bash setup_from_docking.sh -p ~/TobyLo/Project/1.2QSAR-simple-ml/docking/Final_candidate/mol2_for_mds_TOP100/9CVD_before_removing_zn/9CVD.pdb -l ~/TobyLo/Project/1.2QSAR-simple-ml/docking/Final_candidate/mol2_for_mds_TOP100/36354596.mol2 --use-prebuilt-mcpb  ~/TobyLo/Project/1.2QSAR-simple-ml/mds/runs/a1a0m/mcpb_zn
bash setup_from_docking.sh -p ~/TobyLo/Project/1.2QSAR-simple-ml/docking/Final_candidate/mol2_for_mds_TOP100/9CVD_before_removing_zn/9CVD.pdb -l ~/TobyLo/Project/1.2QSAR-simple-ml/docking/Final_candidate/mol2_for_mds_TOP100/58175203.mol2 --use-prebuilt-mcpb  ~/TobyLo/Project/1.2QSAR-simple-ml/mds/runs/a1a0m/mcpb_zn

Monitor:
tail -n 20 ./runs/18198196/cMD-Prod.mdinfo
ls -lh ./runs/18198196/cMD-Prod.nc

What it does (AMBER workflow):
  1) pdb4amber: receptor.pdb -> protein_clean.pdb
  2) antechamber: ligand.mol2 -> ligand_clean.mol2 (GAFF2 + AM1-BCC; uses -nc net_charge)
  3) parmchk2: ligand_clean.mol2 -> ligand.frcmod
  4) Copies MD input templates into run_dir
  5) tleap -f leap.in: generates complex.parm7 + complex.rst7

Then you can run:
  cd <run_dir>
  bash MD_run.in

Notes:
  - You must have AMBER commands in PATH: pdb4amber, antechamber, parmchk2, tleap, pmemd.cuda, cpptraj
  - If your ligand is not neutral, pass the correct integer net charge via -n (e.g. -1, +1).
  - If your receptor contains structural metals (e.g., Zn), consider `--skip-tleap` so you can run MCPB.py first,
    then load the generated frcmod/lib in leap and run tleap manually.
  - If you already generated MCPB.py outputs once for the same protein (same cleaned residue numbering),
    reuse them with `--use-prebuilt-mcpb`. Provide either:
      * the directory `mcpb_zn/` from a previous run, OR
      * a parent directory that contains `mcpb_zn/`.
    The script will copy it into the new run directory and patch `leap.in` to load all `*_mcpbpy.frcmod`.
EOF
}

need_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing command: $1 (is AMBER sourced/loaded?)" >&2
    exit 127
  fi
}

guess_charge_from_mol2() {
  local mol2="$1"
  python3 - "$mol2" <<'PY'
import sys, math
path = sys.argv[1]
in_atoms = False
total = 0.0
with open(path, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        line = line.strip()
        if line.startswith("@<TRIPOS>ATOM"):
            in_atoms = True
            continue
        if line.startswith("@<TRIPOS>") and in_atoms:
            break
        if in_atoms and line:
            parts = line.split()
            # Tripos mol2 atom line: ... <subst_id> <subst_name> <charge>
            try:
                total += float(parts[-1])
            except Exception:
                pass
charge = int(round(total))
print(charge)
PY
}

receptor_pdb=""
ligand_mol2=""
net_charge=""
run_dir=""
skip_tleap="0"
prebuilt_mcpb=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -p|--protein|--receptor) receptor_pdb="${2:-}"; shift 2 ;;
    -l|--ligand) ligand_mol2="${2:-}"; shift 2 ;;
    -n|--net-charge) net_charge="${2:-}"; shift 2 ;;
    -o|--out|--run-dir) run_dir="${2:-}"; shift 2 ;;
    --skip-tleap) skip_tleap="1"; shift 1 ;;
    --use-prebuilt-mcpb) prebuilt_mcpb="${2:-}"; shift 2 ;;
    --version) echo "$VERSION"; exit 0 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "$receptor_pdb" || -z "$ligand_mol2" ]]; then
  usage
  exit 2
fi

if [[ ! -f "$receptor_pdb" ]]; then
  echo "Receptor PDB not found: $receptor_pdb" >&2
  exit 2
fi
if [[ ! -f "$ligand_mol2" ]]; then
  echo "Ligand mol2 not found: $ligand_mol2" >&2
  exit 2
fi

need_cmd pdb4amber
need_cmd antechamber
need_cmd parmchk2
need_cmd python3
if [[ "$skip_tleap" != "1" ]]; then
  need_cmd tleap
fi

if [[ -n "$prebuilt_mcpb" && ! -d "$prebuilt_mcpb" ]]; then
  echo "Prebuilt MCPB dir not found: $prebuilt_mcpb" >&2
  exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
lig_base="$(basename "$ligand_mol2")"
lig_name="${lig_base%.*}"

if [[ -z "$run_dir" ]]; then
  run_dir="${script_dir}/runs/${lig_name}"
fi

mkdir -p "$run_dir"

echo "[setup] run_dir: $run_dir"
echo "[setup] receptor: $receptor_pdb"
echo "[setup] ligand  : $ligand_mol2"

cp -f "$receptor_pdb" "${run_dir}/protein.pdb"
cp -f "$ligand_mol2" "${run_dir}/ligand.mol2"

if [[ -z "$net_charge" ]]; then
  net_charge="$(guess_charge_from_mol2 "${run_dir}/ligand.mol2")"
  echo "[setup] net_charge not provided; guessed: $net_charge"
else
  echo "[setup] net_charge provided: $net_charge"
fi

echo "[1/4] pdb4amber receptor cleanup"
pdb4amber -i "${run_dir}/protein.pdb" -o "${run_dir}/protein_clean.pdb"

has_zn="$(
  python3 - <<'PY' "${run_dir}/protein_clean.pdb"
import sys
pdb = sys.argv[1]
has = False
with open(pdb, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
        # PDB element is typically columns 77-78; fall back to token-based detection.
        elem = line[76:78].strip().upper() if len(line) >= 78 else ""
        if elem.startswith("ZN"):
            has = True
            break
        toks = line.split()
        if toks and toks[-1].upper().startswith("ZN"):
            has = True
            break
print("1" if has else "0")
PY
)"
if [[ "$has_zn" == "1" ]]; then
  cat <<'EOF' >&2
[warn] Detected Zn in protein_clean.pdb.
       A plain nonbonded Zn2+ ion model often loses the correct coordination geometry.
       Recommended (simplest + reasonable in AMBER): parameterize the Zn site with MCPB.py (bonded model),
       then load the generated frcmod/lib in tleap.
       If you used a *_remove_zn.pdb before, this is why.
EOF

  echo "[setup] generating protein_clean_ZnCYM.pdb (CYS near Zn -> CYM)"
  (cd "$run_dir" && python3 - <<'PY'
from __future__ import annotations

import math
from pathlib import Path

inp = Path("protein_clean.pdb")
out = Path("protein_clean_ZnCYM.pdb")

def parse_atoms(lines: list[str]):
    atoms = []
    for raw in lines:
        if not (raw.startswith("ATOM") or raw.startswith("HETATM")):
            continue
        if len(raw) < 54:
            continue
        record = raw[0:6].strip()
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
        element = (raw[76:78].strip() if len(raw) >= 78 else "").strip().upper()
        atoms.append((record, name, resname, chain, resseq, x, y, z, element))
    return atoms

def is_zn(atom) -> bool:
    _, name, resname, _, _, _, _, _, element = atom
    if element.startswith("ZN"):
        return True
    if resname.upper().startswith("ZN"):
        return True
    if name.upper().startswith("ZN"):
        return True
    return False

def dist(a, b) -> float:
    return math.sqrt((a[5]-b[5])**2 + (a[6]-b[6])**2 + (a[7]-b[7])**2)

lines = inp.read_text(errors="ignore").splitlines(True)
atoms = parse_atoms([l.rstrip("\n") for l in lines])
zn_atoms = [a for a in atoms if is_zn(a)]
if not zn_atoms:
    out.write_text("".join(lines))
    print("[setup] no Zn found; wrote protein_clean_ZnCYM.pdb unchanged")
    raise SystemExit(0)

cutoff = 3.0
targets = set()
for zn in zn_atoms:
    for a in atoms:
        record, name, resname, chain, resseq, *_ = a
        if resname.upper() != "CYS":
            continue
        if name.upper() != "SG":
            continue
        if dist(zn, a) <= cutoff:
            targets.add((chain, resseq))

changed = 0
out_lines = []
for line in lines:
    if not (line.startswith("ATOM") or line.startswith("HETATM")):
        out_lines.append(line)
        continue
    if len(line) < 27:
        out_lines.append(line)
        continue
    chain = line[21:22].strip() or "?"
    resseq = line[22:26].strip() + (line[26:27].strip() or "")
    resname = line[17:20].strip().upper()
    if (chain, resseq) in targets and resname == "CYS":
        out_lines.append(f"{line[:17]}{'CYM':>3s}{line[20:]}")
        changed += 1
    else:
        out_lines.append(line)

out.write_text("".join(out_lines))
print(f"[setup] CYM patch targets: {len(targets)} residues; patched lines: {changed}")
PY
  )
fi

protein_res_count="$(
  cd "$run_dir" && python3 - <<'PY'
from pathlib import Path

p = Path("protein_clean.pdb")
res = []
seen = set()
for line in p.read_text(errors="ignore").splitlines():
    if not line.startswith("ATOM"):
        continue
    chain = line[21:22]
    resnum = line[22:26]
    icode = line[26:27]
    key = (chain, resnum, icode)
    if key not in seen:
        seen.add(key)
        res.append(key)
print(len(res))
PY
)"
echo "[setup] detected protein residues (from protein_clean.pdb): $protein_res_count"

echo "[2/4] antechamber GAFF2 + AM1-BCC (this may take a while)"
antechamber -i "${run_dir}/ligand.mol2" -fi mol2 \
  -o "${run_dir}/ligand_clean.mol2" -fo mol2 \
  -at gaff2 -c bcc -s 2 -nc "$net_charge"

echo "[3/4] parmchk2 frcmod generation"
parmchk2 -i "${run_dir}/ligand_clean.mol2" -f mol2 -o "${run_dir}/ligand.frcmod"

echo "[4/4] copy templates + tleap"
cp -f "${script_dir}/leap.in" "${run_dir}/leap.in"
cp -f "${script_dir}/min1.in" "${run_dir}/min1.in"
cp -f "${script_dir}/min2.in" "${run_dir}/min2.in"
cp -f "${script_dir}/heat.in" "${run_dir}/heat.in"
cp -f "${script_dir}/density.in" "${run_dir}/density.in"
cp -f "${script_dir}/equil.in" "${run_dir}/equil.in"
cp -f "${script_dir}/cMD-prod.in" "${run_dir}/cMD-prod.in"
cp -f "${script_dir}/Auto-image.in" "${run_dir}/Auto-image.in"
cp -f "${script_dir}/MD_run.in" "${run_dir}/MD_run.in"

(cd "$run_dir" && python3 - <<'PY'
from __future__ import annotations

from pathlib import Path
import os

leap = Path("leap.in")
if not leap.exists():
    raise SystemExit(0)

use_zncym = Path("protein_clean_ZnCYM.pdb").exists()
txt = leap.read_text(errors="ignore").splitlines(True)
out = []
for line in txt:
    if "receptor=loadPDB" in line:
        if use_zncym:
            out.append("receptor=loadPDB protein_clean_ZnCYM.pdb\n")
        else:
            out.append(line)
    else:
        out.append(line)
leap.write_text("".join(out))
PY
)

(cd "$run_dir" && PROTEIN_RES_COUNT="$protein_res_count" python3 - <<'PY'
from pathlib import Path
import os

protein_res_count = int(os.environ["PROTEIN_RES_COUNT"])
mask = f":1-{protein_res_count} & !@H="

def patch_restraintmask(path: str) -> None:
    p = Path(path)
    if not p.exists():
        return
    txt = p.read_text(errors="ignore").splitlines(True)
    out = []
    changed = False
    for line in txt:
        if "restraintmask=" in line:
            prefix = line.split("restraintmask=", 1)[0]
            out.append(f"{prefix}restraintmask='{mask}',\n")
            changed = True
        else:
            out.append(line)
    if changed:
        p.write_text("".join(out))

for fname in ["min1.in", "heat.in", "density.in"]:
    patch_restraintmask(fname)
print(f"[setup] restraintmask patched to: {mask} (in min1.in/heat.in/density.in)")
PY
)

if [[ -n "$prebuilt_mcpb" ]]; then
  (cd "$run_dir" && PREBUILT_MCPB="$prebuilt_mcpb" python3 - <<'PY'
from __future__ import annotations

from pathlib import Path
import os
import subprocess

leap = Path("leap.in")
if not leap.exists():
    raise SystemExit(0)

prebuilt = os.environ.get("PREBUILT_MCPB", "").strip()
if not prebuilt:
    raise SystemExit(0)

src = Path(prebuilt)
if (src / "mcpb_zn").is_dir():
    src = src / "mcpb_zn"

if not src.is_dir():
    raise SystemExit(f"Invalid prebuilt MCPB directory: {prebuilt}")

dst = Path("mcpb_zn")
if not dst.exists():
    subprocess.check_call(["cp", "-R", str(src), str(dst)])

frcmods = sorted([p.as_posix() for p in dst.rglob("*_mcpbpy.frcmod")])
if not frcmods:
    print("[setup] warning: no *_mcpbpy.frcmod found under mcpb_zn/")
    raise SystemExit(0)

lines = leap.read_text(errors="ignore").splitlines(True)

# Remove old mcpb_zn load lines to avoid duplicates.
filtered = [ln for ln in lines if not ln.strip().startswith("loadAmberParams mcpb_zn/")]

insert_at = None
for i, ln in enumerate(filtered):
    if "LIG=loadmol2" in ln or "=loadmol2" in ln:
        insert_at = i + 1
        break
if insert_at is None:
    for i, ln in enumerate(filtered):
        if "receptor=loadPDB" in ln:
            insert_at = i
            break
if insert_at is None:
    insert_at = len(filtered)

to_insert = [f"loadAmberParams {p}\n" for p in frcmods]
out = filtered[:insert_at] + to_insert + filtered[insert_at:]
leap.write_text("".join(out))
print(f"[setup] patched leap.in to load {len(frcmods)} MCPB frcmod(s)")
PY
  )
fi

if [[ "$skip_tleap" == "1" ]]; then
  echo "[setup] --skip-tleap: templates prepared, but tleap not executed."
  echo "[setup] Next: run MCPB.py for Zn sites (if any), patch leap.in to load frcmod/lib, then run: (cd \"$run_dir\" && tleap -s -f leap.in)"
else
  (cd "$run_dir" && tleap -s -f leap.in)
fi

echo "[done] Generated (expected): complex.parm7, complex.rst7"
echo "Next:"
echo "  cd \"$run_dir\""
echo "  bash MD_run.in"
