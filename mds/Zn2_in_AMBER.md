## Zn2+ in AMBER: simplest reasonable options (for NSD2 SET / PDB 9CVD-like sites)

Your NSD2 SET-domain construct is very likely to contain a *structural* Zn site (Cys/His coordination). For such sites, treating Zn as a plain “free ion” (nonbonded only) often breaks the correct coordination geometry during MD.

Below are 3 practical options, from most-recommended to most-hacky.

### Option A (recommended): MCPB.py bonded model (AmberTools, can use `sqm`)
This is usually the **simplest that is still physically reasonable** for a fixed Zn site.

For your PDB 9CVD neighbor scan (3.0 Å), all three Zn are **Cys4 tetrahedral sites**:
- `ZN A1301` coordinated by `CYS A1144/1191/1193/1198` (SG ~1.97–1.99 Å)
- `ZN A1302` coordinated by `CYS A1016/1018/1026/1032` (SG ~2.15–2.60 Å)
- `ZN A1303` coordinated by `CYS A1026/1041/1046/1052` (SG ~2.19–2.39 Å)

Note: `CYS A1026` appears within cutoff for both `ZN A1302` and `ZN A1303` (2.60 Å vs 2.19 Å). If it is truly a bridging ligand, treat `ZN A1302 + ZN A1303` as **one multinuclear site** in MCPB.py (single group). If you re-check with a tighter cutoff (e.g. 2.5 Å) and `CYS A1026` drops from `ZN A1302`, you can model them as two independent sites.

1) Start from a cleaned PDB that keeps Zn and the coordinating residues (Cys/His/Asp/Glu).
2) Make sure protonation is consistent with coordination:
   - Cys coordinating Zn is typically **deprotonated** (`CYM` in AMBER).
   - Histidine must be `HID` (ND1) or `HIE` (NE2) depending on which N binds Zn.
3) Use MCPB.py to generate bonded parameters for the Zn site (use `sqm` if you don’t have Gaussian).
4) In `tleap`, load the generated `frcmod`/`lib` (and any additional files MCPB.py outputs), then `loadpdb` your protein+Zn.

Minimal MCPB.py pattern:

1) Create a working folder (inside a run dir is fine), copy in `protein_clean.pdb` (must include Zn + coordinating residues).
2) Write an `mcpb.in` describing: which Zn atom, which coordinating residues/atoms, cutoff, force field, and QM/semiempirical backend (`sqm` works without external Gaussian).
3) Run MCPB.py stages (exact stages/options depend on your AmberTools version):
   - `MCPB.py -i mcpb.in -s 1`
   - `MCPB.py -i mcpb.in -s 2`
4) In `leap.in`, load MCPB.py generated `frcmod`/`lib` *before* `loadpdb`.

What you get:
- Zn–ligand bonds/angles are explicitly parameterized so the tetrahedral/trigonal geometry stays stable.

What to verify after running:
- Zn–(S/N/O) distances stay near the crystallographic values (track with `cpptraj distance`).

### Option B (quick but risky): nonbonded Zn2+ + distance restraints
If you must avoid MCPB.py, you can keep Zn2+ as an ion (LJ + charge) and **apply NMR-style distance restraints** to the coordinating atoms to prevent the site from collapsing.

Pros:
- Fast to set up.
Cons:
- Geometry is enforced artificially; energetics of coordination are not correct.

### Option C (only if Zn is irrelevant): remove Zn
If Zn is far from your binding site and you only care about ligand stability elsewhere, removing Zn may be acceptable, but for SET/post-SET structural Zn this can destabilize the local fold and bias results.
