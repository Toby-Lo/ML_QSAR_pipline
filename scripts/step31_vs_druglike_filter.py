#!/usr/bin/env python3
"""
Stage 2: Strict Drug-like Filtering Pipeline (with QED)

Input : zinc_filtered.parquet
Output: zinc_druglike.parquet
"""

from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors, QED

# Silence RDKit warnings
RDLogger.DisableLog("rdApp.*")

# Allowed atoms
ALLOWED_ATOMS = {"C", "H", "N", "O", "S", "P", "F", "Cl", "Br", "I"}


# Safe mol
def safe_mol(smiles: str):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        Chem.SanitizeMol(mol, catchErrors=True)
        return mol
    except:
        return None

# Drug-like filter (STRICT)
def druglike_filter(smiles: str) -> bool:

    mol = safe_mol(smiles)
    if mol is None:
        return False

    # 1. Remove metals
    for atom in mol.GetAtoms():
        if atom.GetSymbol() not in ALLOWED_ATOMS:
            return False
        
    # 2. Size filter
    if mol.GetNumAtoms() < 10:
        return False

    # 3. PhysChem filters
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = rdMolDescriptors.CalcNumHBD(mol)
    hba = rdMolDescriptors.CalcNumHBA(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    rotb = rdMolDescriptors.CalcNumRotatableBonds(mol)

    if not (250 <= mw <= 500):
        return False

    if not (-1 <= logp <= 5):
        return False

    if not (20 <= tpsa <= 140):
        return False

    if hbd > 5 or hba > 10:
        return False

    if rotb > 10:
        return False

    # 4. QED filter
    qed = QED.qed(mol)

    if qed < 0.4:
        return False

    return True


def stage2_pipeline(
    input_parquet: str,
    output_parquet: str,
    chunksize: int = 200_000
):

    pf = pq.ParquetFile(input_parquet)
    writer = None

    for i, batch in tqdm(
        enumerate(pf.iter_batches(batch_size=chunksize)),
        desc="Stage2 Filtering"
    ):

        chunk = batch.to_pandas()

        mask = [druglike_filter(s) for s in chunk["smiles"]]
        chunk = chunk[mask]

        if len(chunk) == 0:
            continue

        chunk["zinc_id"] = chunk["zinc_id"].astype("int64")
        chunk["smiles"] = chunk["smiles"].astype("string")

        table = pa.Table.from_pandas(chunk, preserve_index=False)

        if writer is None:
            writer = pq.ParquetWriter(
                output_parquet,
                table.schema,
                compression="zstd"
            )

        writer.write_table(table)

        print(f"[Chunk {i}] kept: {len(chunk):,}")

    if writer:
        writer.close()

    print(f"\nDone → {output_parquet}")


if __name__ == "__main__":
    stage2_pipeline(
        input_parquet="./data/database/zinc_filtered.parquet",
        output_parquet="./data/database/zinc_druglike.parquet",
        chunksize=200_000
    )
