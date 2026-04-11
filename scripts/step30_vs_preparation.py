#!/usr/bin/env python3
"""
Industrial-scale Virtual Screening Filtering Pipeline

Input : Parquet (SMILES + zinc_id)
Output: Filtered drug-like subset Parquet

Fixes:
- Correct Parquet streaming (pyarrow.iter_batches)
- tqdm progress support
- RDKit + PAINS filtering
- Memory-safe chunk processing

python scripts/step30_vs_preparation.py
"""

from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams


# PAINS filter (init once)
def init_pains_filter():
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
    return FilterCatalog(params)


pains_filter = init_pains_filter()


# Fast prefilter
def fast_filters(smiles_list):
    # heuristic: remove very long SMILES early
    return [len(s) < 120 for s in smiles_list]


# RDKit full filter
def rdkit_filter(smiles: str) -> bool:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False

    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = rdMolDescriptors.CalcNumHBD(mol)
    hba = rdMolDescriptors.CalcNumHBA(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)

    # Lipinski / Veber-like rules
    if mw > 500 or logp > 5 or hbd > 5 or hba > 10 or tpsa > 140:
        return False

    # PAINS filter
    if pains_filter.HasMatch(mol):
        return False

    return True



# Main pipeline
def vs_filter_pipeline(
    input_parquet: str,
    output_parquet: str,
    chunksize: int = 200_000
):

    pf = pq.ParquetFile(input_parquet)

    writer = None

    total_batches = pf.num_row_groups

    for i, batch in tqdm(
        enumerate(pf.iter_batches(batch_size=chunksize)),
        total=total_batches,
        desc="Filtering"
    ):

        chunk = batch.to_pandas()

        smiles = chunk["smiles"].tolist()


        # Stage 1: fast filter
        mask_fast = fast_filters(smiles)
        chunk = chunk[mask_fast]

        if len(chunk) == 0:
            continue

        # Stage 2: RDKit filter
        mask_rdkit = [rdkit_filter(s) for s in chunk["smiles"]]
        chunk = chunk[mask_rdkit]

        if len(chunk) == 0:
            continue

        # Type enforcement
        chunk["zinc_id"] = chunk["zinc_id"].astype("int64")
        chunk["smiles"] = chunk["smiles"].astype("string")

        # Write parquet stream
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


# Run
if __name__ == "__main__":
    vs_filter_pipeline(
        input_parquet="./data/database/all_zinc_combined.parquet",
        output_parquet="./data/database/zinc_filtered.parquet",
        chunksize=200_000
    )
