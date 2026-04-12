#!/usr/bin/env python3
"""
Stage 3: Feature Store (aligned with training features)

Input : zinc_druglike.parquet
Output: zinc_features.parquet

This stage is aligned to the training pipeline in `scripts/step10_qsar_ml.py`:
- Morgan fingerprints: radius=2, nBits=2048, exported as `morgan_0..morgan_2047`
- RDKit descriptors: exported using the same `rdkit.Chem.Descriptors.<name>` functions

By default, descriptor names are loaded from `config/nsd2_ml.yaml` (field: `descriptor_names`).

Remember to delete df_desc_isna.columns as final Virtual Screening step to make sure as same as training
"""

import argparse
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
import numpy as np

from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import Descriptors

RDLogger.DisableLog("rdApp.*")

try:
    import yaml  # type: ignore
except Exception:
    yaml = None

try:
    from rdkit.Chem.rdMolDescriptors import MorganGenerator  # type: ignore
    _MORGAN_GENERATOR_AVAILABLE = True
except Exception:
    from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect  # type: ignore
    _MORGAN_GENERATOR_AVAILABLE = False

DEFAULT_DESCRIPTOR_NAMES = [
    "MolWt",
    "MolLogP",
    "NumHDonors",
    "NumHAcceptors",
    "NumRotatableBonds",
    "TPSA",
    "HeavyAtomCount",
    "NumValenceElectrons",
    "NumAliphaticRings",
    "NumAromaticRings",
    "FractionCSP3",
    "RingCount",
    "LabuteASA",
    "VSA_EState1",
    "VSA_EState2",
    "SlogP_VSA1",
    "SlogP_VSA2",
    "SMR_VSA1",
    "SMR_VSA2",
    "EState_VSA1",
]


def safe_mol(smiles: str) -> Optional[Chem.Mol]:
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return mol
    except Exception:
        return None


def load_descriptor_names(config_path: Path) -> List[str]:
    if config_path.exists() and yaml is not None:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        names = cfg.get("descriptor_names")
        if isinstance(names, list) and all(isinstance(x, str) for x in names) and names:
            return [str(x) for x in names]
    return DEFAULT_DESCRIPTOR_NAMES.copy()


def build_descriptor_funcs(descriptor_names: List[str]) -> Dict[str, Callable[[Chem.Mol], float]]:
    funcs: Dict[str, Callable[[Chem.Mol], float]] = {}
    for name in descriptor_names:
        func = getattr(Descriptors, name, None)
        if func is None:
            raise AttributeError(f"Descriptor not found in rdkit.Chem.Descriptors: {name}")
        funcs[name] = func
    return funcs


def compute_descriptors(
    mol: Chem.Mol,
    descriptor_funcs: Dict[str, Callable[[Chem.Mol], float]],
    descriptor_names: List[str],
) -> Optional[np.ndarray]:
    try:
        out = np.full((len(descriptor_names),), np.nan, dtype=np.float32)
        for idx, name in enumerate(descriptor_names):
            func = descriptor_funcs[name]
            try:
                out[idx] = float(func(mol))
            except Exception:
                out[idx] = np.nan
        return out
    except Exception:
        return None


def make_morgan_generator(radius: int = 2, n_bits: int = 2048):
    if _MORGAN_GENERATOR_AVAILABLE:
        return MorganGenerator(radius=radius, nBits=n_bits)
    return None


def compute_morgan_fp(mol: Chem.Mol, generator, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    if generator is not None:
        fp = generator.GetFingerprintAsBitVect(mol)
    else:
        fp = GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)  # type: ignore[name-defined]
    return np.asarray(fp, dtype=np.uint8)


def featurize(
    smiles: str,
    fp_generator,
    descriptor_funcs: Dict[str, Callable[[Chem.Mol], float]],
    descriptor_names: List[str],
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    mol = safe_mol(smiles)
    if mol is None:
        return None

    try:
        fp = compute_morgan_fp(mol, generator=fp_generator, radius=2, n_bits=2048)
    except Exception:
        return None

    desc = compute_descriptors(mol, descriptor_funcs, descriptor_names)
    if desc is None:
        return None

    return fp, desc


def stage3_pipeline(
    input_parquet: str,
    output_parquet: str,
    descriptor_names: List[str],
    chunksize: int = 100000,
):

    pf = pq.ParquetFile(input_parquet)
    writer = None
    descriptor_funcs = build_descriptor_funcs(descriptor_names)
    fp_generator = make_morgan_generator(radius=2, n_bits=2048)
    fp_col_names = [f"morgan_{i}" for i in range(2048)]

    for i, batch in tqdm(enumerate(pf.iter_batches(batch_size=chunksize)), desc="Stage3"):

        chunk = batch.to_pandas()

        zinc_ids: List[int] = []
        smiles_out: List[str] = []
        fp_rows: List[np.ndarray] = []
        desc_rows: List[np.ndarray] = []

        for row in chunk.itertuples(index=False):
            smiles = getattr(row, "smiles")
            zinc_id = getattr(row, "zinc_id")
            if not isinstance(smiles, str) or not smiles:
                continue

            result = featurize(smiles, fp_generator, descriptor_funcs, descriptor_names)
            if result is None:
                continue

            fp, desc = result

            try:
                zinc_ids.append(int(zinc_id))
            except Exception:
                continue
            smiles_out.append(smiles)
            fp_rows.append(fp)
            desc_rows.append(desc)

        if len(zinc_ids) == 0:
            continue

        fp_matrix = np.stack(fp_rows, axis=0).astype(np.uint8, copy=False)
        desc_matrix = np.stack(desc_rows, axis=0).astype(np.float32, copy=False)

        df_out = pd.DataFrame({"zinc_id": zinc_ids, "smiles": smiles_out})
        df_fp = pd.DataFrame(fp_matrix, columns=fp_col_names, dtype=np.uint8)
        df_desc = pd.DataFrame(desc_matrix, columns=descriptor_names, dtype=np.float32)

        df_desc_isna = df_desc.isna().astype(np.uint8)
        df_desc_isna.columns = [f"{c}__isna" for c in descriptor_names]

        df_desc = df_desc.fillna(0.0)

        df_out = pd.concat([df_out, df_fp, df_desc, df_desc_isna], axis=1)

        df_out["zinc_id"] = df_out["zinc_id"].astype("int64")
        df_out["smiles"] = df_out["smiles"].astype("string")

        table = pa.Table.from_pandas(df_out, preserve_index=False)

        if writer is None:
            writer = pq.ParquetWriter(
                output_parquet,
                table.schema,
                compression="zstd"
            )

        writer.write_table(table)

        print(f"[Chunk {i}] processed: {len(df_out):,}")

    if writer:
        writer.close()

    print(f"\nDone → {output_parquet}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage3 feature store aligned with QSAR training features")
    parser.add_argument("--config", type=str, default="config/nsd2_ml.yaml", help="Training config (descriptor_names)")
    parser.add_argument("--input", type=str, default="./data/database/zinc_druglike.parquet")
    parser.add_argument("--output", type=str, default="./data/database/zinc_features.parquet")
    parser.add_argument("--chunksize", type=int, default=100000)
    args = parser.parse_args()

    descriptor_names = load_descriptor_names(Path(args.config))
    stage3_pipeline(
        input_parquet=args.input,
        output_parquet=args.output,
        descriptor_names=descriptor_names,
        chunksize=int(args.chunksize),
    )
