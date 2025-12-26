import pandas as pd
from dataclasses import dataclass
from typing import Tuple


@dataclass
class LoadedData:
    df: pd.DataFrame
    name_col: str
    id_col: str


def standardize_name_col(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    for col in df.columns:
        if col.lower().startswith("name"):
            return df, col
    raise ValueError("No name column found")


def load_and_standardize(path: str) -> LoadedData:
    df = pd.read_csv(path)

    # Normalize column names (kills hidden spaces / NBSP)
    df.columns = (
    df.columns.astype(str)
    .str.replace("\u00a0", " ", regex=False)
    .str.replace("\t", " ", regex=False)
    .str.strip()
    )

    # ID + name columns
    if "SNo" in df.columns:
        id_col = "SNo"
    else:
        id_col = df.columns[0]

    df, name_col = standardize_name_col(df)

    # ---- Normalize post-test column names (map to P1..P5) ----
    # Run THIS FIRST so pre-test renaming never touches post-test files.
    if "Q3i(1M)" in df.columns and "Q3ii(1M)" in df.columns:
        df = df.rename(columns={
            "Q1 (1M)": "P1",
            "Q2 (2M)": "P2",
            "Q3i(1M)": "P3i",
            "Q3ii(1M)": "P3ii",
            "Q4 (2M)": "P4",
            "Q5 (3M)": "P5",
            "TOTAL(10M)": "TOTAL",
        })

    # ---- Normalize pre-test column names ----
    elif "Q10(1M)" in df.columns or "TOTAL (10M)" in df.columns:
        df = df.rename(columns={
            "Q1 (1M)": "Q1",
            "Q2 (1M)": "Q2",
            "Q3 (1M)": "Q3",
            "Q4 (1M)": "Q4",
            "Q5 (1M)": "Q5",
            "Q6 (1M)": "Q6",
            "Q7 (1M)": "Q7",
            "Q8 (1M)": "Q8",
            "Q9 (1M)": "Q9",
            "Q10(1M)": "Q10",
            "TOTAL (10M)": "TOTAL",
        })

    # Coerce numeric for item columns + total if present
    for c in df.columns:
        if c in ["Q1","Q2","Q3","Q4","Q5","Q6","Q7","Q8","Q9","Q10","P1","P2","P3i","P3ii","P4","P5","TOTAL"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return LoadedData(df=df, name_col=name_col, id_col=id_col)