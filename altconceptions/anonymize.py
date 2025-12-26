import pandas as pd


def anonymize(df: pd.DataFrame, id_col: str, name_col: str):
    df = df.copy()

    df["participant_id"] = [
        f"P{str(i+1).zfill(3)}" for i in range(len(df))
    ]

    mapping = df[[id_col, name_col, "participant_id"]].copy()

    anon = df.drop(columns=[id_col, name_col])
    cols = ["participant_id"] + [c for c in anon.columns if c != "participant_id"]
    anon = anon[cols]

    return anon, mapping