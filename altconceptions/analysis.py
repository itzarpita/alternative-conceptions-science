import numpy as np
import pandas as pd
from scipy.stats import pointbiserialr


def recompute_total(df: pd.DataFrame, item_cols):
    return df[item_cols].sum(axis=1)


def cronbach_alpha(items: pd.DataFrame) -> float:
    items = items.dropna()
    k = items.shape[1]
    if k < 2:
        return np.nan
    variances = items.var(axis=0, ddof=1)
    total_var = items.sum(axis=1).var(ddof=1)
    return (k / (k - 1)) * (1 - variances.sum() / total_var)


def kr20(items: pd.DataFrame) -> float:
    items = items.dropna()
    k = items.shape[1]
    if k < 2:
        return np.nan
    p = items.mean(axis=0)
    q = 1 - p
    total_var = items.sum(axis=1).var(ddof=1)
    return (k / (k - 1)) * (1 - (p * q).sum() / total_var)


def corrected_item_total_corr(items: pd.DataFrame) -> pd.Series:
    corrs = {}
    total = items.sum(axis=1)
    for col in items.columns:
        corrs[col] = items[col].corr(total - items[col])
    return pd.Series(corrs)


def point_biserial(item: pd.Series, total: pd.Series) -> float:
    try:
        return pointbiserialr(item, total)[0]
    except Exception:
        return np.nan


def cohen_d_paired(pre: pd.Series, post: pd.Series) -> float:
    diff = post - pre
    return diff.mean() / diff.std(ddof=1)


def hedges_g_from_d(d: float, n: int) -> float:
    return d * (1 - (3 / (4 * n - 9)))

def item_analysis(items: pd.DataFrame):
    total = items.sum(axis=1)
    difficulty = items.mean(axis=0)
    discrim = corrected_item_total_corr(items)

    out = pd.DataFrame({
        "difficulty_p": difficulty,
        "corr_item_total_corrected": discrim
    })
    return out.reset_index(names="item")

def reliability_report(items: pd.DataFrame, test_name: str) -> pd.DataFrame:
    alpha = cronbach_alpha(items)

    kr20_val = np.nan
    # KR-20 only if all items are binary 0/1
    if items.dropna().isin([0, 1]).all().all():
        kr20_val = kr20(items)

    return pd.DataFrame({
        "test": [test_name],
        "cronbach_alpha": [alpha],
        "kr20": [kr20_val]
    })


def gains_and_effects(pre_totals: pd.Series, post_totals: pd.Series) -> pd.DataFrame:
    """Calculate learning gains and effect sizes"""
    n = len(pre_totals)
    mean_pre = pre_totals.mean()
    mean_post = post_totals.mean()
    mean_gain = mean_post - mean_pre
    
    # Normalized gain (Hake's g)
    max_score = 10.0
    normalized_gain = (mean_post - mean_pre) / (max_score - mean_pre)
    
    # Effect sizes
    cohen_d = cohen_d_paired(pre_totals, post_totals)
    hedges_g = hedges_g_from_d(cohen_d, n)
    
    return pd.DataFrame({
        "mean_pre": [mean_pre],
        "mean_post": [mean_post],
        "mean_gain": [mean_gain],
        "normalized_gain_mean": [normalized_gain],
        "cohen_d_paired": [cohen_d],
        "hedges_g": [hedges_g],
        "n": [n]
    })


def concept_mapping_reports(merged_df: pd.DataFrame) -> pd.DataFrame:
    """
    Concept-level mapping aligned to your real columns:

    Pre: Q1..Q10 (binary 0/1)
    Post: P1 (1), P2 (2), P3i (1), P3ii (1), P4 (2), P5 (3)
    """
    rows = []

    # Seasons & tilt: Q1 vs P3i
    if "Q1" in merged_df.columns and "P3i" in merged_df.columns:
        rows.append({
            "concept": "Seasons & tilt (Q1 → P3i)",
            "pre_mean": merged_df["Q1"].mean(),
            "post_mean": merged_df["P3i"].mean(),
            "mean_gain": (merged_df["P3i"] - merged_df["Q1"]).mean()
        })

    # Astronomers: Q4 vs P3ii
    if "Q4" in merged_df.columns and "P3ii" in merged_df.columns:
        rows.append({
            "concept": "Working of astronomers (Q4 → P3ii)",
            "pre_mean": merged_df["Q4"].mean(),
            "post_mean": merged_df["P3ii"].mean(),
            "mean_gain": (merged_df["P3ii"] - merged_df["Q4"]).mean()
        })

    # Asteroids: Q7 vs P1
    if "Q7" in merged_df.columns and "P1" in merged_df.columns:
        rows.append({
            "concept": "Asteroids (Q7 → P1)",
            "pre_mean": merged_df["Q7"].mean(),
            "post_mean": merged_df["P1"].mean(),
            "mean_gain": (merged_df["P1"] - merged_df["Q7"]).mean()
        })

    # Planets block: Q5+Q6+Q8+Q9+Q10 vs P2 (different scales, report association)
    pre_planets_cols = [c for c in ["Q5", "Q6", "Q8", "Q9", "Q10"] if c in merged_df.columns]
    if pre_planets_cols and "P2" in merged_df.columns:
        pre_sum = merged_df[pre_planets_cols].sum(axis=1)
        rows.append({
            "concept": "Planets block (Q5+Q6+Q8+Q9+Q10 → P2)",
            "pre_mean_out_of_5": pre_sum.mean(),
            "post_mean_out_of_2": merged_df["P2"].mean(),
            "spearman_corr": merged_df[["P2"]].join(pre_sum.rename("pre_sum")).corr(method="spearman").loc["pre_sum", "P2"]
        })

    # Phases: Q2 vs P5 (scale mismatch)
    if "Q2" in merged_df.columns and "P5" in merged_df.columns:
        rows.append({
            "concept": "Phases of Moon (Q2 → P5)",
            "pre_mean": merged_df["Q2"].mean(),
            "post_mean_out_of_3": merged_df["P5"].mean(),
            "mean_gain_raw": (merged_df["P5"] - merged_df["Q2"]).mean()
        })

    # Luminosity: Q3 vs P4 (scale mismatch)
    if "Q3" in merged_df.columns and "P4" in merged_df.columns:
        rows.append({
            "concept": "Luminosity / stars vs planets (Q3 → P4)",
            "pre_mean": merged_df["Q3"].mean(),
            "post_mean_out_of_2": merged_df["P4"].mean(),
            "mean_gain_raw": (merged_df["P4"] - merged_df["Q3"]).mean()
        })

    return pd.DataFrame(rows)


def transition_2x2(pre_bin: pd.Series, post_bin: pd.Series) -> np.ndarray:
    """
    2x2 counts:
      rows = pre (0,1)
      cols = post (0,1)
    """
    a = pre_bin.fillna(0).astype(int)
    b = post_bin.fillna(0).astype(int)

    counts = np.zeros((2, 2), dtype=int)
    for i in (0, 1):
        for j in (0, 1):
            counts[i, j] = int(((a == i) & (b == j)).sum())
    return counts