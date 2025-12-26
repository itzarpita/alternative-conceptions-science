import numpy as np
import pandas as pd
from scipy.stats import pointbiserialr


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