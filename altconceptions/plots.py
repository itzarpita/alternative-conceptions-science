import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _ensure_dir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def save_hist_with_stats(series: pd.Series, path: str, title: str, bins: int = 10):
    _ensure_dir(path)
    x = series.dropna().astype(float)

    plt.figure()
    plt.hist(x, bins=bins)
    plt.title(title)
    plt.xlabel("Score")
    plt.ylabel("Count")

    mean = x.mean()
    med = x.median()

    # mean/median reference lines (default matplotlib colors)
    plt.axvline(mean)
    plt.axvline(med)

    y_max = plt.ylim()[1]
    plt.text(mean, 0.95 * y_max, f"mean={mean:.2f}", rotation=90, va="top")
    plt.text(med, 0.95 * y_max, f"median={med:.2f}", rotation=90, va="top")

    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


# Backward compatibility: older code may still import save_hist
def save_hist(series: pd.Series, path: str, title: str, bins: int = 10):
    return save_hist_with_stats(series, path, title, bins=bins)


def save_boxplot(data: list, labels: list, path: str, title: str, ylabel: str):
    _ensure_dir(path)
    plt.figure()
    plt.boxplot(data, labels=labels, showmeans=True)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def save_ecdf_compare(a: pd.Series, b: pd.Series, labels: tuple, path: str, title: str, xlabel: str = "Score"):
    _ensure_dir(path)
    a = np.sort(a.dropna().astype(float).values)
    b = np.sort(b.dropna().astype(float).values)
    ya = np.arange(1, len(a) + 1) / len(a)
    yb = np.arange(1, len(b) + 1) / len(b)

    plt.figure()
    plt.step(a, ya, where="post", label=labels[0])
    plt.step(b, yb, where="post", label=labels[1])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("ECDF")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def save_scatter(x: pd.Series, y: pd.Series, path: str, title: str, xlabel: str, ylabel: str):
    _ensure_dir(path)
    x = x.astype(float)
    y = y.astype(float)

    plt.figure()
    plt.scatter(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def save_pre_post_paired_scatter(pre: pd.Series, post: pd.Series, path: str, title: str):
    _ensure_dir(path)
    pre = pre.astype(float)
    post = post.astype(float)

    plt.figure()
    plt.scatter(pre, post)

    mn = min(pre.min(), post.min())
    mx = max(pre.max(), post.max())
    plt.plot([mn, mx], [mn, mx])

    plt.title(title)
    plt.xlabel("Pre total")
    plt.ylabel("Post total")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def save_transition_heatmap(counts_2x2: np.ndarray, path: str, title: str):
    _ensure_dir(path)
    mat = np.asarray(counts_2x2, dtype=float)

    plt.figure()
    plt.imshow(mat)
    plt.title(title)
    plt.xticks([0, 1], ["Post 0", "Post 1"])
    plt.yticks([0, 1], ["Pre 0", "Pre 1"])

    for i in range(2):
        for j in range(2):
            plt.text(j, i, f"{int(mat[i, j])}", ha="center", va="center")

    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()