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
    plt.hist(x, bins=bins, alpha=0.7, edgecolor='black')
    plt.title(title)
    plt.xlabel("Score")
    plt.ylabel("Count")

    mean = x.mean()
    med = x.median()

    # Use different colors and linestyles for mean and median
    plt.axvline(mean, color='red', linestyle='-', linewidth=2, alpha=0.8, label=f'Mean = {mean:.2f}')
    plt.axvline(med, color='green', linestyle='--', linewidth=2, alpha=0.8, label=f'Median = {med:.2f}')

    # Add legend
    plt.legend()
    
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

# Add to plots.py
def save_concept_gains_barplot(concept_results, path, title="Concept-Specific Learning Gains"):
    """Create bar plot showing pre/post means for each concept"""
    _ensure_dir(path)
    
    concepts = concept_results['concept'].tolist()
    pre_means = concept_results['pre_mean'].tolist()
    post_means = concept_results['post_mean'].tolist()
    gains = concept_results['mean_gain'].tolist()
    
    x = np.arange(len(concepts))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, pre_means, width, label='Pre-test', alpha=0.8, color='skyblue')
    bars2 = ax.bar(x + width/2, post_means, width, label='Post-test', alpha=0.8, color='lightcoral')
    
    # Add gain values on top
    for i, gain in enumerate(gains):
        ax.text(i, max(pre_means[i], post_means[i]) + 0.05, 
                f'Δ={gain:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # Add significance markers
    if 'p_value' in concept_results.columns:
        for i, p_val in enumerate(concept_results['p_value']):
            if p_val < 0.05:
                y_pos = max(pre_means[i], post_means[i]) + 0.15
                marker = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*'
                ax.text(i, y_pos, marker, ha='center', va='bottom', fontsize=14)
    
    ax.set_ylabel('Mean Score')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(concepts, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    fig.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()



def save_gain_distribution_plot(gains, normalized_gains, path, title="Distribution of Learning Gains"):
    """Create side-by-side histograms of raw and normalized gains"""
    _ensure_dir(path)
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Raw gains
    ax1.hist(gains, bins=15, alpha=0.7, edgecolor='black', color='steelblue')
    ax1.axvline(gains.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean = {gains.mean():.2f}')
    ax1.axvline(gains.median(), color='green', linestyle=':', linewidth=2, label=f'Median = {gains.median():.2f}')
    ax1.set_xlabel('Raw Gain (Post - Pre)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Raw Learning Gains')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add text with basic stats
    ax1.text(0.05, 0.95, f'N = {len(gains)}\nRange: [{gains.min():.1f}, {gains.max():.1f}]',
             transform=ax1.transAxes, va='top', 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Normalized gains with Hake categories
    bins = np.linspace(-1, 1, 21)
    ax2.hist(normalized_gains, bins=bins, alpha=0.7, edgecolor='black', color='lightcoral')
    
    # Add category lines and labels
    ax2.axvline(0.3, color='orange', linestyle='--', alpha=0.7, linewidth=2)
    ax2.axvline(0.7, color='orange', linestyle='--', alpha=0.7, linewidth=2)
    ax2.axvline(0, color='black', linestyle='-', alpha=0.3, linewidth=1)
    
    # Add category labels
    y_max = ax2.get_ylim()[1]
    ax2.text(-0.35, y_max*0.9, 'Low\ngain\n(g < 0.3)', 
             ha='center', fontsize=9, fontweight='bold', 
             bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.3))
    ax2.text(0.5, y_max*0.9, 'Medium\ngain\n(0.3 ≤ g ≤ 0.7)', 
             ha='center', fontsize=9, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='gold', alpha=0.3))
    ax2.text(0.85, y_max*0.9, 'High\ngain\n(g > 0.7)', 
             ha='center', fontsize=9, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    
    # Add mean line
    ax2.axvline(normalized_gains.mean(), color='red', linestyle='--', linewidth=2, 
                label=f'Mean = {normalized_gains.mean():.3f}')
    
    ax2.set_xlabel('Normalized Gain ⟨g⟩')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Normalized Learning Gains (Hake, 1998)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    fig.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

def save_correlation_matrix(merged_df, pre_items, post_items, path):
    """Create correlation matrix heatmap"""
    _ensure_dir(path)
    
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    
    # Select relevant columns
    corr_cols = ['pre_total', 'post_total'] + pre_items + post_items
    corr_cols = [c for c in corr_cols if c in merged_df.columns]
    
    # Add gain if not present
    if 'gain' not in merged_df.columns:
        merged_df = merged_df.copy()
        merged_df['gain'] = merged_df['post_total'] - merged_df['pre_total']
    corr_cols.append('gain')
    
    # Filter to existing columns
    existing_cols = [c for c in corr_cols if c in merged_df.columns]
    
    if len(existing_cols) < 3:
        print(f"Warning: Not enough columns for correlation matrix. Existing: {existing_cols}")
        return
    
    # Compute correlation matrix
    corr_matrix = merged_df[existing_cols].corr()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(max(10, len(corr_cols)*0.8), max(8, len(corr_cols)*0.6)))
    
    # Use seaborn for better heatmap
    sns.heatmap(corr_matrix, 
                annot=True, 
                fmt=".2f", 
                cmap="coolwarm", 
                center=0,
                square=True,
                linewidths=0.5,
                cbar_kws={"shrink": 0.8},
                ax=ax)
    
    # Rotate x-axis labels
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    
    ax.set_title("Correlation Matrix of Test Scores and Items", fontsize=14, fontweight='bold')
    
    fig.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()