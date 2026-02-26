import os
import numpy as np
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from altconceptions.io import load_and_standardize
from altconceptions.anonymize import anonymize
from altconceptions.analysis import (
    item_analysis,
    reliability_report, gains_and_effects,
    concept_mapping_reports, transition_2x2
)
from altconceptions.analysis import recompute_total
from altconceptions.reporting import print_terminal_summary
from altconceptions.plots import (
    save_hist_with_stats,
    save_boxplot,
    save_ecdf_compare,
    save_pre_post_paired_scatter,
    save_scatter,
    save_transition_heatmap,
    save_concept_gains_barplot, 
    save_gain_distribution_plot, 
    save_correlation_matrix
)

# Add to your analyze.py imports
from altconceptions.enhanced_analysis import (
    statistical_tests, 
    correlation_analysis, 
    concept_level_analysis,
    compute_normalized_gain,
    generate_comprehensive_report
)

# --- NEW: Enhanced visualizations ---
def main(pre_path, post_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    reports = os.path.join(out_dir, "reports")
    figs = os.path.join(out_dir, "figures")
    os.makedirs(reports, exist_ok=True)
    os.makedirs(figs, exist_ok=True)

    pre_ld = load_and_standardize(pre_path)
    post_ld = load_and_standardize(post_path)

    pre_anon, mapping = anonymize(pre_ld.df, pre_ld.id_col, pre_ld.name_col)
    post_anon, _ = anonymize(post_ld.df, post_ld.id_col, post_ld.name_col)

    pre_items = [c for c in pre_anon.columns if c.startswith("Q")]
    post_items = ["P1", "P2", "P3i", "P3ii", "P4", "P5"]
    post_items = [c for c in post_items if c in post_anon.columns]

    required_post = {"P1","P2","P3i","P3ii","P4","P5"}
    missing = required_post - set(post_items)
    if missing:
        raise ValueError(f"Post-test items missing after rename: {sorted(missing)}")

    pre_anon["pre_total"] = recompute_total(pre_anon, pre_items)
    post_anon["post_total"] = recompute_total(post_anon, post_items)

    # Defaults (avoid unbound locals if any downstream block is refactored)
    mean_normalized_gain_individual = np.nan
    mean_normalized_gain_class = np.nan
    r_pre_norm, p_pre_norm = np.nan, np.nan

    merged = pd.merge(pre_anon, post_anon, on="participant_id")

    merged["pre_total"] = pd.to_numeric(merged["pre_total"], errors="coerce")
    merged["post_total"] = pd.to_numeric(merged["post_total"], errors="coerce")
    merged = merged.dropna(subset=["pre_total", "post_total"])
    
    # Ensure gain exists early for downstream use
    merged["gain"] = merged["post_total"] - merged["pre_total"]

    # --- Normalized gain (Hake): compute ONCE early so it is available everywhere ---
    denom = (10 - merged["pre_total"]).replace(0, np.nan)
    normalized_gains = (merged["post_total"] - merged["pre_total"]) / denom
    merged["normalized_gain"] = normalized_gains

    # Pre-test performance quartiles (used in Fig: normalized gain by quartile + reviewer-proof checks)
    # pd.qcut produces ~equal-sized groups; with N=150 this will be 37/38 per quartile.
    # Tie-safe quartiles: rank breaks ties so groups are ~equal size (37/38 for N=150)
    merged["_pre_rank"] = merged["pre_total"].rank(method="first")
    merged["pre_quartile"] = pd.qcut(merged["_pre_rank"], 4, labels=["Q1", "Q2", "Q3", "Q4"])

    # Two common summaries of normalized gain
    mean_normalized_gain_individual = float(np.nanmean(normalized_gains))
    pre_mean = float(np.nanmean(merged["pre_total"]))
    post_mean = float(np.nanmean(merged["post_total"]))
    mean_normalized_gain_class = float((post_mean - pre_mean) / (10 - pre_mean)) if (10 - pre_mean) != 0 else np.nan

    # Correlation: pre_total vs normalized_gain
    _ng = merged[["pre_total", "normalized_gain"]].dropna()
    if len(_ng) >= 3:
        r_pre_norm, p_pre_norm = stats.pearsonr(_ng["pre_total"], _ng["normalized_gain"])
    else:
        r_pre_norm, p_pre_norm = np.nan, np.nan

    pre_item = item_analysis(pre_anon[pre_items])
    post_item = item_analysis(post_anon[post_items])

    rel = pd.concat([
        reliability_report(pre_anon[pre_items], "pre"),
        reliability_report(post_anon[post_items], "post")
    ])

    gains = gains_and_effects(merged["pre_total"], merged["post_total"])
    concept = concept_mapping_reports(merged)

    pre_anon.to_csv(os.path.join(out_dir, "pretest_anonymized.csv"), index=False)
    post_anon.to_csv(os.path.join(out_dir, "posttest_anonymized.csv"), index=False)
    merged.to_csv(os.path.join(out_dir, "merged_anonymized.csv"), index=False)
    mapping.to_csv(os.path.join(out_dir, "id_mapping_private.csv"), index=False)

    pre_item.to_csv(os.path.join(reports, "pre_item_analysis.csv"), index=False)
    post_item.to_csv(os.path.join(reports, "post_item_analysis.csv"), index=False)
    rel.to_csv(os.path.join(reports, "reliability_report.csv"), index=False)
    gains.to_csv(os.path.join(reports, "gains_and_effects.csv"), index=False)
    concept.to_csv(os.path.join(reports, "concept_mapping_report.csv"), index=False)

    # --- Core distributions ---
    save_hist_with_stats(pre_anon["pre_total"], os.path.join(figs, "pre_total_hist.png"), "Pre-test Total", bins=10)
    save_hist_with_stats(post_anon["post_total"], os.path.join(figs, "post_total_hist.png"), "Post-test Total", bins=10)

    # Raw gain (ensure column exists)
    if "gain" not in merged.columns:
        merged["gain"] = merged["post_total"] - merged["pre_total"]
    gain = merged["gain"]

    save_hist_with_stats(gain, os.path.join(figs, "gain_hist.png"), "Learning Gain (Post − Pre)", bins=10)

    # --- Boxplots (quick story) ---
    save_boxplot(
        [pre_anon["pre_total"].astype(float), post_anon["post_total"].astype(float), gain.astype(float)],
        ["Pre", "Post", "Gain"],
        os.path.join(figs, "box_pre_post_gain.png"),
        "Pre/Post/Gain Distributions",
        ylabel="Marks"
    )

    # --- ECDF comparison (better than bins) ---
    save_ecdf_compare(
        pre_anon["pre_total"], post_anon["post_total"],
        labels=("Pre", "Post"),
        path=os.path.join(figs, "ecdf_pre_vs_post.png"),
        title="ECDF: Pre vs Post Totals",
        xlabel="Total marks"
    )

    # Ensure gain column exists (guard for code refactoring)
    if "gain" not in merged.columns:
        merged["gain"] = merged["post_total"] - merged["pre_total"]
    gain = merged["gain"]

    # --- Paired relationship plots ---
    save_pre_post_paired_scatter(
        merged["pre_total"], merged["post_total"],
        os.path.join(figs, "scatter_pre_vs_post.png"),
        "Paired Scatter: Pre vs Post"
    )

    save_scatter(
        merged["pre_total"], gain,
        os.path.join(figs, "scatter_pre_vs_gain.png"),
        "Who Improved Most? Gain vs Pre",
        xlabel="Pre total",
        ylabel="Gain (Post − Pre)"
    )
    
    transition_rows = []

    # Binary concept mappings only (0/1 on both sides)
    binary_pairs = [
        ("Seasons & tilt (Q1 → P3i)", "Q1", "P3i"),
        ("Working of astronomers (Q4 → P3ii)", "Q4", "P3ii"),
        ("Asteroids (Q7 → P1)", "Q7", "P1"),
    ]

    for label, pre_col, post_col in binary_pairs:
        if pre_col in merged.columns and post_col in merged.columns:
            mat = transition_2x2(merged[pre_col], merged[post_col])
            save_transition_heatmap(mat, os.path.join(figs, f"transition_{pre_col}_to_{post_col}.png"), label)

            n = mat.sum()
            transition_rows.append({
                "concept": label,
                "pre_col": pre_col,
                "post_col": post_col,
                "n": int(n),
                "pre0_post0": int(mat[0, 0]),
                "pre0_post1": int(mat[0, 1]),
                "pre1_post0": int(mat[1, 0]),
                "pre1_post1": int(mat[1, 1]),
                "persistence_rate_given_wrong": float(mat[0, 0] / (mat[0].sum() if mat[0].sum() else np.nan)),
                "correction_rate_given_wrong": float(mat[0, 1] / (mat[0].sum() if mat[0].sum() else np.nan)),
                "drop_rate_given_right": float(mat[1, 0] / (mat[1].sum() if mat[1].sum() else np.nan)),
                "retention_rate_given_right": float(mat[1, 1] / (mat[1].sum() if mat[1].sum() else np.nan)),
            })

    pd.DataFrame(transition_rows).to_csv(os.path.join(reports, "concept_transitions_binary.csv"), index=False)

    # Enhanced analysis
    comprehensive_report = generate_comprehensive_report(merged)
    
    # Save comprehensive report
    import json
    with open(os.path.join(reports, "comprehensive_analysis.json"), "w") as f:
        json.dump(comprehensive_report, f, indent=2)
    
    # Create a summary CSV
    summary_df = pd.DataFrame({
        'statistic': [
            'pre_test_mean', 'pre_test_std', 'post_test_mean', 'post_test_std',
            'mean_gain', 't_statistic', 'p_value', 'cohens_d', 'hedges_g',
            'improved_pct', 'same_pct', 'declined_pct',
            'normalized_gain_mean_individual',
            'normalized_gain_mean_class',
            'correlation_pre_gain',
            'correlation_pre_normalized_gain',
            'p_value_pre_normalized_gain'
        ],
        'value': [
            comprehensive_report['descriptive_statistics']['pre_mean'],
            comprehensive_report['descriptive_statistics']['pre_std'],
            comprehensive_report['descriptive_statistics']['post_mean'],
            comprehensive_report['descriptive_statistics']['post_std'],
            comprehensive_report['descriptive_statistics']['mean_gain'],
            comprehensive_report['inferential_statistics']['t_statistic'],
            comprehensive_report['inferential_statistics']['p_value'],
            comprehensive_report['inferential_statistics']['cohens_d'],
            comprehensive_report['inferential_statistics']['hedges_g'],
            comprehensive_report['inferential_statistics']['improved_pct'],
            comprehensive_report['inferential_statistics']['same_pct'],
            comprehensive_report['inferential_statistics']['declined_pct'],
            mean_normalized_gain_individual,
            mean_normalized_gain_class,
            comprehensive_report['correlation_analysis']['pearson_pre_gain']['correlation'],
            float(r_pre_norm),
            float(p_pre_norm)
        ]
    })
    summary_df.to_csv(os.path.join(reports, "statistical_summary.csv"), index=False)
    
    # Enhanced concept analysis with more pairs
    concept_pairs_enhanced = [
        ("Seasons & tilt", "Q1", "P3i"),
        ("Working of astronomers", "Q4", "P3ii"),
        ("Asteroids", "Q7", "P1"),
        ("Planets block", ["Q5", "Q6", "Q8", "Q9", "Q10"], "P2"),
        ("Phases of Moon", "Q2", "P5"),
        ("Luminosity / stars vs planets", "Q3", "P4")
    ]
    
    concept_results = concept_level_analysis(merged, concept_pairs_enhanced)
    concept_results.to_csv(os.path.join(reports, "concept_level_analysis_detailed.csv"), index=False)

    # Print enhanced summary
    print_enhanced_summary(
        comprehensive_report,
        concept_results,
        mean_normalized_gain_individual=mean_normalized_gain_individual,
        mean_normalized_gain_class=mean_normalized_gain_class,
        r_pre_norm=r_pre_norm,
        p_pre_norm=p_pre_norm,
        merged=merged,
    )

    print_terminal_summary(
        pre_anon["pre_total"].mean(),
        post_anon["post_total"].mean(),
        rel,
        gains,
        pre_item,
        concept
    )
    # --- Cleaner concept gains plot (reviewer-friendly) ---
    def _col(df, *candidates):
        for c in candidates:
            if c in df.columns:
                return c
        raise ValueError(f"None of {candidates} found in columns: {df.columns.tolist()}")

    c_concept = _col(concept_results, "concept")

    # concept_level_analysis outputs should be pre_mean/post_mean/mean_gain, but we also
    # tolerate older/alternative column names to avoid silently falling back to the old plot.
    c_pre = _col(
        concept_results,
        "pre_mean", "pre",
        "pre_mean_out_of_5", "pre_mean_out_of_3", "pre_mean_out_of_2"
    )
    c_post = _col(
        concept_results,
        "post_mean", "post",
        "post_mean_out_of_5", "post_mean_out_of_3", "post_mean_out_of_2"
    )
    c_delta = _col(concept_results, "mean_gain", "delta", "gain", "mean_gain_raw")
    c_p = _col(concept_results, "p_value", "p")

    labels = concept_results[c_concept]
    pre_vals = concept_results[c_pre]
    post_vals = concept_results[c_post]
    deltas = concept_results[c_delta]
    pvals = concept_results[c_p] if c_p in concept_results.columns else [np.nan]*len(labels)

    x = np.arange(len(labels))
    width = 0.36

    # Larger canvas + more bottom margin for wrapped labels
    plt.figure(figsize=(13.5, 6.5))
    plt.bar(x - width/2, pre_vals, width, label="Pre-test", alpha=0.65)
    plt.bar(x + width/2, post_vals, width, label="Post-test", alpha=0.65)

    # Add headroom so Δ and significance markers never collide with the legend or top border
    _ymax = float(np.nanmax([np.nanmax(pre_vals), np.nanmax(post_vals)]))
    plt.ylim(0, _ymax + 0.6)

    plt.ylabel("Mean score")
    plt.xlabel("Concept area")
    # Wrapped labels are readable without steep rotation
    plt.xticks(x, labels, rotation=0, ha="center")

    # Annotate gain (Δ) label only (no significance stars; keep plot uncluttered)
    for i, (pre, post, delta, p) in enumerate(zip(pre_vals, post_vals, deltas, pvals)):
        mx = max(pre, post)
        y_delta = mx + 0.08
        # Δ label (keep the plot uncluttered; p-values are reported in the text/table)
        plt.text(i, y_delta, f"Δ={delta:.2f}", ha="center", va="bottom", fontsize=11, color="black")

    # Put legend above the plotting area so it never hides bars/annotations
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=2, frameon=True)
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.savefig(os.path.join(figs, "concept_gains_barplot.png"), dpi=300)
    plt.close()

    # Save normalized-gain distribution plot (Hake) -- single panel (no raw-gain duplication)
    _ng = merged["normalized_gain"].dropna().astype(float)

    plt.figure(figsize=(9, 5.5))
    plt.hist(_ng, bins=20, alpha=0.6, edgecolor="black")

    # Hake thresholds
    plt.axvline(0.3,     color='#1a6faf', linestyle='--',  linewidth=2.0, label=r"Medium gain ($g = 0.3$)")
    plt.axvline(0.7,     color='#c0392b', linestyle='-.',  linewidth=2.0, label=r"High gain ($g = 0.7$)")

    # Mean line
    ng_mean = float(np.nanmean(_ng)) if len(_ng) else np.nan
    if not np.isnan(ng_mean):
        plt.axvline(ng_mean, color='black',   linestyle=':',   linewidth=2.2,
           label=rf"Mean $\bar{{g}} = {ng_mean:.3f}$")

    plt.xlabel("Normalized gain (g)")
    plt.ylabel("Frequency")
    # No title here; keep it for the LaTeX caption to avoid redundancy.
    plt.legend(loc="upper left", frameon=True)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(figs, "gain_distribution_comparison.png"), dpi=300)
    plt.close()

    # Save correlation matrix (make figure + colorbar readable in the paper)
    pre_items = [c for c in pre_anon.columns if c.startswith("Q")]
    post_items = [c for c in post_anon.columns if c.startswith("P")]

    # Create merged dataframe with all items
    merged_all_items = pd.merge(
        pre_anon[["participant_id"] + pre_items + ["pre_total"]],
        post_anon[["participant_id"] + post_items + ["post_total"]],
        on="participant_id"
    )
    merged_all_items["gain"] = merged_all_items["post_total"] - merged_all_items["pre_total"]

    # Keep matrix readable: totals + items + gain
    corr_cols = ["pre_total", "post_total"] + pre_items + post_items + ["gain"]
    corr_cols = [c for c in corr_cols if c in merged_all_items.columns]

    # Only render if size is manageable for a single-page figure
    if len(corr_cols) <= 22:
        corr = merged_all_items[corr_cols].corr(method="pearson")

        # Wider figure and a full-height colorbar axis so the colorbar is not squashed.
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_axes([0.08, 0.08, 0.72, 0.84])  # main heatmap area
        cax = fig.add_axes([0.83, 0.12, 0.03, 0.76])  # full-height colorbar

        sns.heatmap(
            corr,
            ax=ax,
            cbar=True,
            cbar_ax=cax,
            vmin=-1,
            vmax=1,
            center=0,
            cmap="RdBu_r",
            annot=True,
            fmt=".2f",
            annot_kws={"size": 8},
            linewidths=0.2,
            linecolor="white",
            square=True
        )

        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="center")
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

        ax.set_title("Correlation matrix of test scores and items")
        ax.tick_params(axis="x", rotation=90)
        ax.tick_params(axis="y", rotation=0)

        fig.savefig(os.path.join(figs, "correlation_matrix.png"), dpi=300)
        plt.close(fig)

    # --- Additional insightful scatter plots ---

    # 1. Gain vs normalized gain
    save_scatter(
        merged["pre_total"],
        normalized_gains,
        os.path.join(figs, "scatter_pre_vs_normalized_gain.png"),
        "Normalized Gain vs Pre-test Score",
        xlabel="Pre-test total",
        ylabel="Normalized gain ⟨g⟩"
    )

    # 2. Post vs pre with performance categories
    def categorize_gain(pre, post):
        denom = (10 - pre)
        if denom == 0:
            return "Low gain"
        normalized = (post - pre) / denom
        if normalized > 0.7:
            return "High gain"
        elif normalized >= 0.3:
            return "Medium gain"
        else:
            return "Low gain"

    merged["gain_category"] = merged.apply(
        lambda row: categorize_gain(row["pre_total"], row["post_total"]), axis=1
    )

    # Save this categorized scatter
    plt.figure(figsize=(8, 6))
    colors = {"High gain": "green", "Medium gain": "orange", "Low gain": "red"}

    for category, color in colors.items():
        subset = merged[merged["gain_category"] == category]
        plt.scatter(subset["pre_total"], subset["post_total"],
                   c=color, label=category, alpha=0.7, s=50)

    # Add equality line
    mn = min(merged["pre_total"].min(), merged["post_total"].min())
    mx = max(merged["pre_total"].max(), merged["post_total"].max())
    plt.plot([mn, mx], [mn, mx], 'k--', alpha=0.5, label='No change')

    plt.xlabel("Pre-test total")
    plt.ylabel("Post-test total")
    plt.title("Pre vs Post Scores with Gain Categories")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(figs, "scatter_gain_categories.png"), dpi=300)
    plt.close()

    # 3. Boxplot of normalized gains by pre-test quartile
    # merged["pre_quartile"] = pd.qcut(merged["pre_total"], 4, labels=["Q1", "Q2", "Q3", "Q4"])  # Already computed above

    plt.figure(figsize=(10, 6))
    data = [merged[merged["pre_quartile"] == q]["normalized_gain"] for q in ["Q1", "Q2", "Q3", "Q4"]]  # NOW CORRECT
    plt.boxplot(data, labels=["Lowest 25%", "25-50%", "50-75%", "Highest 25%"])
    plt.xlabel("Pre-test performance quartile")
    plt.axhline(y=0.3, color='orange', linestyle='--', alpha=0.5, label='Medium gain threshold')
    plt.axhline(y=0.7, color='green', linestyle='--', alpha=0.5, label='High gain threshold')
    plt.ylabel("Normalized gain (g)")
    # Removed title to avoid redundancy with caption
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(figs, "boxplot_normalized_gain_by_quartile.png"), dpi=300)
    plt.close()

    # Save the quartile analysis to CSV
    quartile_summary = merged.groupby("pre_quartile").agg({
        "normalized_gain": ["mean", "std", "count"],
        "gain": ["mean", "std"]
    }).round(3)
    quartile_summary.to_csv(os.path.join(reports, "gain_by_pre_quartile.csv"))

def print_enhanced_summary(comprehensive_report, concept_results, *, mean_normalized_gain_individual=np.nan, mean_normalized_gain_class=np.nan, r_pre_norm=np.nan, p_pre_norm=np.nan, merged=None):
    # --- Additional reviewer-proof checks ---
    print("\nADDITIONAL CHECKS")
    print("-"*70)
    try:
        print(f"Mean normalized gain (mean of individual g_i): {float(mean_normalized_gain_individual):.3f}")
    except Exception:
        print("Mean normalized gain (mean of individual g_i): NA")
    try:
        print(f"Mean normalized gain (class-average from means): {float(mean_normalized_gain_class):.3f}")
    except Exception:
        print("Mean normalized gain (class-average from means): NA")
    try:
        print(f"Correlation pre-test vs normalized gain: r = {float(r_pre_norm):.3f}, p = {float(p_pre_norm):.4f}")
    except Exception:
        print("Correlation pre-test vs normalized gain: NA")

    # Quartile verification (based on pre-test total)
    if isinstance(merged, pd.DataFrame) and ("pre_quartile" in merged.columns):
        q_sizes = merged["pre_quartile"].value_counts().sort_index()
        print("\nPre-test quartile sizes (pd.qcut on pre_total):")
        print(q_sizes.to_string())

        q_medians = merged.groupby("pre_quartile")["normalized_gain"].median().sort_index()
        print("\nMedian normalized gain by quartile:")
        print(q_medians.to_string())

    """Print enhanced terminal summary with statistical significance"""
    print("="*70)
    print("ENHANCED DATA ANALYSIS SUMMARY")
    print("="*70)
    
    desc = comprehensive_report['descriptive_statistics']
    infer = comprehensive_report['inferential_statistics']
    norm_gain = comprehensive_report['normalized_gain_analysis']
    corr = comprehensive_report['correlation_analysis']
    
    print(f"\n1. DESCRIPTIVE STATISTICS:")
    print(f"   Pre-test:  M = {desc['pre_mean']:.2f}, SD = {desc['pre_std']:.2f}")
    print(f"   Post-test: M = {desc['post_mean']:.2f}, SD = {desc['post_std']:.2f}")
    print(f"   Gain:      M = {desc['mean_gain']:.2f}, SD = {desc['gain_std']:.2f}")
    
    print(f"\n2. INFERENTIAL STATISTICS:")
    print(f"   Paired t-test: t({infer['n_students']-1}) = {infer['t_statistic']:.3f}, p = {infer['p_value']:.4f}")
    print(f"   Effect sizes: Cohen's d = {infer['cohens_d']:.3f}, Hedges' g = {infer['hedges_g']:.3f}")
    
    print(f"\n3. STUDENT OUTCOMES:")
    print(f"   Improved: {infer['improved_count']} students ({infer['improved_pct']:.1f}%)")
    print(f"   No change: {infer['same_count']} students ({infer['same_pct']:.1f}%)")
    print(f"   Declined: {infer['declined_count']} students ({infer['declined_pct']:.1f}%)")
    
    print(f"\n4. NORMALIZED GAIN (Hake, 1998):")
    print(f"   Mean normalized gain: ⟨g⟩ = {norm_gain['mean_normalized_gain']:.3f}")
    print(f"   High gain (g > 0.7): {norm_gain['high_gain_count']} students ({norm_gain['high_gain_pct']:.1f}%)")
    print(f"   Medium gain (0.3 ≤ g ≤ 0.7): {norm_gain['medium_gain_count']} students ({norm_gain['medium_gain_pct']:.1f}%)")
    print(f"   Low gain (g < 0.3): {norm_gain['low_gain_count']} students ({norm_gain['low_gain_pct']:.1f}%)")
    
    print(f"\n5. CORRELATION ANALYSIS:")
    print(f"   Pre-test vs Gain: r = {corr['pearson_pre_gain']['correlation']:.3f}, p = {corr['pearson_pre_gain']['p_value']:.4f}")
    print(f"   Pre-test vs Post-test: r = {corr['pearson_pre_post']['correlation']:.3f}, p = {corr['pearson_pre_post']['p_value']:.4f}")
    
    print(f"\n6. CONCEPT-LEVEL ANALYSIS:")
    for _, row in concept_results.iterrows():
        if 'p_value' in row and not pd.isna(row['p_value']):
            sig = "***" if row['p_value'] < 0.001 else "**" if row['p_value'] < 0.01 else "*" if row['p_value'] < 0.05 else ""
            print(f"   {row['concept']}:")
            print(f"     Pre: {row['pre_mean']:.3f} → Post: {row['post_mean']:.3f} (Δ = {row['mean_gain']:.3f})")
            print(f"     t = {row['t_statistic']:.2f}, p = {row['p_value']:.3f}{sig}, d = {row.get('cohens_d', 'NA'):.2f}")
    
    print("="*70)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pre", required=True)
    ap.add_argument("--post", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    main(args.pre, args.post, args.out)