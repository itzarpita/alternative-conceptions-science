import os
import numpy as np
import argparse
import pandas as pd
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
    save_transition_heatmap
)


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

    merged = pd.merge(pre_anon, post_anon, on="participant_id")

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

    gain = merged["post_total"] - merged["pre_total"]
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

    print_terminal_summary(
        pre_anon["pre_total"].mean(),
        post_anon["post_total"].mean(),
        rel,
        gains,
        pre_item,
        concept
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pre", required=True)
    ap.add_argument("--post", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    main(args.pre, args.post, args.out)