import pandas as pd


def hr(title):
    print("\n" + "=" * 70)
    print(title.upper())
    print("=" * 70)


def fmt(x, nd=3):
    if pd.isna(x):
        return "NA"
    return f"{x:.{nd}f}"


def print_terminal_summary(pre_mean, post_mean, rel_df, gains_df, item_df, concept_df):
    hr("DATA SUMMARY")
    print(f"Pre-test mean score   : {fmt(pre_mean)} / 10")
    print(f"Post-test mean score  : {fmt(post_mean)} / 10")

    hr("RELIABILITY OF TESTS")
    for _, r in rel_df.iterrows():
        print(f"{r['test']} Cronbach α : {fmt(r['cronbach_alpha'])}")
        if "kr20" in r:
            print(f"{r['test']} KR-20      : {fmt(r['kr20'])}")

    hr("OVERALL LEARNING GAIN")
    g = gains_df.iloc[0]
    print(f"Mean gain             : {fmt(g['mean_gain'])}")
    print(f"Normalized gain ⟨g⟩   : {fmt(g['normalized_gain_mean'])}")
    print(f"Cohen’s d             : {fmt(g['cohen_d_paired'])}")
    print(f"Hedges’ g             : {fmt(g['hedges_g'])}")

    hr("MAJOR PRE-TEST MISCONCEPTIONS")
    hardest = item_df.sort_values("difficulty_p").head(5)
    for _, r in hardest.iterrows():
        print(f"{r['item']} | p = {fmt(r['difficulty_p'])}")

    hr("CONCEPT-LEVEL LEARNING")
    for _, r in concept_df.iterrows():
        print(r["concept"])
        for k, v in r.items():
            if k == "concept" or pd.isna(v):
                continue
            print(f"  {k}: {fmt(v)}")