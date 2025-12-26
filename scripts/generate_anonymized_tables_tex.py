#!/usr/bin/env python3
"""
Generate a standalone LaTeX document with full anonymized tables
(Pre-test + Achievement/Post-test) from CSV files.

Expected inputs:
- pretest_anonymized.csv  columns: id, Q1..Q10, TOTAL (case-insensitive)
- posttest_anonymized.csv columns: id, P1, P2, P3i, P3ii, P4, P5, TOTAL (case-insensitive)

Usage:
python generate_anonymized_tables_tex.py \
  --pre data/processed/pretest_anonymized.csv \
  --post data/processed/posttest_anonymized.csv \
  --out docs/data/anonymized-marks.tex
"""

from __future__ import annotations

import argparse
import os
from typing import List, Tuple

import pandas as pd


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.replace("\u00a0", " ", regex=False)  # NBSP
        .str.replace("\t", " ", regex=False)
        .str.strip()
    )
    # also normalize for matching
    df.columns = [c.strip() for c in df.columns]
    return df


def _find_id_col(df: pd.DataFrame) -> str:
    # Prefer "id", "ID", "student_id", etc. If not found, take first col.
    candidates = [c for c in df.columns if c.lower() in ("id", "student_id", "anon_id", "sno")]
    return candidates[0] if candidates else df.columns[0]


def _ensure_columns(df: pd.DataFrame, required: List[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}\nAvailable columns: {list(df.columns)}")


def _coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _as_int_or_clean(x) -> str:
    # Produce clean cell text for LaTeX: integers without .0, blanks as empty
    if pd.isna(x):
        return ""
    try:
        # handle values like 1.0
        if float(x).is_integer():
            return str(int(float(x)))
        return str(x)
    except Exception:
        return str(x)


def _latex_escape_minimal(s: str) -> str:
    # IDs should be numeric; still escape minimal set just in case.
    # Do NOT over-escape to keep output readable.
    return (
        s.replace("\\", r"\textbackslash{}")
         .replace("&", r"\&")
         .replace("%", r"\%")
         .replace("#", r"\#")
         .replace("_", r"\_")
         .replace("{", r"\{")
         .replace("}", r"\}")
    )


def _make_longtable(
    title: str,
    caption: str,
    colspec: str,
    header_cells: List[str],
    rows: List[List[str]],
    continued_cols: int,
) -> str:
    """
    Build a longtable block. header_cells must match number of columns.
    """
    header_line = " & ".join([rf"\textbf{{{h}}}" for h in header_cells]) + r" \\"
    top = rf"""
\section*{{{title}}}

\begin{{longtable}}{{{colspec}}}
\caption{{{caption}}}\\
\toprule
{header_line}
\midrule
\endfirsthead

\toprule
{header_line}
\midrule
\endhead

\midrule
\multicolumn{{{continued_cols}}}{{r}}{{\textit{{Continued on next page}}}}\\
\endfoot

\bottomrule
\endlastfoot
""".lstrip("\n")

    body_lines = []
    for r in rows:
        body_lines.append(" & ".join(r) + r" \\")
    body = "\n".join(body_lines)

    tail = r"""
\end{longtable}
""".lstrip("\n")

    return top + body + tail


def build_tex(pre_path: str, post_path: str, out_path: str) -> None:
    pre = _normalize_columns(pd.read_csv(pre_path))
    post = _normalize_columns(pd.read_csv(post_path))

    pre_id = _find_id_col(pre)
    post_id = _find_id_col(post)

    # Standardize TOTAL column name if needed
    # (allow TOTAL, Total, total)
    def find_total(df: pd.DataFrame) -> str:
        for c in df.columns:
            if c.strip().lower() == "total":
                return c
        raise ValueError("No TOTAL column found (expected a column named TOTAL/Total/total).")

    pre_total_col = find_total(pre)
    post_total_col = find_total(post)

    # Rename to standard internal names
    pre = pre.rename(columns={pre_id: "ID", pre_total_col: "TOTAL"})
    post = post.rename(columns={post_id: "ID", post_total_col: "TOTAL"})

    # Pre expected: Q1..Q10
    pre_items = [f"Q{i}" for i in range(1, 11)]
    _ensure_columns(pre, ["ID", "TOTAL"] + pre_items)

    # Post expected: P1, P2, P3i, P3ii, P4, P5
    post_items = ["P1", "P2", "P3i", "P3ii", "P4", "P5"]
    _ensure_columns(post, ["ID", "TOTAL"] + post_items)

    # Coerce numeric (ID might be numeric, keep as string later)
    pre = _coerce_numeric(pre, pre_items + ["TOTAL"])
    post = _coerce_numeric(post, post_items + ["TOTAL"])

    # Sort by ID (numeric-safe)
    def sort_key(x):
        try:
            return int(x)
        except Exception:
            return str(x)

    pre = pre.sort_values(by="ID", key=lambda s: s.map(sort_key))
    post = post.sort_values(by="ID", key=lambda s: s.map(sort_key))

    # Build rows for LaTeX
    pre_rows = []
    for _, row in pre.iterrows():
        rid = _latex_escape_minimal(str(_as_int_or_clean(row["ID"])))
        cells = [rid] + [_as_int_or_clean(row[c]) for c in pre_items] + [_as_int_or_clean(row["TOTAL"])]
        pre_rows.append(cells)

    post_rows = []
    for _, row in post.iterrows():
        rid = _latex_escape_minimal(str(_as_int_or_clean(row["ID"])))
        cells = [rid] + [_as_int_or_clean(row[c]) for c in post_items] + [_as_int_or_clean(row["TOTAL"])]
        post_rows.append(cells)

    # LaTeX document wrapper
    tex_parts = []
    tex_parts.append(r"""\documentclass[12pt]{article}

\usepackage[a4paper,margin=1in]{geometry}
\usepackage{longtable}
\usepackage{booktabs}
\usepackage{caption}
\usepackage{array}

\setlength{\parindent}{0pt}

\begin{document}

\begin{center}
    \Large \textbf{Anonymized Student Performance Data} \\[0.25cm]
    \normalsize Diagnostic Pre-Test and Achievement (Post) Test Scores \\[0.1cm]
\end{center}

\vspace{0.4cm}

\textbf{Ethical Note:}
All student data presented in this document have been anonymized. No names, roll numbers,
or personally identifiable information are included. The anonymized IDs are consistent across tests
to allow paired analysis.

\vspace{0.6cm}
""")

    # Pre table
    # 12 columns total: ID + Q1..Q10 + Total
    pre_header = ["ID"] + pre_items + ["Total"]
    pre_colspec = "c " + " ".join(["c"] * 10) + " c"
    tex_parts.append(_make_longtable(
        title="Table 1: Anonymized Pre-Test Scores",
        caption="Anonymized Pre-Test Item Scores and Total (Max = 10)",
        colspec=pre_colspec,
        header_cells=pre_header,
        rows=pre_rows,
        continued_cols=len(pre_header),
    ))

    tex_parts.append(r"\clearpage" + "\n")

    # Post table
    # 8 columns total: ID + P1,P2,P3i,P3ii,P4,P5 + Total
    post_header = ["ID"] + post_items + ["Total"]
    post_colspec = "c " + " ".join(["c"] * 6) + " c"
    tex_parts.append(_make_longtable(
        title="Table 2: Anonymized Achievement/Post-Test Scores",
        caption="Anonymized Post-Test Item Scores and Total (Max = 10)",
        colspec=post_colspec,
        header_cells=post_header,
        rows=post_rows,
        continued_cols=len(post_header),
    ))

    tex_parts.append(r"""
\vspace{0.5cm}

\textbf{Note:}
These anonymized datasets correspond to the CSV files used in the analysis pipeline of this repository.

\end{document}
""".lstrip("\n"))

    tex = "\n".join(tex_parts)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(tex)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pre", required=True, help="Path to pretest_anonymized.csv")
    ap.add_argument("--post", required=True, help="Path to posttest_anonymized.csv")
    ap.add_argument("--out", required=True, help="Output .tex path (e.g., docs/data/anonymized-marks.tex)")
    args = ap.parse_args()

    build_tex(args.pre, args.post, args.out)
    print(f"[OK] Wrote LaTeX to: {args.out}")


if __name__ == "__main__":
    main()
