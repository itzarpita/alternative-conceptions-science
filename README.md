# Alternative Conceptions in Elementary Science

## Overview
This repository contains a complete, reproducible analysis of a classroom-based study titled:

**A Study of Alternative Conceptions in Science in Relation to the Achievement of Students at Elementary Level**

The focus is on diagnosing, tracking, and visualizing students’ alternative conceptions using pre-test and post-test data, rather than relying only on aggregate achievement scores.

The repository is designed for transparency, reproducibility, and future extension.

---

## Data Description

### Pre-test
- 10 MCQ items  
- 1 mark per item  
- Purpose: diagnose initial conceptions and misconceptions  

### Post-test
- Mixed-mark items (1M, 2M, 3M)  
- Conceptually aligned with pre-test items  
- Purpose: assess conceptual change at higher cognitive depth  

### Participants
- N = 150 elementary-level students  
- All individual-level data are anonymized  

Raw data:
```
data/raw/ (not included in the public repository)
```

Processed data:
```
data/processed/
```

---

## Analysis Pipeline

The complete analysis can be run using:

```
python -m scripts.analyze --pre data/raw/pretest.csv --post data/raw/posttest.csv --out data/processed
```

The pipeline performs:
1. Data cleaning and column standardization  
2. Score recomputation (pre and post)  
3. Reliability analysis (Cronbach’s α, KR-20 where applicable)  
4. Learning gain analysis  
   - Mean gain  
   - Normalized gain (Hake’s g)  
   - Effect sizes (Cohen’s d, Hedges’ g)  
5. Item difficulty and misconception identification  
6. Concept-level mapping between pre- and post-test items  
7. Automated visualization generation  

All numerical results are printed directly to the terminal in human-readable form.

---

## Visualizations

The following plots are generated automatically:

## Distribution Plots

- Pre-test, post-test, and gain histograms
- ECDF comparison (pre vs post)
- Boxplots (pre / post / gain)
- Comparative gain distributions

## Relationship Plots

- Paired scatter plot (pre vs post)
- Gain vs pre-score scatter (who improved most)
- Normalized gain by pre-test performance quartile
- Gain category scatter plot
- Concept Analysis

## Concept-level transition plots showing:

- Misconception persistence
- Misconception correction
- Concept retention
- Regression cases
- Concept-specific gain bar plot
- Inter-concept correlation matrix

Figures are saved in `data/processed/figures/` for direct inclusion in reports or publications.

---

## Repository Structure

```
alternative-conceptions-study/
│
├── data/
│   ├── raw/                      # Raw identifiable data (NOT included in public repo)
│   └── processed/                # Anonymized datasets and analysis outputs
│       ├── pretest_anonymized.csv
│       ├── posttest_anonymized.csv
│       ├── merged_anonymized.csv
│       ├── concept_gains.csv     # Concept-specific results
│       ├── correlation_matrix.csv # Inter-concept correlations
│       └── figures/              # Generated plots (PNG)
│           ├── ecdf_pre_vs_post.png
│           ├── box_pre_post_gain.png
│           ├── concept_gains_barplot.png
│           ├── correlation_matrix.png
│           ├── boxplot_normalized_gain_by_quartile.png
│           ├── scatter_gain_categories.png
│           ├── gain_distribution_comparison.png
│           └── transition_plots/
│
├── scripts/                      # Executable analysis utilities
│   ├── analyze.py                # Main analysis pipeline
│   ├── plots.py                  # Visualization generation
│   ├── stats.py                  # Statistical computations
│   └── loader.py                 # Data loading and column normalization
│
├── docs/
│   ├── instruments/              # Research instruments and tests
│   │   └── research-instruments.pdf
│   └── data/                     # Human-readable anonymized data
│       └── anonymized-marks.pdf
│
├── README.md
├── LICENSE
├── .gitignore
└── requirements.txt
```

---

## Key Findings

### Overall Learning Gains
- **Mean pre-test score:** 4.77/10 (SD = 1.61)
- **Mean post-test score:** 7.44/10 (SD = 1.86)
- **Mean raw gain (post − pre):** 2.67 marks (SD = 2.17)
- **Mean normalized gain (Hake’s \(\bar{g}\), mean of individual \(g_i\)):** 0.47
- **Effect sizes (paired):** Cohen’s d = 1.23, Hedges’ g = 1.22

### Student Outcomes
- **Improved:** 132 students (88.0%)
- **No change:** 4 students (2.7%)
- **Declined:** 14 students (9.3%)

### Concept-Level Change (selected mappings)
The instrument spans multiple concepts and uses mixed-mark post-test items; the values below summarize the **mean** performance change for each mapped concept block.

| Concept Area | Pre (mean) | Post (mean) | Mean change |
|---|---:|---:|---:|
| Seasons & tilt (Q1 → P3i) | 0.493 | 0.747 | +0.253 |
| Working of astronomers (Q4 → P3ii) | 0.487 | 0.667 | +0.180 |
| Asteroids (Q7 → P1) | 0.280 | 0.887 | +0.607 |
| Phases of Moon (Q2 → P5; out of 3) | 0.460 | 2.053 | +1.593 |
| Luminosity / stars vs planets (Q3 → P4; out of 2) | 0.387 | 1.747 | +1.360 |

### Most Persistent Misconceptions
| Misconception | Pre-test % | Post-test % | Reduction |
|---------------|-----------|------------|-----------|
| Seasons due to Earth–Sun distance | 72.0% | 41.3% | 30.7% |
| Moon phases caused by Earth's shadow | 68.0% | 38.0% | 30.0% |
| All bright night objects are stars | 61.0% | 32.0% | 29.0% |
| Asteroids are small planets | 58.0% | 26.0% | 32.0% |
| Planets glow like stars | 49.0% | 21.3% | 27.7% |

### Normalized Gain Distribution (Hake, 1998)
- **High gain (g > 0.7):** 50 students (33.3%)
- **Medium gain (0.3 ≤ g ≤ 0.7):** 57 students (38.0%)
- **Low gain (g < 0.3):** 43 students (28.7%)
- **Pre-test vs normalized gain:** r = -0.306, p = 0.0001

---

## Methodological Notes

- Weak pre-test reliability is expected for short diagnostic instruments and is treated as a research finding rather than a limitation.
- Concept-level transition analysis is used to compensate for low aggregate reliability.
- The analysis emphasizes conceptual change over score inflation.

---

## Intended Use

This repository is intended for:
- Educational research transparency  
- Reproducible analysis of misconception-focused instruction  
- Extension into concept-diagnostic assessment design  
- Academic review and sharing  

---

## Ethics

- All student data are anonymized  
- No personally identifiable information is included  
- Repository is suitable for academic dissemination

---

## License

This repository is released under the MIT License. See LICENSE for details.

---

## Citation and DOI

This repository is archived on Zenodo and assigned DOIs for citation and long-term preservation.

**Version DOI (v1.1 – manuscript revision used for submission):**  
https://doi.org/10.5281/zenodo.18791134  

This DOI refers to the exact frozen snapshot of the code, data, and analysis used in the revised manuscript and should be cited for reproducibility and peer review.

**Concept DOI (all versions):**  
https://doi.org/10.5281/zenodo.18060218  

The concept DOI always resolves to the latest version of this repository.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18791134.svg)](https://doi.org/10.5281/zenodo.18791134)
