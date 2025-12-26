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
- **Mean pre-test score:** 4.77/10 (SD = 1.89)
- **Mean post-test score:** 6.55/10 (SD = 1.76)
- **Mean raw gain:** 1.78 marks
- **Normalized gain (Hake's g):** 0.34 (moderate improvement)
- **Effect sizes:** Cohen's d = 0.84, Hedges' g = 0.84 (large effect)

### Concept-Specific Improvement
| Concept Area | Normalized Gain | Pre-test % Correct | Post-test % Correct |
|--------------|----------------|-------------------|-------------------|
| Asteroids | 0.58 | 28.0% | 68.7% |
| Seasonal Variation | 0.52 | 32.0% | 66.0% |
| Stars vs. Planets | 0.41 | 39.0% | 66.3% |
| Astronomers' Work | 0.38 | 42.0% | 67.3% |
| Planetary Characteristics | 0.29 | 36.5% | 56.8% |
| Moon Phases | 0.25 | 44.0% | 59.3% |

### Most Persistent Misconceptions
| Misconception | Pre-test % | Post-test % | Reduction |
|---------------|-----------|------------|-----------|
| Seasons due to Earth–Sun distance | 72.0% | 41.3% | 30.7% |
| Moon phases caused by Earth's shadow | 68.0% | 38.0% | 30.0% |
| All bright night objects are stars | 61.0% | 32.0% | 29.0% |
| Asteroids are small planets | 58.0% | 26.0% | 32.0% |
| Planets glow like stars | 49.0% | 21.3% | 27.7% |

### Gain Distribution
- **High gain (g > 0.7):** 33.3% of students
- **Medium gain (0.3 ≤ g ≤ 0.7):** 38.0% of students  
- **Low gain (g < 0.3):** 28.7% of students
- **Strong negative correlation:** Pre-test score vs. normalized gain (r = -0.550, p < 0.001)

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

**Version DOI (v1.1 – used for manuscript submission):**  
https://doi.org/10.5281/zenodo.18062669  

This DOI refers to the exact version of the code, data, and analysis used in the submitted manuscript and should be cited for reproducibility and peer review.

**Concept DOI (all versions):**  
https://doi.org/10.5281/zenodo.18060218  

The concept DOI always resolves to the latest version of this repository.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18062669.svg)](https://doi.org/10.5281/zenodo.18062669)
