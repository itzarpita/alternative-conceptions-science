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

- Pre-test, post-test, and gain histograms  
- ECDF comparison (pre vs post)  
- Boxplots (pre / post / gain)  
- Paired scatter plot (pre vs post)  
- Gain vs pre-score scatter (who improved most)  
- Concept-level transition heatmaps showing:
  - Misconception persistence  
  - Misconception correction  
  - Concept retention  
  - Regression cases  

Figures are saved for direct inclusion in reports or publications.

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
│       └── figures/              # Generated plots (PNG)
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

This repository is archived on Zenodo and assigned a DOI for citation and long-term preservation.

**DOI:** https://doi.org/10.5281/zenodo.18060218

This DOI corresponds to the version used for **manuscript submission** and supports reproducibility and peer review.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18060218.svg)](https://doi.org/10.5281/zenodo.18060218)
