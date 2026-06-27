# ASM-Net Hemorrhage Segmentation

This repository provides anonymous code and supplementary statistical material for:

**ASM-Net: Anisotropic Small-hemorrhage Modeling Network for Thick-Slice Intracranial Hemorrhage Segmentation**

The repository supports anonymous peer review and contains the model implementation and supplementary paired statistical analysis used in the manuscript.

## Repository Contents

| File | Description |
| --- | --- |
| `asm_net.py` | PyTorch implementation of ASM-Net. |
| `supplementary.pdf` | Anonymous supplementary statistical analysis, including paired Wilcoxon tests, bootstrap 95% confidence intervals, and Holm--Bonferroni-corrected p-values. |

## Model Overview

ASM-Net is designed for binary intracranial hemorrhage segmentation on thick-slice non-contrast CT. It combines axial-aware morphology modeling, hemorrhage context stabilization, and gated small-hemorrhage modulation to preserve compact lesion evidence while suppressing hyperdense distractors.

## Minimal Usage

```python
from asm_net import build_asm_net

model = build_asm_net(
    in_channels=1,
    out_channels=2,
    base_channels=32,
)
```

## Supplementary Statistical Analysis

The supplementary PDF reports paired case-level analysis on the primary cohort, including improvement magnitudes, bootstrap 95% confidence intervals, Holm--Bonferroni-corrected p-values, Dice improvements in percentage points, and HD95/ASD reductions in millimeters.

## Anonymization

This repository is anonymized for peer review. It contains no author or institution names, original case identifiers, DICOM metadata, private clinical data, or local filesystem paths.

## Notes

The code is provided for architectural review and reproducibility. Training data and private clinical annotations are not included because of privacy restrictions.
