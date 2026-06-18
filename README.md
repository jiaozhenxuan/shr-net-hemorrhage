# SHR-Net Hemorrhage Segmentation

This repository provides anonymous code and supplementary statistical material for:

**SHR-Net: Small Hemorrhage Refinement Network for Thick-Slice NCCT Intracranial Hemorrhage Segmentation**

The repository is intended to support anonymous peer review. It contains the model implementation and supplementary paired statistical analysis used to support the manuscript.

## Repository Contents

| File | Description |
| --- | --- |
| `shr_net.py` | PyTorch implementation of SHR-Net. |
| `supplementary.pdf` | Anonymous supplementary statistical analysis, including paired Wilcoxon testing, bootstrap 95% confidence intervals, and Holm-Bonferroni corrected p-values. |

## Model Overview

SHR-Net is designed for binary intracranial hemorrhage segmentation on thick-slice non-contrast CT. The model focuses on small-hemorrhage recovery and spatial robustness through anisotropy-aware convolution, multi-scale hemorrhage context aggregation, and decoder-side refinement.

## Supplementary Statistical Analysis

The supplementary PDF reports paired case-level statistical analysis on the primary cohort. The table reports:

- improvement magnitude over compared methods;
- bootstrap 95% confidence intervals;
- Holm-Bonferroni corrected p-values;
- Dice and Small Dice improvements as percentage points;
- HD95 and ASD reductions in millimeters.

Absolute case identifiers and institution-specific information are not included.

## Anonymization

This repository is anonymized for peer review. It does not include author names, hospital names, original case identifiers, DICOM metadata, or local filesystem paths.

## Notes

The code is provided for review and reproducibility of the model architecture. Training data and private clinical annotations are not included because of privacy restrictions.
