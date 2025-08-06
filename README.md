# EPPS2025 Workshop

This repository contains materials for the EPPS2025 workshop on hyperspectral imaging analysis.

## Contents

- `EPPS2025_workhsop_draft.ipynb` - Main workshop notebook with hyperspectral analysis examples
- `Lamewarden_tools/` - Submodule containing hyperspectral analysis tools

## Setup

This repository uses the `Lamewarden_tools` submodule. To clone with all submodules:

```bash
git clone --recursive https://github.com/lamewarden/EPPS2025_workshop.git
```

Or if you've already cloned the repository:

```bash
git submodule update --init --recursive
```

## Dependencies

The workshop requires Python with the following packages:
- numpy
- spectral
- pandas
- matplotlib
- scikit-learn
- torch
- optuna
- imbalanced-learn
- scipy

## Usage

Open the Jupyter notebook `EPPS2025_workhsop_draft.ipynb` to follow along with the workshop materials.
