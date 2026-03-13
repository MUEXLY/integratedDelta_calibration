# integratedDelta_calibration

Scripts for calibrating simulation data to experimental observations using an embedded discrepancy Gaussian Process (GP) emulator.

---

## Overview

This repository implements a Bayesian calibration framework that:

- Builds a GP emulator of simulator outputs  
- Embeds a model discrepancy term  
- Calibrates simulation parameters to observational data using MCMC  

The workflow is driven by `main.py` and configured via `config.json`.

---

## Environment Setup

This project uses a Python virtual environment.

### 1. Create a virtual environment

    python3 -m venv venv

### 2. Activate the environment

**macOS/Linux**
    
    source venv/bin/activate

**Windows**
    
    venv\Scripts\activate

### 3. Install required packages

    pip install -r requirements.txt

Core dependencies include:

- numpy  
- scipy  
- matplotlib  
- pandas  
- scikit-learn  

---

## Running the Calibration

To execute the integrated delta calibration routine:

    python3 main.py

All runtime behavior is controlled through `config.json`.

---

## Repository Structure

    integratedDelta_calibration/
    │
    ├── main.py
    ├── config.json
    ├── requirements.txt
    │
    ├── modelData/
    │   ├── appDomain.txt
    │   ├── modelPredictions.txt
    │   └── thetaVals.txt
    │
    └── observationData/
        ├── appDomain.txt
        └── observationData.txt

---

## Input Data Format

Input data is organized into two subdirectories:

### `/modelData`
Simulator outputs to be emulated and calibrated.

- `appDomain.txt` — Application domain locations  
- `modelPredictions.txt` — Simulator outputs  
- `thetaVals.txt` — Parameter values corresponding to simulations  

### `/observationData`
Experimental data to be matched.

- `appDomain.txt` — Observation domain locations  
- `observationData.txt` — Measured responses  

### Formatting Requirements

All `.txt` files must:

- Be column-formatted  
- Include a header row labeling each column  
- Use the delimiter specified in `config.json`  

The provided template files illustrate formatting for MD (observational) and DDD (simulator) predictions of  
$\tau^{CRSS}$.

---

## Configuration

All calibration settings are controlled via `config.json`.

### `calibration_settings`
Controls MCMC behavior:

- `N_mcmc` — Total MCMC iterations  
- `N_burn` — Burn-in samples  
- `N_samp_post` — Posterior samples retained  

### `input_settings`

- Input file delimiter  

### `output_settings`

- Controls which plots and diagnostics are generated  

### `results_path` & `results_options`

*(In development)*  
Controls output of serialized model objects for downstream analysis.

---

## Notes

- The virtual environment directory (`venv/`) should not be committed to the repository.
- Only `requirements.txt` is required to reproduce the environment.

## Citation

If you use this repository or the associated methodology in your research, please cite the following paper.

**Bayesian Model Calibration with Integrated Discrepancy: Addressing Inexact Dislocation Dynamics Models**
Liam Myhill, Enrique Martinez Saez, Sez Russcher (2026)

[![arXiv](https://img.shields.io/badge/arXiv-2603.11960-b31b1b.svg)](https://arxiv.org/abs/2603.11960)
[![DOI](https://img.shields.io/badge/DOI-coming%20soon-blue)]()

Paper: https://arxiv.org/abs/2603.11960

```bibtex
@misc{myhill2026bayesianmodelcalibrationintegrated,
  title   = {Bayesian Model Calibration with Integrated Discrepancy: Addressing Inexact Dislocation Dynamics Models},
  author  = {Myhill, Liam and Martinez Saez, Enrique and Russcher, Sez},
  year    = {2026},
  eprint  = {2603.11960},
  archivePrefix = {arXiv},
  primaryClass  = {stat.ME},
  url     = {https://arxiv.org/abs/2603.11960}
}
```

---

### Repository Citation

This repository includes a `CITATION.cff` file so that GitHub can generate a **"Cite this repository"** button automatically.

