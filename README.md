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
