# Anomaly Detection in Orbital Trajectories using Unsupervised Machine Learning

## Overview

This project implements an unsupervised machine learning framework for detecting orbital anomalies in satellite Two-Line Element (TLE) data. The system identifies maneuvers, reboosts, and other orbital modifications without requiring labeled training data.

**Key Features:**
- Physics-based feature engineering from raw TLE parameters
- Ground truth generation using orbital mechanics rules
- Comparison of 5 anomaly detection algorithms
- Temporal validation with train/validation/test split
- Comprehensive visualization dashboard

---

## Table of Contents

- [Background](#background)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Results](#results)
- [Usage](#usage)
- [Future Work](#future-work)
- [References](#references)
- [Authors](#authors)

---

## Background

Orbital debris and the increasing complexity of space operations necessitate robust, autonomous methods for monitoring Resident Space Objects. Traditional approaches rely on manual analysis of orbital parameters, which becomes impractical as the number of tracked objects grows.

This project leverages unsupervised machine learning to automatically detect anomalies in orbital trajectories, focusing on the International Space Station (ISS) as a benchmark due to its well-documented operational history and frequent maneuvers.

### Problem Statement

Given a time series of TLE observations for a satellite:
1. **Feature Engineering**: Transform raw orbital parameters into meaningful features
2. **Anomaly Detection**: Identify observations that deviate significantly from normal orbital behavior
3. **Validation**: Evaluate detection performance against physics-based ground truth

---

## Dataset

### Source
TLE data obtained from [Space-Track.org](https://www.space-track.org/), the official source for U.S. Space Command orbital data.

### Satellites Analyzed
| Satellite | NORAD ID | Type | Orbit |
|-----------|----------|------|-------|
| ISS (ZARYA) | 25544 | Payload | LEO (408-420 km) |
| ASTRA 2F | 38778 | Payload | GEO |
| ARIANE 5 DEB | 44336 | Debris | MEO |
| SL-8 DEB | 4084 | Debris | LEO |

### Time Period
- **Training**: January - October 2023
- **Validation**: November - December 2023
- **Test**: January - December 2024

### Raw TLE Parameters
```
MEAN_MOTION, ECCENTRICITY, INCLINATION, RA_OF_ASC_NODE,
ARG_OF_PERICENTER, MEAN_ANOMALY, BSTAR, SEMIMAJOR_AXIS,
APOGEE, PERIGEE, PERIOD
```

---

## Methodology

### 1. Feature Engineering

We transform raw TLE parameters into 70+ engineered features:

| Category | Features | Description |
|----------|----------|-------------|
| **Temporal** | Rates, accelerations, moving averages | First and second derivatives of orbital parameters |
| **Energy** | Orbital energy, energy rate | Specific orbital energy and its variations |
| **Angular** | Sin/cos transformations, angular rates | Circular statistics for angular parameters |
| **Shape** | Focal distance, perimeter, circularity | Orbital geometry indicators |
| **Drag** | BSTAR rate, altitude decay | Atmospheric drag indicators |
| **Statistical** | Rolling std, min, max, range | 20-observation window statistics |

### 2. Ground Truth Generation

Physics-based rules identify anomalies based on orbital mechanics principles:

| Rule | Threshold | Physical Meaning |
|------|-----------|------------------|
| Altitude Jump | > 0.15 km/h | Significant semi-major axis change |
| Mean Motion Change | > 0.00005 rev/day/h | Orbital period modification |
| Energy Change | > 0.005 km²/s²/h | Propulsive maneuver indicator |
| High Drag | > Mean + 3σ | Unusual atmospheric drag |
| Apogee/Perigee Jump | > 1.5 km | Orbital shape modification |
| Reboost | Both apogee & perigee increase | Altitude-raising maneuver |

### 3. Anomaly Detection Algorithms

Five unsupervised algorithms were evaluated:

| Algorithm | Type | Key Parameters |
|-----------|------|----------------|
| **Isolation Forest** | Tree-based | n_estimators, contamination |
| **One-Class SVM** | Boundary-based | nu, gamma, kernel |
| **Elliptic Envelope** | Statistical | contamination, support_fraction |
| **Local Outlier Factor** | Density-based | n_neighbors, contamination |
| **DBSCAN** | Clustering-based | eps, min_samples |

### 4. Evaluation Pipeline

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Training Set   │────▶│  Fit Scaler &   │────▶│  Train Models   │
│  (Jan-Oct 2023) │     │  PCA on Train   │     │  on Train Only  │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                        │
┌─────────────────┐     ┌─────────────────┐            ▼
│ Validation Set  │────▶│   Transform &   │────▶ Hyperparameter
│ (Nov-Dec 2023)  │     │    Predict      │      Tuning (F1)
└─────────────────┘     └─────────────────┘            │
                                                        ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│    Test Set     │────▶│   Transform &   │────▶│ Final Evaluation│
│     (2024)      │     │    Predict      │     │   & Reporting   │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

---

## Installation

### Requirements

```bash
Python >= 3.9
numpy >= 1.21.0
pandas >= 1.3.0
scikit-learn >= 1.0.0
matplotlib >= 3.4.0
seaborn >= 0.11.0
scipy >= 1.7.0
jupyter >= 1.0.0
```

## Project Structure

```
├── DATASET/
│   ├── ISS(ZARYA)_25544_data.csv
│   ├── ASTRA 2F_38778_data.csv
│   ├── ARIANE 5 DEB (SYLDA)_44336_data.csv
│   └── SL-8 DEB_4084_data.csv
│
├── assets/
│   └── dashboard_preview.png
│
├── Analyse ISS.ipynb          # Main analysis notebook
├── README.md
```

---

## Results

### Model Comparison

| Model | F1-Score | Precision | Recall | Silhouette |
|-------|----------|-----------|--------|------------|
| **Elliptic Envelope** | **0.5647** | 0.6857 | 0.4800 | 0.7587 |
| DBSCAN | 0.5567 | 0.5745 | 0.5400 | 0.7053 |
| Local Outlier Factor | 0.5333 | 0.6000 | 0.4800 | 0.6974 |
| Isolation Forest | 0.4706 | 0.5714 | 0.4000 | 0.7611 |
| One-Class SVM | 0.2857 | 0.2034 | 0.4800 | 0.1671 |

### Ground Truth Distribution

| Split | Period | Samples | Anomaly Rate |
|-------|--------|---------|--------------|
| Train | Jan-Oct 2023 | 1,621 | 7.6% |
| Validation | Nov-Dec 2023 | 343 | 14.6% |
| Test | 2024 | 2,119 | 15.7% |

### Key Findings

1. **Elliptic Envelope** achieved the best F1-score (0.5647), suggesting that the Gaussian assumption holds reasonably well for normal orbital behavior.

2. **Mean Motion Change** is the most triggered detection rule, indicating that orbital period variations are the primary anomaly signature.

3. The increasing anomaly rate from train (7.6%) to test (15.7%) reflects higher ISS maneuver frequency in 2024.

4. **Temporal validation** is crucial: models trained on 2023 data generalize well to 2024, demonstrating robustness to operational changes.

---

## Usage

### Running the Analysis

1. Open `Analyse ISS.ipynb` in Jupyter Notebook
2. Execute cells sequentially from top to bottom
3. Results and visualizations will be generated inline

### Customizing Parameters

To adjust ground truth sensitivity, modify the thresholds in the `create_ground_truth_labels` function:

```python
thresholds = {
    'semimajor_rate': 0.15,      # km/h - decrease for more sensitivity
    'mean_motion_rate': 0.00005, # rev/day/h
    'energy_rate': 0.005,        # km²/s²/h
    'bstar_sigma': 3.0,          # standard deviations
    'apogee_jump': 1.5,          # km
    'perigee_jump': 1.5,         # km
}
```

### Applying to Other Satellites

The framework can be applied to any satellite with TLE data:

1. Download TLE data from [Space-Track.org](https://www.space-track.org/)
2. Place CSV file in `DATASET/` folder
3. Update file path in the notebook
4. Adjust thresholds based on orbital regime (LEO/MEO/GEO)

---

## Authors

**Thibaut LEMARIE**  
**Emilie LIMERY**  
**Alexandre LANEN**  

