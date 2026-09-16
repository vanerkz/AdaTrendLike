# AdaTrendLike

Official implementation of:

**AdaTrendLike: Adaptive Hierarchical Detrending and Trend Likelihood Scoring for Time Series Anomaly Detection**

**Authors:** Van Kwan Zhi Koh, Songnan Lin, Zhiping Lin, Bihan Wen

**Venue:** IEEE Transactions on Industrial Informatics

**Paper:** [IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/11684397)

---

## Overview

AdaTrendLike is an unsupervised framework for multivariate time-series anomaly detection based on **adaptive hierarchical detrending and trend likelihood scoring**.

The key idea is to explicitly model the trend component of a time series and evaluate whether the observed trend is likely under the learned normal trend distribution.

Unlike methods that primarily rely on point-wise reconstruction errors, AdaTrendLike models the underlying temporal trend and derives an anomaly score from the likelihood of the observed trend.

The resulting model learns the distribution of normal trends and uses the corresponding likelihood to identify anomalous temporal behavior.

---

# Repository Structure

```text
AdaTrendLike/
│
├── dataset/
│   ├── SMD/
│   ├── PSM/
│   ├── SWaT/
│   ├── MSL/
│   └── SMAP/
│
├── checkpoints/
├── results/
│
├── Start.sh
├── requirements.txt
└── README.md
```

The benchmark datasets should be placed under the `dataset/` directory.

---

# Requirements

We recommend creating a dedicated Python environment.

For example:

```bash
conda create -n adatrendlike python=3.9
conda activate adatrendlike
```

Then install the required packages:

```bash
pip install -r requirements.txt
```

---

# Dataset Preparation

The experiments use the following commonly used multivariate time-series anomaly detection benchmarks:

* Server Machine Dataset (SMD)
* Pooled Server Metrics (PSM)
* Secure Water Treatment (SWaT)
* Mars Science Laboratory (MSL)
* Soil Moisture Active Passive (SMAP)

The datasets are **not included in this repository**.

Users should download the datasets from their original sources and place them in the corresponding directories.

---

## 1. SMD

### Server Machine Dataset

SMD is a multivariate time-series anomaly detection benchmark based on server telemetry.

The original dataset is available through the OmniAnomaly repository:

https://github.com/NetManAIOps/OmniAnomaly

Download the SMD data and place it in:

```text
dataset/SMD/
```

The directory should contain the training and testing data required by the implementation.

A typical organization is:

```text
dataset/
└── SMD/
    ├── train/
    ├── test/
    └── test_label/
```

Do not rename files unless the corresponding dataset loader is also changed.

---

## 2. PSM

### Pooled Server Metrics

PSM is a multivariate server-monitoring dataset released by eBay and widely used for time-series anomaly detection.

Original repository:

https://github.com/eBay/RANSynCoders

Place the downloaded files in:

```text
dataset/PSM/
```

For example:

```text
dataset/
└── PSM/
    ├── train.csv
    ├── test.csv
    └── test_label.csv
```

---

## 3. SWaT

### Secure Water Treatment

SWaT is an industrial control-system dataset collected from a scaled water-treatment testbed.

Because SWaT has specific dataset access and distribution conditions, users should obtain the dataset from the original dataset provider: https://www.sutd.edu.sg/itrust/itrust-labs/datasets/dataset-characteristics/swat/.

After obtaining the dataset, place the required files in:

```text
dataset/
└── SWaT/
    ├── train.csv
    └── test.csv
```

The last column of each CSV file should contain the state label:

Normal or Attack

---

## 4. MSL

### Mars Science Laboratory

MSL contains spacecraft telemetry used for anomaly detection.

The data can be obtained through the Telemanom repository:

https://github.com/khundman/telemanom

Place the required MSL files in:

```text
dataset/
└── MSL/
    ├── train.npy
    ├── test.npy
    └── test_label.npy
```

---

## 5. SMAP

### Soil Moisture Active Passive

SMAP is a spacecraft telemetry dataset commonly used for multivariate time-series anomaly detection.

The dataset can be obtained from the official Telemanom repository:

https://github.com/khundman/telemanom

After downloading the dataset, organize the SMAP files in the following directory structure:

```text
dataset/
└── SMAP/
    ├── train/
    │   ├── A-1.npy
    │   ├── A-2.npy
    │   ├── ...
    │
    ├── test/
    │   ├── A-1.npy
    │   ├── A-2.npy
    │   ├── ...
    │
    └── labeled_anomalies.csv
```

* `train/` contains the SMAP training time-series files.
* `test/` contains the SMAP testing time-series files.
* `labeled_anomalies.csv` contains the anomaly labels for the test data.

---

# Final Dataset Structure

After preparing all datasets, the repository should look like:

```text
AdaTrendLike/
│
├── dataset/
│   ├── SMD/
│   │   └── ...
│   │
│   ├── PSM/
│   │   └── ...
│   │
│   ├── SWaT/
│   │   └── ...
│   │
│   ├── MSL/
│   │   └── ...
│   │
│   └── SMAP/
│       └── ...
│
├── checkpoints/
├── results/
├── Start.sh
├── requirements.txt
└── README.md
```

---

# Running the Code

After installing the dependencies and preparing the datasets, the experiments can be started using:

```bash
bash Start.sh
```

---

# Dataset Disclaimer

The datasets used by this repository belong to their respective original creators and providers.

This repository does **not** redistribute the benchmark datasets.

Users are responsible for:

1. Obtaining the datasets from the original sources.
2. Following the corresponding dataset licenses and terms of use.
3. Preparing the datasets according to the structure expected by this repository.

---

# Citation

Please cite the following paper when using this implementation:

```bibtex
@article{koh2026adatrendlike,
  author  = {Van Kwan Zhi Koh and Songnan Lin and Zhiping Lin and Bihan Wen},
  title   = {AdaTrendLike: Adaptive Hierarchical Detrending and Trend Likelihood Scoring for Time Series Anomaly Detection},
  journal = {IEEE Transactions on Industrial Informatics},
  year    = {2026}
}
```

Paper:

https://ieeexplore.ieee.org/abstract/document/11684397

---

# Acknowledgements

We thank the authors and maintainers of the benchmark datasets and publicly available resources used in this work.

In particular, we acknowledge the resources associated with:

* SMD / OmniAnomaly
* PSM / eBay
* SWaT
* MSL / Telemanom
* SMAP / Telemanom
* Anomaly Transformer — https://github.com/thuml/Anomaly-Transformer
