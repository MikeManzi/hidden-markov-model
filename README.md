# Human Activity Recognition Using Hidden Markov Models

A machine learning project that uses Hidden Markov Models (HMM) to classify human activities from smartphone IMU sensor data (accelerometer + gyroscope).

## Overview

This project implements a complete HMM-based Human Activity Recognition (HAR) pipeline capable of classifying four activities:

| Label | Activity |
| ----- | -------- |
| 0     | Still    |
| 1     | Standing |
| 2     | Walking  |
| 3     | Jumping  |

## Dataset

- **Structure**: 4 activities × 15 recordings × 2 sensors = 120 CSV files
- **Sensors**: Accelerometer (~100 Hz) and Gyroscope (~50 Hz)
- **Location**: `data/` directory
- **Naming convention**: `{activity}_{sensor}_{id}.csv` (e.g., `walking_acc_1.csv`)

### Train/Test Split

| Split | File IDs | Samples per Class | Total Samples |
| ----- | -------- | ----------------- | ------------- |
| Train | 1–12     | 12                | 48            |
| Test  | 13–15    | 3                 | 12            |

## Features

The pipeline extracts **60 features** from each recording:

| Category          | Features                                                      | Count |
| ----------------- | ------------------------------------------------------------- | ----- |
| Time-domain       | Mean, variance, std, SMA per axis                             | 24    |
| Axis correlations | Pairwise Pearson r (accel + gyro)                             | 6     |
| Frequency-domain  | Dominant freq, spectral energy, top-3 FFT magnitudes per axis | 30    |

**Signal Magnitude Area (SMA)**: $\text{SMA} = \frac{1}{N}\sum_{i=1}^{N}|x_i|$

## Pipeline Architecture

1. **Data Loading**: Load and merge accelerometer/gyroscope CSV files using `merge_asof`
2. **Preprocessing**: Normalize sensor data with StandardScaler
3. **Feature Extraction**: Extract 60 time/frequency domain features
4. **Model Training**: Train GaussianHMM with 4 hidden states
5. **Activity Decoding**: Viterbi decoding + majority-vote state-to-label mapping
6. **Evaluation**: Confusion matrix, classification report, sensitivity/specificity

## Requirements

```
numpy
pandas
matplotlib
seaborn
scipy
scikit-learn
hmmlearn
```

## Usage

1. Clone the repository
2. Ensure data files are in the `data/` directory
3. Open and run `HMM_Activity_Recognition.ipynb`

```python
# The notebook will:
# - Install hmmlearn if not present
# - Load and preprocess the sensor data
# - Train a Gaussian HMM model
# - Evaluate and visualize results
```

## Outputs

The notebook generates the following visualizations:

- `transition_matrix.png` - HMM state transition probabilities
- `decoded_sequence.png` - True vs predicted activity sequence
- `feature_distributions.png` - Feature boxplots by activity
- `confusion_matrix.png` - Test set confusion matrix
- `sensitivity_specificity.png` - Per-class metrics bar chart

## Model Details

- **Algorithm**: Gaussian Hidden Markov Model (`GaussianHMM`)
- **Hidden States**: 4 (one per activity)
- **Covariance Type**: Diagonal
- **Training Iterations**: 100
- **Decoding**: Viterbi algorithm with majority-vote mapping

## Project Structure

```
hidden-markov-model/
├── HMM_Activity_Recognition.ipynb  # Main notebook
├── README.md                        # This file
└── data/
    ├── jumping_acc_*.csv           # Jumping accelerometer data
    ├── jumping_gyro_*.csv          # Jumping gyroscope data
    ├── standing_acc_*.csv          # Standing accelerometer data
    ├── standing_gyro_*.csv         # Standing gyroscope data
    ├── still_acc_*.csv             # Still accelerometer data
    ├── still_gyro_*.csv            # Still gyroscope data
    ├── walking_acc_*.csv           # Walking accelerometer data
    └── walking_gyro_*.csv          # Walking gyroscope data
```

## License

This project is for educational purposes.
