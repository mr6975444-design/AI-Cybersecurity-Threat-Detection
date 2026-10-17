# AI-Powered Cybersecurity Threat Detection System

A machine-learning-based network intrusion detection prototype that classifies network connection records as **normal** or **potential threats** using the NSL-KDD benchmark dataset and a Random Forest classifier.

## Overview

This project demonstrates how machine learning can support network security analysis by identifying suspicious patterns in network connection records, generating threat alerts, and visualizing model performance.

**This is an offline, dataset-based prototype—not a production intrusion detection system.**

## Features

- NSL-KDD dataset loading and preprocessing
- Binary classification: normal traffic vs. potential threats
- Random Forest classification pipeline
- Categorical feature encoding and missing-value handling
- Evaluation using accuracy, precision, recall, and F1 score
- Threat prediction with estimated model probabilities
- CSV-based alert generation
- Confusion matrix and prediction visualizations
- Single-command execution of the complete pipeline

## Technology Stack

- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib and Seaborn
- Joblib

## Project Structure

```text
AI-Cybersecurity-Threat-Detection/
├── data/
├── images/
│   ├── class_distribution.png
│   ├── confusion_matrix.png
│   └── prediction_distribution.png
├── models/
├── outputs/
│   ├── alerts.csv
│   ├── metrics.txt
│   └── predictions.csv
├── src/
│   ├── config.py
│   ├── detect.py
│   ├── preprocess.py
│   ├── train.py
│   └── visualize.py
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```

Generated model files and CSV outputs are excluded from Git where configured. Run the pipeline to regenerate them.

## Dataset

The project uses the NSL-KDD network intrusion detection benchmark, with separate training and test files.

Place the following files in the `data/` directory before running the project:

- `KDDTrain+.txt`
- `KDDTest+.txt`

The dataset contains normal records and multiple categories of network attacks. The project maps these labels into two classes: `Normal` and `Threat`.

## Installation

Python 3.11 or newer is recommended. Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install the dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Ensure both NSL-KDD files are present in `data/`.

## Run the Project

From the project root, execute:

```bash
python3 main.py
```

The pipeline trains the classifier, evaluates it on the test dataset, generates predictions and alerts, and saves visualizations.

## Evaluation Results

Results from the current run on the 22,544-record NSL-KDD test set:

| Metric | Result |
|---|---:|
| Accuracy | 77.71% |
| Precision | 96.90% |
| Recall | 62.85% |
| F1 Score | 76.25% |

These are results from this specific experiment, not a guarantee of performance on real-world network traffic.

**Interpretation:** The model has high precision for the threat class but detects only 62.85% of actual threat records in this test set. Improving recall and evaluating false-positive and false-negative behavior are important next steps.

## Output Files

- `outputs/metrics.txt` — evaluation metrics and classification report
- `outputs/predictions.csv` — actual labels and model predictions
- `outputs/alerts.csv` — predicted class, estimated threat probability, and alert label
- `images/confusion_matrix.png` — comparison of actual and predicted classes
- `images/class_distribution.png` — test-set class distribution
- `images/prediction_distribution.png` — model prediction distribution

## Limitations

- Uses a historical benchmark dataset rather than live network traffic.
- Does not capture packets or monitor a real network.
- Benchmark results may not generalize to modern traffic or previously unseen attacks.
- A predicted threat is a model classification, not proof of malicious activity.
- The model is a learning prototype and has not been validated for operational security use.

## Future Improvements

- Compare additional classification algorithms.
- Investigate recall and false-negative reduction.
- Add cross-validation and more detailed error analysis.
- Explore anomaly detection as a separate experiment.
- Build a simple dashboard for reviewing generated alerts.
- Evaluate on additional datasets before making real-world performance claims.

## Responsible Use

This project is intended for education, defensive cybersecurity learning, and offline analysis of benchmark data. It is not a substitute for professional security monitoring.

## Author

Developed as a student portfolio project to explore machine learning, network intrusion detection, model evaluation, and cybersecurity data visualization.

