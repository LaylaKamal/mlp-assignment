
## How to Run
pip install -r requirements.txt
```bash
python3 assignment1_mlp_classification.py


## Project Overview
This project implements a Multi-Layer Perceptron (MLP) neural network from scratch using Python and NumPy to classify breast cancer tumors as malignant or benign.

## Dataset
The dataset comes from the UCI Machine Learning Repository and contains 30 numerical features extracted from breast cell images.

### Labels
- M → Malignant
- B → Benign

Converted to:
- 1 → Malignant
- 0 → Benign

## Project Pipeline
1. Download dataset if not available
2. Load data using Pandas
3. Extract features and labels
4. Convert labels to binary
5. Split dataset into training and testing sets
6. Normalize features
7. Train the MLP model
8. Evaluate the model

## Model Architecture

### Input Layer
30 neurons (one for each feature)

### Hidden Layer
Configurable number of neurons

### Output Layer
1 neuron with sigmoid activation

## Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-score

## Experiments

### Learning Rate Experiment
Learning rates tested:
- 1.0
- 0.5
- 0.1
- 0.01

### Hidden Layer Size Experiment
Hidden neurons tested:
- 5
- 10
- 15
- 20
- 25
- 30

## Program Output
The program generates:
- Loss curves
- Accuracy curves
- Learning rate comparison
- Hidden layer size vs accuracy


