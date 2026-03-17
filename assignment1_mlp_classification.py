import os
import urllib.request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Split dataset into train and test while keeping class distribution balanced
def stratified_train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int | None = None,
):
    
    rng = np.random.RandomState(random_state)
    train_idx = []
    test_idx = []

    # Process each class separately to maintain stratification
    for label in np.unique(y):
        label_idx = np.where(y == label)[0]
        rng.shuffle(label_idx)
        n_test = int(np.floor(len(label_idx) * test_size))
        test_idx.extend(label_idx[:n_test])
        train_idx.extend(label_idx[n_test:])

    rng.shuffle(train_idx)
    rng.shuffle(test_idx)

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


# Ensure vectors are binary and flattened
def _binary_vector(y: np.ndarray) -> np.ndarray:
    return np.asarray(y).reshape(-1).astype(int)


# Compute classification accuracy
def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = _binary_vector(y_true)
    y_pred = _binary_vector(y_pred)
    return float(np.mean(y_true == y_pred))


# Compute precision (positive prediction quality)
def precision_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = _binary_vector(y_true)
    y_pred = _binary_vector(y_pred)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return float(tp / (tp + fp)) if tp + fp > 0 else 0.0


# Compute recall (ability to detect positives)
def recall_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = _binary_vector(y_true)
    y_pred = _binary_vector(y_pred)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return float(tp / (tp + fn)) if tp + fn > 0 else 0.0


# Compute F1-score (balance between precision and recall)
def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    return float(2 * p * r / (p + r)) if p + r > 0 else 0.0


# Dataset source
DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
)
DATA_FILE = "wdbc.data"


# Download dataset only if it doesn't exist locally
def maybe_download_data(path: str, url: str):
   
    if os.path.exists(path):
        return
    print(f"Downloading {path}...")
    urllib.request.urlretrieve(url, path)


# Load dataset and convert labels to binary values
def read_wdbc(path: str):
    col_names = ["id", "diagnosis"] + [f"feature_{i+1}" for i in range(30)]
    df = pd.read_csv(path, header=None, names=col_names)

    X = df.iloc[:, 2:].astype(float).to_numpy()
    y = (df["diagnosis"] == "M").astype(int).to_numpy()
    return X, y


# Simple implementation of a 1-hidden-layer neural network (MLP)
class SimpleMLP:

    def __init__(self, input_dim: int, hidden_dim: int, lr: float):
        self.lr = lr

        # Initialize weights with small random values
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.01
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, 1) * 0.01
        self.b2 = np.zeros((1, 1))

    # Sigmoid activation function
    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-z))

    # Derivative of sigmoid used in backpropagation
    @staticmethod
    def _sigmoid_derivative(a: np.ndarray) -> np.ndarray:
        return a * (1 - a)

    # Forward propagation through the network
    def forward(self, X: np.ndarray) -> np.ndarray:
        self.z1 = X @ self.W1 + self.b1
        self.a1 = self._sigmoid(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.y_hat = self._sigmoid(self.z2)
        return self.y_hat

    # Mean Squared Error loss function
    @staticmethod
    def mse_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true = y_true.reshape(-1, 1)
        return np.mean((y_pred - y_true) ** 2)

    # Backpropagation to compute gradients and update weights
    # Although the division by m is not explicitly shown in the formula,
    # it is included here to average the gradients and ensure stable training.
    def backward(self, X: np.ndarray, y_true: np.ndarray):
        m = X.shape[0]
        y_true = y_true.reshape(-1, 1)

        dz2 = 2 * (self.y_hat - y_true) * self._sigmoid_derivative(self.y_hat)
        dW2 = self.a1.T @ dz2 / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m

        dz1 = (dz2 @ self.W2.T) * self._sigmoid_derivative(self.a1)
        dW1 = X.T @ dz1 / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m

        # Gradient descent parameter update
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1


# Training loop for the neural network
def train_loop(
    model: SimpleMLP,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int,
):
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for _ in range(epochs):

        # Forward pass on training data
        y_hat_train = model.forward(X_train)
        train_loss = model.mse_loss(y_train, y_hat_train)

        # Backpropagation step
        model.backward(X_train, y_train)

        # Forward pass on validation/test data
        y_hat_val = model.forward(X_val)
        val_loss = model.mse_loss(y_val, y_hat_val)

        train_pred = (y_hat_train >= 0.5).astype(int)
        val_pred = (y_hat_val >= 0.5).astype(int)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(accuracy_score(y_train, train_pred))
        val_accs.append(accuracy_score(y_val, val_pred))

    return train_losses, val_losses, train_accs, val_accs


# Evaluate final model performance on test data
def print_final_eval(model: SimpleMLP, X_test: np.ndarray, y_test: np.ndarray):
    y_hat = model.forward(X_test)
    y_pred = (y_hat >= 0.5).astype(int)

    print("Final evaluation on test set:")
    print("  Accuracy:", accuracy_score(y_test, y_pred))
    print("  Precision:", precision_score(y_test, y_pred))
    print("  Recall:", recall_score(y_test, y_pred))
    print("  F1-score:", f1_score(y_test, y_pred))


# Plot loss and accuracy during training
def plot_training_curves(
    train_loss, val_loss, train_acc, val_acc, show: bool = True
):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label="Train loss")
    plt.plot(val_loss, label="Test loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("Loss per epoch")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label="Train acc")
    plt.plot(val_acc, label="Test acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy per epoch")
    plt.legend()

    plt.tight_layout()
    if show:
        plt.show()


# Compare different learning rates: Plot train accuracy and train loss (MSE) for multiple LRs
def plot_learning_rates(
    X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, epochs: int
):
    lrs = [1.0, 0.5, 0.1, 0.01]

    all_train_losses = []
    all_val_losses = []
    all_train_accs = []
    all_val_accs = []

    for lr in lrs:
        np.random.seed(42)
        model = SimpleMLP(input_dim=X_train.shape[1], hidden_dim=7, lr=lr)
        train_loss, val_loss, train_acc, val_acc = train_loop(model, X_train, y_train, X_val, y_val, epochs)
        all_train_losses.append(train_loss)
        all_val_losses.append(val_loss)
        all_train_accs.append(train_acc)
        all_val_accs.append(val_acc)

    plt.figure(figsize=(12, 6))

    # 1. Train Accuracy vs Epoch (for each LR)
    plt.subplot(2, 2, 1)
    for lr, acc in zip(lrs, all_train_accs):
        plt.plot(acc, label=f"LR = {lr}", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Train Accuracy")
    plt.title("Train Accuracy vs Epoch")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 2. Test (Val) Accuracy vs Epoch (for each LR)
    plt.subplot(2, 2, 2)
    for lr, acc in zip(lrs, all_val_accs):
        plt.plot(acc, label=f"LR = {lr}", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy")
    plt.title("Test Accuracy vs Epoch")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 3. Train Loss vs Epoch (for each LR)
    plt.subplot(2, 2, 3)
    for lr, loss in zip(lrs, all_train_losses):
        plt.plot(loss, label=f"LR = {lr}", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Train MSE Loss")
    plt.title("Train Loss (MSE) vs Epoch")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 4. Test (Val) Loss vs Epoch (for each LR)
    plt.subplot(2, 2, 4)
    for lr, loss in zip(lrs, all_val_losses):
        plt.plot(loss, label=f"LR = {lr}", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Test MSE Loss")
    plt.title("Test Loss (MSE) vs Epoch")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# Evaluate impact of hidden layer size
def plot_hidden_layer_sizes(
    X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, epochs: int
):
    sizes = [5, 10, 15, 20, 25, 30]
    final_acc = []

    for hidden in sizes:
        np.random.seed(42)

        # Train model with different hidden units
        model = SimpleMLP(input_dim=X_train.shape[1], hidden_dim=hidden, lr=.9)

        _, _, train_acc, _ = train_loop(model, X_train, y_train, X_val, y_val, epochs)
        final_acc.append(train_acc[-1])

    plt.figure(figsize=(8, 5))
    plt.bar(sizes, final_acc)
    plt.title("Train accuracy vs hidden layer size")
    plt.xlabel("Hidden units")
    plt.ylabel("Accuracy")
    plt.ylim(0.8, 1.0)

    for i, acc in enumerate(final_acc):
        plt.text(sizes[i], acc + 0.002, f"{acc:.3f}", ha="center")

    plt.show()


# Main execution pipeline
def main():

    # Ensure dataset exists
    maybe_download_data(DATA_FILE, DATA_URL)

    # Load data
    X, y = read_wdbc(DATA_FILE)

    # Train-test split
    X_train, X_test, y_train, y_test = stratified_train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Feature normalization
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0)
    X_train = (X_train - mu) / sigma
    X_test = (X_test - mu) / sigma

    np.random.seed(42)

    # Create MLP model
    model = SimpleMLP(input_dim=X_train.shape[1], hidden_dim=15, lr=0.5)

    epochs = 100

    # Train the model
    train_loss, test_loss, train_acc, test_acc = train_loop(
        model, X_train, y_train, X_test, y_test, epochs
    )

    # Print final results
    print_final_eval(model, X_test, y_test)

    # Plot training performance
    plot_training_curves(train_loss, test_loss, train_acc, test_acc)

    # Experiment with learning rates
    plot_learning_rates(X_train, y_train, X_test, y_test, epochs)

    # Experiment with hidden layer sizes
    plot_hidden_layer_sizes(X_train, y_train, X_test, y_test, epochs)


if __name__ == "__main__":
    main()
    