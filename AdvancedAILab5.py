import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR

# ============================================================
# Creating Different Types of Tensors
# ============================================================
# 1D Tensor (Vector)
tensor_1d = torch.tensor([1, 2, 3, 4, 5])
print("1D Tensor:", tensor_1d)
print("Shape:", tensor_1d.shape)
print("Data type:", tensor_1d.dtype)
# 2D Tensor (Matrix)
tensor_2d = torch.tensor([[1, 2, 3],
[4, 5, 6]])
print("\n2D Tensor:")
print(tensor_2d)
print("Shape:", tensor_2d.shape)

# ============================================================
# Special Tensor Initializations
# ============================================================
# Zeros matrix
zeros = torch.zeros(3, 3)
# Ones matrix
ones = torch.ones(2, 4)
# Random values (normal distribution)
random = torch.randn(2, 3)
# Identity matrix
identity = torch.eye(3)
print("\nZeros matrix:")
print(zeros)
print("\nRandom matrix:")
print(random)

# ============================================================
# Basic Arithmetic Operations
# ============================================================
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])
# Element-wise operations
addition = a + b  # [5.0, 7.0, 9.0]
multiplication = a * b  # [4.0, 10.0, 18.0]
division = b / a  # [4.0, 2.5, 2.0]
print("Addition:", addition)
print("Multiplication:", multiplication)
print("Division:", division)

# ============================================================
# Matrix Operations
# ============================================================
matrix_a = torch.randn(2, 3)  # 2x3 matrix
matrix_b = torch.randn(3, 4)  # 3x4 matrix
# Matrix multiplication: (2x3) @ (3x4) = (2x4)
result = torch.matmul(matrix_a, matrix_b)
# Alternative syntax: result = matrix_a @ matrix_b
print("\nResult shape:", result.shape) # torch.Size([2, 4])

# ============================================================
# Reshaping Tensors
# ============================================================
x = torch.arange(12) # [0, 1, 2, ..., 11]
x_reshaped = x.view(3, 4) # Reshape to 3x4
x_transposed = x_reshaped.t() # Transpose
print("\nOriginal shape:", x.shape)
print("Reshaped:", x_reshaped.shape)
print("Transposed:", x_transposed.shape)


import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# ============================================================
# Define Neural Network Class
# ============================================================
class SimpleNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initialize the neural network.
        Parameters:
            input_size: Number of input features
            hidden_size: Number of neurons in hidden layer
            output_size: Number of output classes
        """
        super(SimpleNeuralNetwork, self).__init__()
        # Define layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Forward pass through the network.
        Parameters:
            x: Input tensor (batch_size, input_size)
        Returns:
            Output tensor (batch_size, output_size)
        """
        # First layer + ReLU activation
        x = F.relu(self.fc1(x))
        # Output layer (no activation)
        x = self.fc2(x)
        return x
    
# ============================================================
# Create and Test the Model
# ============================================================
# Initialize model
model = SimpleNeuralNetwork(
input_size=10,
hidden_size=20,
output_size=3
)
print(model)
print()
# Test with random input
test_input = torch.randn(5, 10)
output = model(test_input)
print("Input shape:", test_input.shape)
print("Output shape:", output.shape)


class ImprovedNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout_rate=0.2):
        """
        Multi-layer network with dropout regularization.
        Parameters:
            input_size: Number of input features
            hidden_sizes: List of hidden layer sizes [64, 32, 16]
            output_size: Number of output classes
            dropout_rate: Dropout probability
        """
        super(ImprovedNetwork, self).__init__()
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# ============================================================
# Create Model with Custom Architecture
# ============================================================
model = ImprovedNetwork(
    input_size=20,
    hidden_sizes=[64, 32, 16],
    output_size=2,
    dropout_rate=0.3
)
print(model)
print()
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")


# ============================================================
# Generate Synthetic Dataset
# ============================================================
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    n_classes=2,
    random_state=42
)
# ============================================================
# Split Data
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
# ============================================================
# Standardize Features (IMPORTANT!)
# ============================================================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ============================================================
# Convert to PyTorch Tensors
# ============================================================
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.LongTensor(y_train)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.LongTensor(y_test)
print(f"Training samples: {len(X_train_tensor)}")
print(f"Test samples: {len(X_test_tensor)}")
print(f"Features: {X_train_tensor.shape[1]}")

# ============================================================
# Define Binary Classifier
# ============================================================
class BinaryClassifier(nn.Module):
    def __init__(self, input_size):
        super(BinaryClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)  # 2 classes
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# ============================================================
# Initialize Training Components
# ============================================================
model = BinaryClassifier(input_size=X_train_tensor.shape[1])
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
print("Model Architecture:")
print(model)
print()
print(f"Optimizer: {optimizer.__class__.__name__}")
print(f"Loss Function: {criterion.__class__.__name__}")


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            return
        if val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


# ============================================================
# Training Configuration
# ============================================================
num_epochs = 100
batch_size = 32
use_plateau_scheduler = False  # Toggle to True to use ReduceLROnPlateau
train_losses = []
test_accuracies = []
early_stopping = EarlyStopping(patience=15, min_delta=0.001)
step_scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
plateau_scheduler = ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=10,
)
print("Starting training...\n")

# ============================================================
# Main Training Loop
# ============================================================
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0

    for i in range(0, len(X_train_tensor), batch_size):
        batch_X = X_train_tensor[i:i + batch_size]
        batch_y = y_train_tensor[i:i + batch_size]
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    avg_loss = epoch_loss / (len(X_train_tensor) / batch_size)
    train_losses.append(avg_loss)

    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        val_loss = criterion(test_outputs, y_test_tensor).item()
        _, predicted = torch.max(test_outputs, 1)
        accuracy = (predicted == y_test_tensor).float().mean().item()
        test_accuracies.append(accuracy)

    if use_plateau_scheduler:
        plateau_scheduler.step(val_loss)
    else:
        step_scheduler.step()

    early_stopping(val_loss)

    if (epoch + 1) % 10 == 0 or early_stopping.early_stop:
        print(f"Epoch [{epoch + 1}/{num_epochs}]")
        print(f" Loss: {avg_loss:.4f}")
        print(f" Accuracy: {accuracy:.4f}")
        print()

    if early_stopping.early_stop:
        print("Early stopping triggered!")
        break

print("Training completed!\n")

# ============================================================
# Generate Predictions
# ============================================================
model.eval()
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    _, predictions = torch.max(test_outputs, 1)
y_pred = predictions.numpy()
y_true = y_test_tensor.numpy()

# ============================================================
# Calculate Metrics
# ============================================================
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average="binary")
recall = recall_score(y_true, y_pred, average="binary")
f1 = f1_score(y_true, y_pred, average="binary")
print("Model Performance:")
print(f" Accuracy: {accuracy:.4f}")
print(f" Precision: {precision:.4f}")
print(f" Recall: {recall:.4f}")
print(f" F1 Score: {f1:.4f}")

# ============================================================
# Classification Report
# ============================================================
print("\nDetailed Report:")
print(
    classification_report(
        y_true,
        y_pred,
        target_names=["Class 0", "Class 1"],
    )
)
# ============================================================
# Confusion Matrix
# ============================================================
cm = confusion_matrix(y_true, y_pred)
print("\nConfusion Matrix:")
print(cm)
print(f"\nTrue Negatives: {cm[0, 0]}")
print(f"False Positives: {cm[0, 1]}")
print(f"False Negatives: {cm[1, 0]}")
print(f"True Positives: {cm[1, 1]}")


def predict_new_samples(model, X_new, scaler):
    """
    Make predictions on new numpy samples using the trained model.
    """
    X_scaled = scaler.transform(X_new)
    X_tensor = torch.FloatTensor(X_scaled)
    model.eval()
    with torch.no_grad():
        outputs = model(X_tensor)
        probabilities = F.softmax(outputs, dim=1)
        _, predictions = torch.max(outputs, 1)
    return predictions.numpy(), probabilities.numpy()


# ============================================================
# Example: Predict on New Samples
# ============================================================
new_samples = np.random.randn(5, 20)
pred_classes, pred_probs = predict_new_samples(model, new_samples, scaler)
print("Predictions:")
print("-" * 50)
for i in range(len(new_samples)):
    print(f"Sample {i + 1}:")
    print(f" Class: {pred_classes[i]}")
    print(f" Class 0 Prob: {pred_probs[i][0]:.4f}")
    print(f" Class 1 Prob: {pred_probs[i][1]:.4f}")
    print()


# ============================================================
# Model Persistence Examples
# ============================================================
torch.save(model.state_dict(), "model_weights.pth")
reloaded_model = BinaryClassifier(input_size=20)
reloaded_model.load_state_dict(torch.load("model_weights.pth"))
reloaded_model.eval()

torch.save(model, "complete_model.pth")
complete_model = torch.load("complete_model.pth", weights_only=False)
complete_model.eval()

checkpoint = {
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "loss": loss.item(),
    "accuracy": accuracy,
}
torch.save(checkpoint, "checkpoint.pth")
loaded_checkpoint = torch.load("checkpoint.pth", weights_only=False)
checkpoint_model = BinaryClassifier(input_size=20)
checkpoint_model.load_state_dict(loaded_checkpoint["model_state_dict"])
checkpoint_optimizer = optim.Adam(checkpoint_model.parameters(), lr=0.001)
checkpoint_optimizer.load_state_dict(loaded_checkpoint["optimizer_state_dict"])
