
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(1)  

# 1. LOAD AND PREPARE THE DATA


data = pd.read_csv("student_data.csv")

# Features (X) and label (y)
X = data[["Study_Hours", "Attendance", "Previous_Marks", "Assignment_Scores"]].values
y = data["Result"].values.reshape(-1, 1)   




X_min = X.min(axis=0)
X_max = X.max(axis=0)
X = (X - X_min) / (X_max - X_min)


np.random.seed(1)
indices = np.random.permutation(len(X))
split = int(0.8 * len(X))

train_idx, test_idx = indices[:split], indices[split:]
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

print(f"Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")



# 2. ACTIVATION FUNCTIONS

def relu(z):
    """ReLU activation: returns 0 for negative values, z otherwise."""
    return np.maximum(0, z)


def relu_derivative(z):
    """Derivative of ReLU: 1 where z > 0, else 0."""
    return (z > 0).astype(float)


def sigmoid(z):
    """Sigmoid activation: squashes values between 0 and 1."""
    return 1 / (1 + np.exp(-z))



# 3. INITIALIZE WEIGHTS AND BIASES


n_input = X_train.shape[1]   
n_hidden = 4                 
n_output = 1                 


W1 = np.random.randn(n_input, n_hidden) * 0.1
b1 = np.zeros((1, n_hidden))

W2 = np.random.randn(n_hidden, n_output) * 0.1
b2 = np.zeros((1, n_output))



# 4. FORWARD PROPAGATION


def forward_propagation(X):
    """
    Passes the input through the network and returns everything needed
    for both prediction and backpropagation.
    """
    Z1 = np.dot(X, W1) + b1     
    A1 = relu(Z1)             

    Z2 = np.dot(A1, W2) + b2    
    A2 = sigmoid(Z2)         

    cache = (Z1, A1, Z2, A2)
    return A2, cache



# 5. LOSS FUNCTION (Binary Cross-Entropy)


def compute_loss(y_true, y_pred):
    """
    Binary cross-entropy loss. Measures how far the predicted
    probabilities are from the actual Pass/Fail labels.
    """
    m = y_true.shape[0]
    epsilon = 1e-8  # tiny value to avoid log(0)
    loss = -np.sum(y_true * np.log(y_pred + epsilon) +
                   (1 - y_true) * np.log(1 - y_pred + epsilon)) / m
    return loss



# 6. BACKPROPAGATION


def backward_propagation(X, y, cache):
    """
    Calculates how much each weight and bias contributed to the error,
    using the chain rule, so we know which direction to adjust them.
    """
    m = X.shape[0]
    Z1, A1, Z2, A2 = cache

    
    dZ2 = A2 - y                      
    dW2 = np.dot(A1.T, dZ2) / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m

    
    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * relu_derivative(Z1)     
    dW1 = np.dot(X.T, dZ1) / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m

    gradients = (dW1, db1, dW2, db2)
    return gradients



# 7. GRADIENT DESCENT (TRAINING LOOP)


learning_rate = 0.5
epochs = 2000
loss_history = []

for epoch in range(epochs):
    # Forward pass
    A2, cache = forward_propagation(X_train)

    # Compute loss
    loss = compute_loss(y_train, A2)
    loss_history.append(loss)

    # Backward pass
    dW1, db1, dW2, db2 = backward_propagation(X_train, y_train, cache)

    # Update weights and biases (gradient descent step)
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    if epoch % 200 == 0:
        print(f"Epoch {epoch:4d} | Loss: {loss:.4f}")

print(f"Final Training Loss: {loss_history[-1]:.4f}")



# 8. VISUALIZE TRAINING LOSS

plt.figure(figsize=(7, 5))
plt.plot(loss_history)
plt.title("Training Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss (Binary Cross-Entropy)")
plt.grid(True)
plt.savefig("training_loss.png")
plt.show()
print("Loss curve saved as training_loss.png")


# 9. MODEL EVALUATION


def predict(X):
    """Returns 1 (Pass) if probability >= 0.5, else 0 (Fail)."""
    A2, _ = forward_propagation(X)
    return (A2 >= 0.5).astype(int)


train_predictions = predict(X_train)
test_predictions = predict(X_test)

train_accuracy = np.mean(train_predictions == y_train) * 100
test_accuracy = np.mean(test_predictions == y_test) * 100

print(f"\nTraining Accuracy: {train_accuracy:.2f}%")
print(f"Testing Accuracy:  {test_accuracy:.2f}%")



# 10. PREDICT A NEW STUDENT


def predict_new_student(study_hours, attendance, previous_marks, assignment_scores):
    """
    Takes raw (un-normalized) values for a new student and predicts
    whether they will Pass or Fail.
    """
    new_data = np.array([[study_hours, attendance, previous_marks, assignment_scores]])
    new_data_scaled = (new_data - X_min) / (X_max - X_min)

    probability, _ = forward_propagation(new_data_scaled)
    result = "Pass" if probability[0][0] >= 0.5 else "Fail"

    print(f"\nNew Student -> Study Hours: {study_hours}, Attendance: {attendance}, "
          f"Previous Marks: {previous_marks}, Assignment Scores: {assignment_scores}")
    print(f"Predicted Probability of Passing: {probability[0][0]:.4f}")
    print(f"Prediction: {result}")


# Example prediction
predict_new_student(study_hours=0, attendance=100 , previous_marks=100, assignment_scores=100)
