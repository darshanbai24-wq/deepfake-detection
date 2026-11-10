from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# Actual labels
y_true = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]

# Model predictions
y_pred = [1, 1, 1, 1, 0, 0, 0, 0, 0, 1]

# Calculate metrics
print("Accuracy:", accuracy_score(y_true, y_pred))
print("Precision:", precision_score(y_true, y_pred))
print("Recall:", recall_score(y_true, y_pred))
print("F1 Score:", f1_score(y_true, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))