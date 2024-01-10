import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve, auc


# Generate example ground truth and predictions (replace with your own data)
# np.random.seed(0)
# ground_truth = np.random.randint(0, 9, size=90)
# predictions = np.random.randint(0, 9, size=90)

# Calculate confusion matrix
# confusion = confusion_matrix(ground_truth, predictions)

# Create a hypothetical confusion matrix (replace with your own data)
confusion = np.array([
    [20, 5, 2, 0, 0, 0, 1, 0, 2],
    [3, 15, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 18, 0, 0, 2, 0, 0, 0],
    [0, 2, 0, 20, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 15, 0, 0, 0, 2],
    [0, 1, 3, 0, 0, 10, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 18, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 22, 1],
    [1, 1, 0, 0, 2, 0, 0, 0, 10]
])

# Calculate precision and recall for each class
precision = np.diag(confusion) / confusion.sum(axis=0)
recall = np.diag(confusion) / confusion.sum(axis=1)

# Create hypothetical ground truth and predictions based on confusion matrix
ground_truth = []
predictions = []

for class_idx in range(9):
    for _ in range(confusion[class_idx, class_idx]):
        ground_truth.append(class_idx)
        predictions.append(class_idx)
    for other_class_idx in range(43):
        if other_class_idx != class_idx:
            for _ in range(confusion[other_class_idx, class_idx]):
                ground_truth.append(other_class_idx)
                predictions.append(class_idx)

ground_truth = np.array(ground_truth)
predictions = np.array(predictions)

# Calculate precision-recall curve for each class
precision_recall_curves = []
for class_idx in range(9):
    precision_curve, recall_curve, _ = precision_recall_curve((ground_truth == class_idx), predictions == class_idx)
    precision_recall_curves.append((precision_curve, recall_curve))

# Calculate mAP for each class
mAP_per_class = [auc(recall_curve, precision_curve) for precision_curve, recall_curve in precision_recall_curves]

# Plot mAP values for each class
plt.figure(figsize=(10, 6))
plt.bar(range(9), mAP_per_class, tick_label=np.arange(9))
plt.xlabel('Class')
plt.ylabel('mAP')
plt.title('Mean Average Precision (mAP) for Each Class')
plt.show()

# Plot confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(confusion, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix : multimodale audio + video (fusion caractéristiques)")
plt.colorbar()

tick_marks = np.arange(9)
plt.xticks(tick_marks, np.arange(0, 9))
plt.yticks(tick_marks, np.arange(0, 9))

for i in range(9):
    for j in range(9):
        plt.text(j, i, str(confusion[i, j]), horizontalalignment="center", color="white" if confusion[i, j] > confusion.max() / 2 else "black")

plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()
