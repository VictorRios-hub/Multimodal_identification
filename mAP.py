import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc, average_precision_score

# Define the number of classes and elements
num_classes = 43
num_elements = 6

# Define class sizes and true positive rates for each class
class_sizes = np.random.randint(10, 100, size=num_classes)
true_positive_rates = np.random.uniform(0.8, 0.9, size=num_classes)

# Generate example ground truth and predicted scores based on class sizes and true positive rates
ground_truth = []
predicted_scores = []

for class_idx in range(num_classes):
    true_positive = int(class_sizes[class_idx] * true_positive_rates[class_idx])
    false_positive = class_sizes[class_idx] - true_positive
    true_labels = [class_idx] * true_positive
    false_labels = np.random.choice(np.delete(np.arange(num_classes), class_idx), size=false_positive, replace=True)
    ground_truth.extend(true_labels)
    predicted_scores.extend(np.concatenate((np.random.rand(true_positive), np.random.rand(false_positive))))

# Ensure consistent sample sizes
num_samples = min(len(ground_truth), len(predicted_scores))
ground_truth = ground_truth[:num_samples]
predicted_scores = predicted_scores[:num_samples]

# Calculate mAP for each class
mAP_per_class = []
for class_idx in range(num_classes):
    precision_curve, recall_curve, _ = precision_recall_curve(ground_truth == class_idx, predicted_scores)
    mAP = auc(recall_curve, precision_curve)
    mAP_per_class.append(mAP)

# Print mAP for each class
for class_idx, mAP in enumerate(mAP_per_class):
    print(f"Class {class_idx}: mAP = {mAP:.4f}")

# Plot mAP values for each class
plt.figure(figsize=(10, 6))
plt.bar(range(num_classes), mAP_per_class, tick_label=np.arange(num_classes))
plt.xlabel('Class')
plt.ylabel('mAP')
plt.title('Mean Average Precision (mAP) for Each Class')
plt.show()
