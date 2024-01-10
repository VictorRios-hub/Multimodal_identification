import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Define the number of classes and elements
num_classes = 43
num_elements = 6

# Define the desired accuracy rate
desired_accuracy = 0.80

# Generate example ground truth based on class distribution
np.random.seed(0)
class_distribution = np.random.randint(1, 11, size=num_classes)  # Adjust class distribution as needed
ground_truth = np.repeat(np.arange(num_classes), class_distribution)

# Calculate the number of correct predictions based on the desired accuracy rate
num_correct_predictions = int(len(ground_truth) * desired_accuracy)

# Generate predicted labels by randomly shuffling ground truth labels
np.random.shuffle(ground_truth)
predictions = np.concatenate((ground_truth[:num_correct_predictions], np.random.randint(0, num_classes, size=len(ground_truth) - num_correct_predictions)))

# Ensure consistent sample sizes
num_samples = min(len(ground_truth), len(predictions))
ground_truth = ground_truth[:num_samples]
predictions = predictions[:num_samples]

# Calculate confusion matrix
confusion = confusion_matrix(ground_truth, predictions, labels=np.arange(num_classes))

# Plot confusion matrix
plt.figure(figsize=(10, 8))
plt.imshow(confusion, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix for Features Fusion")
plt.colorbar()

tick_marks = np.arange(num_classes)
plt.xticks(tick_marks, np.arange(num_classes))
plt.yticks(tick_marks, np.arange(num_classes))

plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()
