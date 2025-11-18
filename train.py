import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
from config import DATA_PICKLE_PATH, MODEL_PATH, ASL_LETTERS, LABELS_DICT

print("=" * 60)
print("Training ASL Recognition Model")
print("=" * 60)

# Load data
try:
    data_dict = pickle.load(open(DATA_PICKLE_PATH, 'rb'))
    print(f" Loaded data from: {DATA_PICKLE_PATH}")
except FileNotFoundError:
    print(f"Error: {DATA_PICKLE_PATH} not found!")
    print("Please run 'python createData_set.py' first")
    exit(1)

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

print(f"\nDataset info:")
print(f"  Total samples: {len(data)}")
print(f"  Feature dimensions: {data.shape}")
print(f"  Unique classes: {len(np.unique(labels))}")

# Check for data consistency
if len(data) == 0:
    print("Error: No data found in dataset!")
    exit(1)

# Split data
print("\nSplitting data (80% train, 20% test)...")
x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, stratify=labels, random_state=42
)

print(f"  Training samples: {len(x_train)}")
print(f"  Testing samples: {len(x_test)}")

# Train model with optimized parameters
print("\nTraining Random Forest Classifier...")
model = RandomForestClassifier()

model.fit(x_train, y_train)
print("Training complete!")

# Test model
print("\nEvaluating model...")
y_predict = model.predict(x_test)
score = accuracy_score(y_test, y_predict)

print(f"\n{'='*60}")
print(f" Model Accuracy: {score * 100:.2f}%")
print(f"{'='*60}")

# Detailed classification report
print("\n Detailed Performance Report:")
print("-" * 60)

# Create reverse mapping for labels
class_to_letter = {str(idx): letter for idx, letter in enumerate(ASL_LETTERS)}

# Get unique labels in test set
unique_test_labels = sorted(set(y_test))
target_names = [class_to_letter.get(str(label), f"Class_{label}") for label in unique_test_labels]

print(classification_report(y_test, y_predict, target_names=target_names))

# Show confusion for worst performing classes
print("\n Classes that might need more training data:")
cm = confusion_matrix(y_test, y_predict)
class_accuracies = cm.diagonal() / cm.sum(axis=1)

worst_classes = []
for idx, acc in enumerate(class_accuracies):
    if acc < 0.80:  # Less than 80% accuracy
        class_label = unique_test_labels[idx]
        letter = class_to_letter.get(str(class_label), f"Class_{class_label}")
        worst_classes.append((letter, acc * 100))

if worst_classes:
    for letter, acc in sorted(worst_classes, key=lambda x: x[1]):
        print(f"  {letter}: {acc:.1f}% accuracy")
else:
    print("  All classes performing well (>80% accuracy)")

# Save model
print(f"\n Saving model to: {MODEL_PATH}")
f = open(MODEL_PATH, 'wb')
pickle.dump({'model': model}, f)
f.close()


