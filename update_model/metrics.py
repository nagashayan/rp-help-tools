import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import tensorflow as tf

# Ensure your validation_data generator is defined and NOT shuffled for evaluation
validation_data.shuffle = False
validation_data.reset()

# 1. Predict
Y_pred = model.predict(validation_data)
y_pred = (Y_pred > 0.5).astype(int)

# 2. Calculate Precision, Recall, and F1
precision, recall, f1, _ = precision_recall_fscore_support(validation_data.classes, y_pred, average='binary')
tn, fp, fn, tp = confusion_matrix(validation_data.classes, y_pred).ravel()
specificity = tn / (tn + fp)

print(f"--- Technical Metrics for Paper ---")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")
print(f"Specificity: {specificity:.4f}")

# 3. Save Confusion Matrix
cm = confusion_matrix(validation_data.classes, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
            xticklabels=['None', 'Handshake'], 
            yticklabels=['None', 'Handshake'])
plt.ylabel('Ground Truth')
plt.xlabel('System Prediction')
plt.title('Confusion Matrix: Real-time Handshake Verification')
plt.savefig('confusion_matrix_final.png')