import matplotlib.pyplot as plt

# ==========================================
# 1. Authentic Data Extracted from Logs
# ==========================================
# Custom CNN (Stuck at 49% - Memorization/Overfitting)
epochs_cnn = range(1, 16)
train_cnn = [0.61, 0.66, 0.73, 0.73, 0.78, 0.79, 0.82, 0.83, 0.84, 0.85, 0.88, 0.85, 0.89, 0.87, 0.89]
val_cnn = [0.49, 0.49, 0.49, 0.49, 0.48, 0.48, 0.48, 0.48, 0.48, 0.48, 0.49, 0.48, 0.48, 0.48, 0.49]

# MobileNetV3-Small (Underfitting - Low Train & Val)
epochs_v3s = range(1, 23) 
train_v3s = [0.50, 0.50, 0.49, 0.48, 0.50, 0.51, 0.50, 0.54, 0.55, 0.51, 0.55, 0.55, 0.56, 0.53, 0.56, 0.49, 0.54, 0.62, 0.58, 0.60, 0.68, 0.66]
val_v3s = [0.47, 0.52, 0.55, 0.47, 0.52, 0.47, 0.59, 0.56, 0.56, 0.54, 0.55, 0.58, 0.50, 0.58, 0.60, 0.59, 0.55, 0.63, 0.55, 0.59, 0.53, 0.53]

# MobileNetV3-Large (Overfitting - High Train, Plateaued Val)
epochs_v3l = range(1, 15)
train_v3l = [0.59, 0.72, 0.82, 0.88, 0.88, 0.91, 0.90, 0.94, 0.92, 0.84, 0.90, 0.91, 0.93, 0.92]
val_v3l = [0.55, 0.60, 0.61, 0.61, 0.65, 0.61, 0.66, 0.65, 0.59, 0.62, 0.66, 0.66, 0.65, 0.63]

# MobileNetV2 (The Champion - Optimal Balance)
epochs_v2 = range(1, 11) 
train_v2 = [0.63, 0.75, 0.82, 0.85, 0.86, 0.72, 0.84, 0.86, 0.92, 0.92]
val_v2 = [0.63, 0.59, 0.62, 0.60, 0.61, 0.65, 0.68, 0.69, 0.66, 0.65]

# ==========================================
# 2. IEEE Formatting & Plotting
# ==========================================
# Use a serif font typical for academic papers if available
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 12

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

# Colors and Markers for clarity in black & white printing
style_cnn = {'color': '#d62728', 'linestyle': ':', 'linewidth': 2, 'label': 'Custom CNN (Baseline)'}
style_v3s = {'color': '#ff7f0e', 'linestyle': '--', 'linewidth': 2, 'label': 'CNN + MobileNetV3-Small'}
style_v3l = {'color': '#2ca02c', 'linestyle': '-.', 'linewidth': 2, 'label': 'CNN + MobileNetV3-Large'}
style_v2  = {'color': '#1f77b4', 'linestyle': '-', 'linewidth': 3, 'label': 'CNN + MobileNetV2 (Ours)'}

# --- SUBPLOT 1: Training Accuracy ---
ax1.plot(epochs_cnn, train_cnn, **style_cnn)
ax1.plot(epochs_v3s, train_v3s, **style_v3s)
ax1.plot(epochs_v3l, train_v3l, **style_v3l)
ax1.plot(epochs_v2, train_v2, **style_v2)

ax1.set_title('Training Accuracy Across Architectures', fontweight='bold')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Accuracy')
ax1.set_ylim(0.45, 1.0)
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend(loc='lower right')

# --- SUBPLOT 2: Validation Accuracy ---
ax2.plot(epochs_cnn, val_cnn, **style_cnn)
ax2.plot(epochs_v3s, val_v3s, **style_v3s)
ax2.plot(epochs_v3l, val_v3l, **style_v3l)
ax2.plot(epochs_v2, val_v2, **style_v2)

ax2.set_title('Validation Accuracy Across Architectures', fontweight='bold')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Accuracy')
ax2.set_ylim(0.45, 0.75) # Zoomed in to show the V2 advantage
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.legend(loc='lower right')

plt.tight_layout()
plt.savefig('architecture_comparison.png', bbox_inches='tight')
print("✅ Graph saved successfully as architecture_comparison.png")