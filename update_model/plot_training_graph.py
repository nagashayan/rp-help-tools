import matplotlib.pyplot as plt
import seaborn as sns

# Data pulled straight from your v6_notes.md (Early Stopping at Epoch 6 for Phase 1)
acc_phase1 = [0.5583, 0.6967, 0.6985, 0.7487, 0.7782, 0.7965]
val_acc_phase1 = [0.5600, 0.6240, 0.6000, 0.6000, 0.5520, 0.5360]
loss_phase1 = [0.7440, 0.6429, 0.5397, 0.5260, 0.4786, 0.4346]
val_loss_phase1 = [0.7067, 0.7275, 0.7178, 0.7293, 0.8044, 0.8380]

# Phase 2 Data (15 Epochs)
acc_phase2 = [0.6252, 0.6865, 0.6606, 0.7008, 0.6942, 0.7109, 0.7403, 0.7724, 0.7385, 0.7780, 0.7881, 0.7983, 0.8028, 0.7994, 0.8203]
val_acc_phase2 = [0.6320, 0.6160, 0.5840, 0.5840, 0.5920, 0.5840, 0.6320, 0.6320, 0.6080, 0.5840, 0.6080, 0.6000, 0.5760, 0.5600, 0.6240]
loss_phase2 = [0.6526, 0.5948, 0.6155, 0.5680, 0.5859, 0.5732, 0.5163, 0.5029, 0.5131, 0.4461, 0.4613, 0.4351, 0.4254, 0.4485, 0.4220]
val_loss_phase2 = [0.6669, 0.6753, 0.7134, 0.7161, 0.6841, 0.7335, 0.7072, 0.7234, 0.7637, 0.7856, 0.7486, 0.7397, 0.7752, 0.7869, 0.7796]

all_epochs = list(range(1, 22))
all_acc = acc_phase1 + acc_phase2
all_val_acc = val_acc_phase1 + val_acc_phase2
all_loss = loss_phase1 + loss_phase2
all_val_loss = val_loss_phase1 + val_loss_phase2

# Setup 1x2 subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
sns.set_theme(style="whitegrid")

# --- Plot 1: Accuracy ---
ax1.plot(all_epochs, all_acc, label='Training Accuracy', color='#1f77b4', linewidth=2)
ax1.plot(all_epochs, all_val_acc, label='Validation Accuracy', color='#ff7f0e', linewidth=2)
ax1.axvline(x=6.5, color='red', linestyle='--', alpha=0.8, label='Fine-Tuning Started')
ax1.set_title('Accuracy: Phase 1 vs Phase 2', fontsize=13, weight='bold')
ax1.set_xlabel('Total Epochs', fontsize=11, weight='bold')
ax1.set_ylabel('Accuracy', fontsize=11, weight='bold')
ax1.legend(loc='lower right')

# --- Plot 2: Loss ---
ax2.plot(all_epochs, all_loss, label='Training Loss', color='#1f77b4', linewidth=2)
ax2.plot(all_epochs, all_val_loss, label='Validation Loss', color='#ff7f0e', linewidth=2)
ax2.axvline(x=6.5, color='red', linestyle='--', alpha=0.8, label='Fine-Tuning Started')
ax2.set_title('Loss: Phase 1 vs Phase 2', fontsize=13, weight='bold')
ax2.set_xlabel('Total Epochs', fontsize=11, weight='bold')
ax2.set_ylabel('Loss', fontsize=11, weight='bold')
ax2.legend(loc='upper right')

plt.tight_layout()
plt.savefig('learning_curves_updated.png', bbox_inches='tight')
print("✅ Saved as 'learning_curves_updated.png'")