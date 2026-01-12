# create_visualizations.py
# This script generates comprehensive visualizations for bearing fault diagnosis results,
# including training curves, confusion matrices, cross-validation results, and comparison charts.
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")

# Create output directory
output_dir = Path("visualizations")
output_dir.mkdir(exist_ok=True)

print("=" * 70)
print("CREATING COMPREHENSIVE VISUALIZATIONS")
print("=" * 70)

# ==================== Load Data ====================
print("\nLoading training history data...")

with open("Paderborn_4Class_Supervised/outputs_4class_improved/paderborn_4class_improved_history.json", 'r') as f:
    data_4class = json.load(f)

with open("Paderborn_3Class_Supervised/outputs_3class_improved/paderborn_3class_improved_history.json", 'r') as f:
    data_3class = json.load(f)

print(f"4-Class Model: {len(data_4class['train_loss'])} epochs, Test Acc: {data_4class['test_acc']:.2f}%")
print(f"3-Class Model: {len(data_3class['train_loss'])} epochs, Test Acc: {data_3class['test_acc']:.2f}%")

# ==================== 1. Training Curves (4-Class) ====================
print("\n[1/7] Creating 4-class training curves...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss curve
epochs = range(1, len(data_4class['train_loss']) + 1)
axes[0].plot(epochs, data_4class['train_loss'], 'b-', linewidth=2, label='Train Loss', alpha=0.8)
axes[0].plot(epochs, data_4class['val_loss'], 'r-', linewidth=2, label='Val Loss', alpha=0.8)
axes[0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Cross-Entropy Loss', fontsize=12, fontweight='bold')
axes[0].set_title('4-Class Model: Training and Validation Loss', fontsize=13, fontweight='bold')
axes[0].legend(fontsize=11, frameon=True, shadow=True)
axes[0].grid(True, alpha=0.3)
axes[0].set_xlim([1, len(epochs)])

# Accuracy curve
axes[1].plot(epochs, data_4class['train_acc'], 'b-', linewidth=2, label='Train Accuracy', alpha=0.8)
axes[1].plot(epochs, data_4class['val_acc'], 'r-', linewidth=2, label='Val Accuracy', alpha=0.8)
axes[1].axhline(y=data_4class['test_acc'], color='g', linestyle='--', linewidth=2, label=f'Test Acc: {data_4class["test_acc"]:.2f}%')
axes[1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
axes[1].set_title('4-Class Model: Training and Validation Accuracy', fontsize=13, fontweight='bold')
axes[1].legend(fontsize=11, frameon=True, shadow=True)
axes[1].grid(True, alpha=0.3)
axes[1].set_xlim([1, len(epochs)])
axes[1].set_ylim([65, 100])

plt.tight_layout()
plt.savefig(output_dir / "01_training_curves_4class.png", dpi=300, bbox_inches='tight')
plt.savefig(output_dir / "01_training_curves_4class.pdf", bbox_inches='tight')
print(f"   Saved: {output_dir / '01_training_curves_4class.png'}")
plt.close()

# ==================== 2. Training Curves (3-Class) ====================
print("[2/7] Creating 3-class training curves...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss curve
epochs = range(1, len(data_3class['train_loss']) + 1)
axes[0].plot(epochs, data_3class['train_loss'], 'b-', linewidth=2, label='Train Loss', alpha=0.8)
axes[0].plot(epochs, data_3class['val_loss'], 'r-', linewidth=2, label='Val Loss', alpha=0.8)
axes[0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Cross-Entropy Loss', fontsize=12, fontweight='bold')
axes[0].set_title('3-Class Model: Training and Validation Loss', fontsize=13, fontweight='bold')
axes[0].legend(fontsize=11, frameon=True, shadow=True)
axes[0].grid(True, alpha=0.3)
axes[0].set_xlim([1, len(epochs)])

# Accuracy curve
axes[1].plot(epochs, data_3class['train_acc'], 'b-', linewidth=2, label='Train Accuracy', alpha=0.8)
axes[1].plot(epochs, data_3class['val_acc'], 'r-', linewidth=2, label='Val Accuracy', alpha=0.8)
axes[1].axhline(y=data_3class['test_acc'], color='g', linestyle='--', linewidth=2, label=f'Test Acc: {data_3class["test_acc"]:.2f}%')
axes[1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
axes[1].set_title('3-Class Model: Training and Validation Accuracy', fontsize=13, fontweight='bold')
axes[1].legend(fontsize=11, frameon=True, shadow=True)
axes[1].grid(True, alpha=0.3)
axes[1].set_xlim([1, len(epochs)])
axes[1].set_ylim([65, 100])

plt.tight_layout()
plt.savefig(output_dir / "02_training_curves_3class.png", dpi=300, bbox_inches='tight')
plt.savefig(output_dir / "02_training_curves_3class.pdf", bbox_inches='tight')
print(f"   Saved: {output_dir / '02_training_curves_3class.png'}")
plt.close()

# ==================== 3. Confusion Matrix (4-Class) ====================
print("[3/7] Creating 4-class confusion matrix heatmap...")

cm_4class = np.array(data_4class['confusion_matrix'])
class_names_4 = ['Healthy', 'Inner', 'Outer', 'Cage']

# Normalize confusion matrix to percentages
cm_4class_pct = cm_4class.astype('float') / cm_4class.sum(axis=1)[:, np.newaxis] * 100

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(cm_4class_pct, annot=True, fmt='.1f', cmap='Blues', cbar_kws={'label': 'Percentage (%)'},
            xticklabels=class_names_4, yticklabels=class_names_4, ax=ax, linewidths=0.5, linecolor='gray')
ax.set_xlabel('Predicted Label', fontsize=13, fontweight='bold')
ax.set_ylabel('True Label', fontsize=13, fontweight='bold')
ax.set_title(f'4-Class Confusion Matrix (Test Accuracy: {data_4class["test_acc"]:.2f}%)', fontsize=14, fontweight='bold')

# Add counts as secondary annotation
for i in range(len(class_names_4)):
    for j in range(len(class_names_4)):
        count = cm_4class[i, j]
        pct = cm_4class_pct[i, j]
        ax.text(j + 0.5, i + 0.7, f'n={count}', ha='center', va='center', fontsize=9, color='gray')

plt.tight_layout()
plt.savefig(output_dir / "03_confusion_matrix_4class.png", dpi=300, bbox_inches='tight')
plt.savefig(output_dir / "03_confusion_matrix_4class.pdf", bbox_inches='tight')
print(f"   Saved: {output_dir / '03_confusion_matrix_4class.png'}")
plt.close()

# ==================== 4. Confusion Matrix (3-Class) ====================
print("[4/7] Creating 3-class confusion matrix heatmap...")

cm_3class = np.array(data_3class['confusion_matrix'])
class_names_3 = ['Healthy', 'Inner', 'Outer']

# Normalize confusion matrix to percentages
cm_3class_pct = cm_3class.astype('float') / cm_3class.sum(axis=1)[:, np.newaxis] * 100

fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(cm_3class_pct, annot=True, fmt='.1f', cmap='Blues', cbar_kws={'label': 'Percentage (%)'},
            xticklabels=class_names_3, yticklabels=class_names_3, ax=ax, linewidths=0.5, linecolor='gray')
ax.set_xlabel('Predicted Label', fontsize=13, fontweight='bold')
ax.set_ylabel('True Label', fontsize=13, fontweight='bold')
ax.set_title(f'3-Class Confusion Matrix (Test Accuracy: {data_3class["test_acc"]:.2f}%)', fontsize=14, fontweight='bold')

# Add counts as secondary annotation
for i in range(len(class_names_3)):
    for j in range(len(class_names_3)):
        count = cm_3class[i, j]
        pct = cm_3class_pct[i, j]
        ax.text(j + 0.5, i + 0.7, f'n={count}', ha='center', va='center', fontsize=9, color='gray')

plt.tight_layout()
plt.savefig(output_dir / "04_confusion_matrix_3class.png", dpi=300, bbox_inches='tight')
plt.savefig(output_dir / "04_confusion_matrix_3class.pdf", bbox_inches='tight')
print(f"   Saved: {output_dir / '04_confusion_matrix_3class.png'}")
plt.close()

# ==================== 5. Cross-Validation Results ====================
print("[5/7] Creating cross-validation results plot...")

# Data from cross-validation (from README)
cv_seeds = ['Seed 42\n(Best)', 'Seed 123', 'Seed 456', 'Seed 789', 'Seed 1011', 'Mean ± Std']
cv_test_acc = [99.26, 93.75, 96.88, 97.92, 94.79, 95.00]
cv_test_std = [0, 0, 0, 0, 0, 2.22]
cv_bal_acc = [99.31, 91.67, 96.53, 97.22, 93.75, 93.75]
cv_bal_std = [0, 0, 0, 0, 0, 2.88]

x_pos = np.arange(len(cv_seeds))

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Test Accuracy
bars1 = axes[0].bar(x_pos[:5], cv_test_acc[:5], color=['#2ecc71', '#3498db', '#3498db', '#3498db', '#3498db'], alpha=0.8, edgecolor='black', linewidth=1.5)
bars2 = axes[0].bar(x_pos[5], cv_test_acc[5], yerr=cv_test_std[5], capsize=10, color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.5)
axes[0].set_xlabel('Random Seed', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
axes[0].set_title('3-Class Model: 5-Seed Cross-Validation Results', fontsize=13, fontweight='bold')
axes[0].set_xticks(x_pos)
axes[0].set_xticklabels(cv_seeds, fontsize=10)
axes[0].set_ylim([88, 102])
axes[0].grid(axis='y', alpha=0.3)
axes[0].axhline(y=95.00, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Mean: 95.00%')

# Add value labels on bars
all_bars = list(bars1.patches) + list(bars2.patches)
for i, (bar, val) in enumerate(zip(all_bars, cv_test_acc)):
    height = bar.get_height()
    if i < 5:
        axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.5, f'{val:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    else:
        axes[0].text(bar.get_x() + bar.get_width()/2., height + cv_test_std[i] + 0.5, f'{val:.2f}±{cv_test_std[i]:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

axes[0].legend(fontsize=10, frameon=True, shadow=True)

# Balanced Accuracy
bars3 = axes[1].bar(x_pos[:5], cv_bal_acc[:5], color=['#2ecc71', '#3498db', '#3498db', '#3498db', '#3498db'], alpha=0.8, edgecolor='black', linewidth=1.5)
bars4 = axes[1].bar(x_pos[5], cv_bal_acc[5], yerr=cv_bal_std[5], capsize=10, color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.5)
axes[1].set_xlabel('Random Seed', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Balanced Accuracy (%)', fontsize=12, fontweight='bold')
axes[1].set_title('3-Class Model: 5-Seed Cross-Validation (Balanced)', fontsize=13, fontweight='bold')
axes[1].set_xticks(x_pos)
axes[1].set_xticklabels(cv_seeds, fontsize=10)
axes[1].set_ylim([88, 102])
axes[1].grid(axis='y', alpha=0.3)
axes[1].axhline(y=93.75, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Mean: 93.75%')

# Add value labels on bars
all_bars_bal = list(bars3.patches) + list(bars4.patches)
for i, (bar, val) in enumerate(zip(all_bars_bal, cv_bal_acc)):
    height = bar.get_height()
    if i < 5:
        axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.5, f'{val:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    else:
        axes[1].text(bar.get_x() + bar.get_width()/2., height + cv_bal_std[i] + 0.5, f'{val:.2f}±{cv_bal_std[i]:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

axes[1].legend(fontsize=10, frameon=True, shadow=True)

plt.tight_layout()
plt.savefig(output_dir / "05_cross_validation_results.png", dpi=300, bbox_inches='tight')
plt.savefig(output_dir / "05_cross_validation_results.pdf", bbox_inches='tight')
print(f"   Saved: {output_dir / '05_cross_validation_results.png'}")
plt.close()

# ==================== 6. Comparison Bar Chart ====================
print("[6/7] Creating comparison bar chart...")

experiments = ['CWRU\nBaseline', 'CWRU\nImproved', 'Paderborn\n3-Class\nBaseline', 'Paderborn\n3-Class\nImproved', 'Paderborn\n4-Class\nImproved']
accuracies = [81.82, 90.91, 91.67, 99.26, 98.15]
colors = ['#95a5a6', '#3498db', '#95a5a6', '#2ecc71', '#2ecc71']

fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.bar(experiments, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

ax.set_ylabel('Test Accuracy (%)', fontsize=13, fontweight='bold')
ax.set_title('Comparison of All Experiments (Baseline vs Improved)', fontsize=14, fontweight='bold')
ax.set_ylim([75, 102])
ax.grid(axis='y', alpha=0.3)
ax.axhline(y=95, color='red', linestyle='--', linewidth=1.5, alpha=0.5, label='95% Target')

# Add value labels on bars
for bar, val in zip(bars.patches, accuracies):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.5, f'{val:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Add improvement annotations
ax.annotate('', xy=(1, 90.91), xytext=(0, 81.82), arrowprops=dict(arrowstyle='->', lw=2, color='green'))
ax.text(0.5, 86.5, '+9.09%', ha='center', fontsize=10, color='green', fontweight='bold', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

ax.annotate('', xy=(3, 99.26), xytext=(2, 91.67), arrowprops=dict(arrowstyle='->', lw=2, color='green'))
ax.text(2.5, 95.5, '+7.59%', ha='center', fontsize=10, color='green', fontweight='bold', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

ax.legend(fontsize=11, frameon=True, shadow=True)

plt.tight_layout()
plt.savefig(output_dir / "06_comparison_all_experiments.png", dpi=300, bbox_inches='tight')
plt.savefig(output_dir / "06_comparison_all_experiments.pdf", bbox_inches='tight')
print(f"   Saved: {output_dir / '06_comparison_all_experiments.png'}")
plt.close()

# ==================== 7. Combined Summary Dashboard ====================
print("[7/7] Creating combined summary dashboard...")

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 4-Class Loss
ax1 = fig.add_subplot(gs[0, 0])
epochs_4 = range(1, len(data_4class['train_loss']) + 1)
ax1.plot(epochs_4, data_4class['train_loss'], 'b-', linewidth=1.5, label='Train', alpha=0.7)
ax1.plot(epochs_4, data_4class['val_loss'], 'r-', linewidth=1.5, label='Val', alpha=0.7)
ax1.set_xlabel('Epoch', fontsize=10, fontweight='bold')
ax1.set_ylabel('Loss', fontsize=10, fontweight='bold')
ax1.set_title('4-Class: Loss Curves', fontsize=11, fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# 4-Class Accuracy
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(epochs_4, data_4class['train_acc'], 'b-', linewidth=1.5, label='Train', alpha=0.7)
ax2.plot(epochs_4, data_4class['val_acc'], 'r-', linewidth=1.5, label='Val', alpha=0.7)
ax2.axhline(y=data_4class['test_acc'], color='g', linestyle='--', linewidth=1.5, label=f'Test: {data_4class["test_acc"]:.2f}%')
ax2.set_xlabel('Epoch', fontsize=10, fontweight='bold')
ax2.set_ylabel('Accuracy (%)', fontsize=10, fontweight='bold')
ax2.set_title('4-Class: Accuracy Curves', fontsize=11, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_ylim([65, 100])

# 4-Class Confusion Matrix
ax3 = fig.add_subplot(gs[0, 2])
sns.heatmap(cm_4class_pct, annot=True, fmt='.1f', cmap='Blues', cbar=False,
            xticklabels=class_names_4, yticklabels=class_names_4, ax=ax3, linewidths=0.5)
ax3.set_xlabel('Predicted', fontsize=10, fontweight='bold')
ax3.set_ylabel('True', fontsize=10, fontweight='bold')
ax3.set_title(f'4-Class: CM ({data_4class["test_acc"]:.2f}%)', fontsize=11, fontweight='bold')

# 3-Class Loss
ax4 = fig.add_subplot(gs[1, 0])
epochs_3 = range(1, len(data_3class['train_loss']) + 1)
ax4.plot(epochs_3, data_3class['train_loss'], 'b-', linewidth=1.5, label='Train', alpha=0.7)
ax4.plot(epochs_3, data_3class['val_loss'], 'r-', linewidth=1.5, label='Val', alpha=0.7)
ax4.set_xlabel('Epoch', fontsize=10, fontweight='bold')
ax4.set_ylabel('Loss', fontsize=10, fontweight='bold')
ax4.set_title('3-Class: Loss Curves', fontsize=11, fontweight='bold')
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3)

# 3-Class Accuracy
ax5 = fig.add_subplot(gs[1, 1])
ax5.plot(epochs_3, data_3class['train_acc'], 'b-', linewidth=1.5, label='Train', alpha=0.7)
ax5.plot(epochs_3, data_3class['val_acc'], 'r-', linewidth=1.5, label='Val', alpha=0.7)
ax5.axhline(y=data_3class['test_acc'], color='g', linestyle='--', linewidth=1.5, label=f'Test: {data_3class["test_acc"]:.2f}%')
ax5.set_xlabel('Epoch', fontsize=10, fontweight='bold')
ax5.set_ylabel('Accuracy (%)', fontsize=10, fontweight='bold')
ax5.set_title('3-Class: Accuracy Curves', fontsize=11, fontweight='bold')
ax5.legend(fontsize=9)
ax5.grid(True, alpha=0.3)
ax5.set_ylim([65, 100])

# 3-Class Confusion Matrix
ax6 = fig.add_subplot(gs[1, 2])
sns.heatmap(cm_3class_pct, annot=True, fmt='.1f', cmap='Blues', cbar=False,
            xticklabels=class_names_3, yticklabels=class_names_3, ax=ax6, linewidths=0.5)
ax6.set_xlabel('Predicted', fontsize=10, fontweight='bold')
ax6.set_ylabel('True', fontsize=10, fontweight='bold')
ax6.set_title(f'3-Class: CM ({data_3class["test_acc"]:.2f}%)', fontsize=11, fontweight='bold')

# Cross-Validation
ax7 = fig.add_subplot(gs[2, :2])
x_pos_cv = np.arange(len(cv_seeds))
bars_cv = ax7.bar(x_pos_cv[:5], cv_test_acc[:5], color=['#2ecc71', '#3498db', '#3498db', '#3498db', '#3498db'], alpha=0.8, edgecolor='black')
bars_mean = ax7.bar(x_pos_cv[5], cv_test_acc[5], yerr=cv_test_std[5], capsize=8, color='#e74c3c', alpha=0.8, edgecolor='black')
ax7.set_xlabel('Random Seed', fontsize=10, fontweight='bold')
ax7.set_ylabel('Test Accuracy (%)', fontsize=10, fontweight='bold')
ax7.set_title('3-Class: Cross-Validation Results', fontsize=11, fontweight='bold')
ax7.set_xticks(x_pos_cv)
ax7.set_xticklabels(cv_seeds, fontsize=9)
ax7.set_ylim([88, 102])
ax7.grid(axis='y', alpha=0.3)
ax7.axhline(y=95.00, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
all_bars_cv_dash = list(bars_cv.patches) + list(bars_mean.patches)
for i, (bar, val) in enumerate(zip(all_bars_cv_dash, cv_test_acc)):
    height = bar.get_height()
    if i < 5:
        ax7.text(bar.get_x() + bar.get_width()/2., height + 0.3, f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
    else:
        ax7.text(bar.get_x() + bar.get_width()/2., height + cv_test_std[i] + 0.3, f'{val:.2f}±{cv_test_std[i]:.2f}%', ha='center', va='bottom', fontsize=9)

# Comparison Chart
ax8 = fig.add_subplot(gs[2, 2])
exp_short = ['CWRU\nBase', 'CWRU\nImp', 'Pad\n3-Base', 'Pad\n3-Imp', 'Pad\n4-Imp']
bars_comp = ax8.bar(exp_short, accuracies, color=colors, alpha=0.8, edgecolor='black')
ax8.set_ylabel('Test Acc (%)', fontsize=10, fontweight='bold')
ax8.set_title('All Experiments', fontsize=11, fontweight='bold')
ax8.set_ylim([75, 102])
ax8.grid(axis='y', alpha=0.3)
ax8.tick_params(axis='x', labelsize=8)
for bar, val in zip(bars_comp.patches, accuracies):
    height = bar.get_height()
    ax8.text(bar.get_x() + bar.get_width()/2., height + 0.3, f'{val:.1f}%', ha='center', va='bottom', fontsize=8)

# Add main title
fig.suptitle('Bearing Fault Diagnosis: Complete Results Dashboard', fontsize=16, fontweight='bold', y=0.98)

plt.savefig(output_dir / "07_complete_dashboard.png", dpi=300, bbox_inches='tight')
plt.savefig(output_dir / "07_complete_dashboard.pdf", bbox_inches='tight')
print(f"   Saved: {output_dir / '07_complete_dashboard.png'}")
plt.close()

# ==================== Summary ====================
print("\n" + "=" * 70)
print("VISUALIZATION GENERATION COMPLETE")
print("=" * 70)
print(f"\nAll visualizations saved to: {output_dir.absolute()}")
print("\nGenerated files:")
print("  1. 01_training_curves_4class.png/pdf       - Loss and accuracy curves (4-class)")
print("  2. 02_training_curves_3class.png/pdf       - Loss and accuracy curves (3-class)")
print("  3. 03_confusion_matrix_4class.png/pdf      - Confusion matrix heatmap (4-class)")
print("  4. 04_confusion_matrix_3class.png/pdf      - Confusion matrix heatmap (3-class)")
print("  5. 05_cross_validation_results.png/pdf     - 5-seed cross-validation bar plots")
print("  6. 06_comparison_all_experiments.png/pdf   - Comparison of all experiments")
print("  7. 07_complete_dashboard.png/pdf           - Combined summary dashboard")
print("\nTotal: 7 visualizations x 2 formats = 14 files")
print("=" * 70)
