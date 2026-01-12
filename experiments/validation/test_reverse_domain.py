"""Test Paderborn-trained model on CWRU data (reverse domain adaptation)."""
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score

# Add paths
sys.path.append(str(Path(__file__).parent / "Paderborn_Supervised"))
sys.path.append(str(Path(__file__).parent / "DANN_CrossDataset"))

from Paderborn_Supervised.model_paderborn_3class import create_model_3class
from DANN_CrossDataset.dataset_dann import load_cwru_signal
from Paderborn_Supervised.dataset_paderborn_3class_windowed import extract_envelope_spectrum, extract_multiscale_fft, extract_statistics


def create_cwru_test_loader(cwru_root: str, batch_size: int = 32):
    """Create CWRU test dataloader in Paderborn format."""
    from scipy.io import loadmat
    from torch.utils.data import Dataset, DataLoader

    class CWRUTestDataset(Dataset):
        def __init__(self, cwru_root: str):
            self.samples = []

            # 3 classes: Healthy, Inner, Outer
            # CWRU has Normal (0), Inner (1), Outer (2) - same as Paderborn!

            cwru_path = Path(cwru_root)
            mat_files = list(cwru_path.glob("*.mat"))

            print(f"\nFound {len(mat_files)} CWRU .mat files")

            for mat_file in mat_files:
                try:
                    # Determine label from filename
                    fname = mat_file.stem
                    if 'Normal' in fname or 'normal' in fname:
                        label = 0  # Healthy
                    elif 'IR' in fname or 'inner' in fname.lower():
                        label = 1  # Inner damage
                    elif 'OR' in fname or 'B' in fname or 'outer' in fname.lower():
                        label = 2  # Outer damage
                    else:
                        continue  # Skip unknown files

                    # Load signal
                    signal = load_cwru_signal(str(mat_file))

                    # Use first 64k samples (same as training)
                    if len(signal) > 65536:
                        signal = signal[:65536]

                    # Create windows
                    window_size = 2048
                    step = 512  # Overlapping

                    for start in range(0, len(signal) - window_size + 1, step):
                        window = signal[start:start + window_size]

                        # Extract features
                        envelope = extract_envelope_spectrum(window)
                        fft1, fft2, fft3 = extract_multiscale_fft(window)
                        stats = extract_statistics(window)

                        self.samples.append({
                            'envelope': torch.FloatTensor(envelope),
                            'fft_scale1': torch.FloatTensor(fft1),
                            'fft_scale2': torch.FloatTensor(fft2),
                            'fft_scale3': torch.FloatTensor(fft3),
                            'stats': torch.FloatTensor(stats),
                            'label': label,
                            'file': fname
                        })

                except Exception as e:
                    print(f"Error loading {mat_file.name}: {e}")
                    continue

            print(f"Created {len(self.samples)} CWRU test windows")

            # Count per class
            labels = [s['label'] for s in self.samples]
            print(f"  Healthy: {labels.count(0)} windows")
            print(f"  Inner: {labels.count(1)} windows")
            print(f"  Outer: {labels.count(2)} windows")

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            return self.samples[idx]

    dataset = CWRUTestDataset(cwru_root)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return loader


def test_paderborn_on_cwru(
    paderborn_checkpoint: str,
    cwru_root: str,
    batch_size: int = 32
):
    """Test Paderborn model on CWRU data."""

    print("\n" + "="*80)
    print("REVERSE DOMAIN ADAPTATION TEST")
    print("="*80)
    print("\nTrain: Paderborn (640 files, 99.26% test acc)")
    print("Test:  CWRU (different dataset)")
    print("\nQuestion: Does Paderborn model generalize to CWRU?")
    print("="*80)

    # Device
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Load CWRU test data
    print("\nLoading CWRU test data...")
    test_loader = create_cwru_test_loader(cwru_root, batch_size)

    # Create model
    print("\nCreating model...")
    sample_batch = next(iter(test_loader))
    model = create_model_3class(
        sample_batch=sample_batch,
        hidden_dim=64,
        fusion_dim=128,
        num_classes=3,
        dropout=0.5  # Same as training
    ).to(device)

    # Load Paderborn checkpoint
    print(f"\nLoading Paderborn checkpoint: {paderborn_checkpoint}")
    checkpoint = torch.load(paderborn_checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Checkpoint trained to {checkpoint.get('val_acc', 'N/A')}% validation accuracy on Paderborn")

    # Test
    print("\nTesting on CWRU data...")
    model.eval()

    all_preds = []
    all_labels = []
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            envelope = batch['envelope'].to(device)
            fft1 = batch['fft_scale1'].to(device)
            fft2 = batch['fft_scale2'].to(device)
            fft3 = batch['fft_scale3'].to(device)
            stats = batch['stats'].to(device)
            labels = batch['label'].to(device)

            # Forward
            logits = model(envelope, fft1, fft2, fft3, stats)
            _, predicted = logits.max(1)

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Results
    accuracy = 100. * correct / total
    balanced_acc = 100. * balanced_accuracy_score(all_labels, all_preds)

    print("\n" + "="*80)
    print("RESULTS: Paderborn Model on CWRU Data")
    print("="*80)
    print(f"Test Accuracy:          {accuracy:.2f}%")
    print(f"Test Balanced Accuracy: {balanced_acc:.2f}%")

    # Classification report
    print("\n" + "="*80)
    print("CLASSIFICATION REPORT")
    print("="*80)
    print(classification_report(
        all_labels,
        all_preds,
        target_names=['Healthy', 'Inner Damage', 'Outer Damage'],
        digits=4
    ))

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("\nConfusion Matrix:")
    print("                    Predicted")
    print("              Healthy  Inner  Outer")
    print(f"Actual Healthy   {cm[0,0]:5d}   {cm[0,1]:5d}  {cm[0,2]:5d}")
    print(f"       Inner     {cm[1,0]:5d}   {cm[1,1]:5d}  {cm[1,2]:5d}")
    print(f"       Outer     {cm[2,0]:5d}   {cm[2,1]:5d}  {cm[2,2]:5d}")

    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)
    print(f"Paderborn model accuracy on Paderborn: 99.26%")
    print(f"Paderborn model accuracy on CWRU:      {accuracy:.2f}%")
    print(f"Generalization gap:                     {99.26 - accuracy:.2f}%")

    if accuracy > 80:
        print("\nOK GOOD: Model generalizes well across datasets!")
    elif accuracy > 50:
        print("\nWARNING: MODERATE: Some generalization, but domain gap exists")
    else:
        print("\nERROR: POOR: Model does not generalize across domains")

    print("="*80)

    return accuracy, balanced_acc


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent

    # Paths
    paderborn_checkpoint = str(Path(__file__).parent / "Paderborn_Supervised" / "checkpoints_3class_improved" / "best_model.pth")
    cwru_data = str(base_dir / "CWRU_Project" / "data")

    # Test
    test_paderborn_on_cwru(
        paderborn_checkpoint=paderborn_checkpoint,
        cwru_root=cwru_data,
        batch_size=64
    )
