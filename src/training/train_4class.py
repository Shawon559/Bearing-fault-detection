"""Train improved 4-class model on Paderborn dataset."""
import json
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score

from dataset_paderborn_4class_windowed import Paderborn4ClassWindowedDataset, create_balanced_loader, create_loader
from model_paderborn_4class import create_model_4class


def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()

    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="Training")
    for batch in pbar:
        # Move to device
        envelope = batch['envelope'].to(device)
        fft1 = batch['fft_scale1'].to(device)
        fft2 = batch['fft_scale2'].to(device)
        fft3 = batch['fft_scale3'].to(device)
        stats = batch['stats'].to(device)
        labels = batch['label'].to(device)

        # Forward
        optimizer.zero_grad()
        logits = model(envelope, fft1, fft2, fft3, stats)
        loss = criterion(logits, labels)

        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Metrics
        total_loss += loss.item()
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.3f}',
            'acc': f'{100.*correct/total:.1f}%'
        })

    avg_loss = total_loss / len(loader)
    accuracy = 100. * correct / total

    return avg_loss, accuracy


def validate(model, loader, criterion, device):
    """Validate model."""
    model.eval()

    total_loss = 0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating"):
            envelope = batch['envelope'].to(device)
            fft1 = batch['fft_scale1'].to(device)
            fft2 = batch['fft_scale2'].to(device)
            fft3 = batch['fft_scale3'].to(device)
            stats = batch['stats'].to(device)
            labels = batch['label'].to(device)

            # Forward
            logits = model(envelope, fft1, fft2, fft3, stats)
            loss = criterion(logits, labels)

            # Metrics
            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    accuracy = 100. * correct / total
    balanced_acc = 100. * balanced_accuracy_score(all_labels, all_preds)

    return avg_loss, accuracy, balanced_acc, all_preds, all_labels


def train_paderborn_4class_improved(
    data_root: str = "../../data",
    checkpoint_dir: str = "checkpoints_4class_improved",
    output_dir: str = "outputs_4class_improved",
    num_epochs: int = 60,  # Less epochs with cosine annealing
    batch_size: int = 64,  # Larger batch since we have more data
    learning_rate: float = 1e-4,  # Lower LR for cosine annealing
    weight_decay: float = 1e-4,
    dropout: float = 0.5,  # Higher dropout using best practices
    window_size: int = 2048,
    step: int = 512,  # Overlapping windows!
    seed: int = 42,
):
    """Train improved 4-class model on Paderborn."""

    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create directories
    Path(checkpoint_dir).mkdir(exist_ok=True, parents=True)
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    # Device
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("\n" + "="*70)
    print("IMPROVED 4-CLASS SUPERVISED PADERBORN TRAINING")
    print("="*70)
    print("\nImprovements:")
    print(f"  OK Overlapping windows (size={window_size}, stride={step})")
    print(f"  OK Cosine annealing LR scheduler")
    print(f"  OK Higher dropout ({dropout})")
    print(f"  OK {num_epochs} epochs")
    print(f"  OK Multi-scale attention fusion")

    print("\nLoading datasets...")
    train_dataset = Paderborn4ClassWindowedDataset(
        data_root=data_root,
        split='train',
        window_size=window_size,
        step=step,
        seed=seed,
    )

    val_dataset = Paderborn4ClassWindowedDataset(
        data_root=data_root,
        split='val',
        window_size=window_size,
        step=step,
        seed=seed,
    )

    test_dataset = Paderborn4ClassWindowedDataset(
        data_root=data_root,
        split='test',
        window_size=window_size,
        step=step,
        seed=seed,
    )

    # Create balanced train loader
    train_loader = create_balanced_loader(train_dataset, batch_size=batch_size)
    val_loader = create_loader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = create_loader(test_dataset, batch_size=batch_size, shuffle=False)

    print("\nCreating model...")
    sample_batch = next(iter(train_loader))
    model = create_model_4class(
        sample_batch=sample_batch,
        hidden_dim=64,
        fusion_dim=128,
        num_classes=4,  # 4 classes: Healthy, Inner, Outer, Cage
        dropout=dropout  # Higher dropout!
    ).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Cosine annealing scheduler (like literature best practices!)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_balanced_acc': [],
    }

    best_val_acc = 0
    best_epoch = 0

    print("\nStarting training...")
    print("="*70)

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 70)

        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")

        # Validate
        val_loss, val_acc, val_balanced_acc, _, _ = validate(model, val_loader, criterion, device)
        print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%, Balanced Acc: {val_balanced_acc:.2f}%")
        print(f"LR: {scheduler.get_last_lr()[0]:.6f}")

        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_balanced_acc'].append(val_balanced_acc)

        # Step scheduler
        scheduler.step()

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, f"{checkpoint_dir}/best_model.pth")

            print(f"OK New best validation accuracy: {val_acc:.2f}%")

    # Load best model for testing
    print("\n" + "="*70)
    print(f"Loading best model from epoch {best_epoch+1}...")
    checkpoint = torch.load(f"{checkpoint_dir}/best_model.pth")
    model.load_state_dict(checkpoint['model_state_dict'])

    # Test
    test_loss, test_acc, test_balanced_acc, test_preds, test_labels = validate(
        model, test_loader, criterion, device
    )

    # Classification report
    print("\n" + "="*70)
    print("CLASSIFICATION REPORT")
    print("="*70)
    print(classification_report(
        test_labels,
        test_preds,
        target_names=['Healthy', 'Inner Damage', 'Outer Damage', 'Cage Damage'],
        digits=4
    ))

    # Confusion matrix
    cm = confusion_matrix(test_labels, test_preds)
    print("\nConfusion Matrix:")
    print("                    Predicted")
    print("              Healthy  Inner  Outer  Cage")
    print(f"Actual Healthy   {cm[0,0]:3d}     {cm[0,1]:3d}    {cm[0,2]:3d}   {cm[0,3]:3d}")
    print(f"       Inner     {cm[1,0]:3d}     {cm[1,1]:3d}    {cm[1,2]:3d}   {cm[1,3]:3d}")
    print(f"       Outer     {cm[2,0]:3d}     {cm[2,1]:3d}    {cm[2,2]:3d}   {cm[2,3]:3d}")
    print(f"       Cage      {cm[3,0]:3d}     {cm[3,1]:3d}    {cm[3,2]:3d}   {cm[3,3]:3d}")

    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"Test Accuracy:          {test_acc:.2f}%")
    print(f"Test Balanced Accuracy: {test_balanced_acc:.2f}%")
    print("="*70)

    # Save results
    history['best_val_acc'] = best_val_acc
    history['best_epoch'] = best_epoch
    history['test_acc'] = test_acc
    history['test_balanced_acc'] = test_balanced_acc
    history['test_loss'] = test_loss
    history['confusion_matrix'] = cm.tolist()

    with open(f"{output_dir}/paderborn_4class_improved_history.json", 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\nOK Results saved to {output_dir}/paderborn_4class_improved_history.json")
    print("="*70)

    return test_acc


if __name__ == "__main__":
    import sys
    sys.path.append(str(Path(__file__).parent))

    # Use absolute paths
    base_dir = Path(__file__).resolve().parent.parent.parent
    paderborn_data = base_dir / "data"

    test_acc = train_paderborn_4class_improved(
        data_root=str(paderborn_data),
        checkpoint_dir="checkpoints_4class_improved",
        output_dir="outputs_4class_improved",
        num_epochs=60,  # Like literature best practices
        batch_size=64,
        learning_rate=1e-4,
        weight_decay=1e-4,
        dropout=0.5,  # Higher dropout using best practices
        window_size=2048,
        step=512,  # Overlapping windows!
        seed=42
    )

    print(f"\nFinal 4-Class Improved Test Accuracy: {test_acc:.2f}%")
