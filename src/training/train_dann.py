import json
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_dann import create_dann_dataloaders
from model_dann import create_dann_model


def compute_alpha(epoch, max_epochs):
    """Compute alpha for gradient reversal layer based on epoch."""
    p = epoch / max_epochs
    return 2.0 / (1.0 + np.exp(-10 * p)) - 1.0


def train_epoch(model, loader, criterion_class, criterion_domain, optimizer, device, epoch, max_epochs):
    """Train for one epoch with DANN."""
    model.train()

    total_class_loss = 0
    total_domain_loss = 0
    total_loss = 0

    correct_class = 0
    total_labeled = 0
    correct_domain = 0
    total_samples = 0

    # Compute alpha for GRL
    alpha = compute_alpha(epoch, max_epochs)

    pbar = tqdm(loader, desc=f"Training (α={alpha:.3f})")
    for batch in pbar:
        # Move to device
        envelope = batch['envelope'].to(device)
        fft1 = batch['fft_scale1'].to(device)
        fft2 = batch['fft_scale2'].to(device)
        fft3 = batch['fft_scale3'].to(device)
        stats = batch['stats'].to(device)
        labels = batch['label'].to(device)
        domains = batch['domain'].to(device)
        has_label = batch['has_label']

        batch_size = envelope.size(0)

        # Forward pass
        optimizer.zero_grad()
        class_logits, domain_logits = model(envelope, fft1, fft2, fft3, stats, alpha=alpha)

        # Classification loss (only labeled samples)
        labeled_mask = (labels != -1)
        if labeled_mask.sum() > 0:
            class_loss = criterion_class(class_logits[labeled_mask], labels[labeled_mask])
        else:
            class_loss = torch.tensor(0.0, device=device)

        # Domain loss (all samples)
        domain_loss = criterion_domain(domain_logits, domains)

        # Total loss
        loss = class_loss + domain_loss

        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Metrics
        total_class_loss += class_loss.item()
        total_domain_loss += domain_loss.item()
        total_loss += loss.item()

        # Classification accuracy (labeled samples only)
        if labeled_mask.sum() > 0:
            _, predicted = class_logits[labeled_mask].max(1)
            total_labeled += labeled_mask.sum().item()
            correct_class += predicted.eq(labels[labeled_mask]).sum().item()

        # Domain accuracy
        _, predicted_domain = domain_logits.max(1)
        total_samples += batch_size
        correct_domain += predicted_domain.eq(domains).sum().item()

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.3f}',
            'cls': f'{class_loss.item():.3f}',
            'dom': f'{domain_loss.item():.3f}',
            'acc': f'{100.*correct_class/total_labeled:.1f}%' if total_labeled > 0 else 'N/A'
        })

    avg_class_loss = total_class_loss / len(loader)
    avg_domain_loss = total_domain_loss / len(loader)
    avg_loss = total_loss / len(loader)
    class_accuracy = 100. * correct_class / total_labeled if total_labeled > 0 else 0
    domain_accuracy = 100. * correct_domain / total_samples

    return avg_loss, avg_class_loss, avg_domain_loss, class_accuracy, domain_accuracy


def validate(model, loader, criterion_class, criterion_domain, device):
    """Validate model."""
    model.eval()

    total_class_loss = 0
    total_domain_loss = 0

    correct_class = 0
    total_labeled = 0
    correct_domain = 0
    total_samples = 0

    # Track per-domain accuracy
    correct_by_domain = {0: 0, 1: 0}
    total_by_domain = {0: 0, 1: 0}

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating"):
            envelope = batch['envelope'].to(device)
            fft1 = batch['fft_scale1'].to(device)
            fft2 = batch['fft_scale2'].to(device)
            fft3 = batch['fft_scale3'].to(device)
            stats = batch['stats'].to(device)
            labels = batch['label'].to(device)
            domains = batch['domain'].to(device)

            # Forward (use alpha=1.0 for validation)
            class_logits, domain_logits = model(envelope, fft1, fft2, fft3, stats, alpha=1.0)

            # Classification loss (only labeled)
            labeled_mask = (labels != -1)
            if labeled_mask.sum() > 0:
                class_loss = criterion_class(class_logits[labeled_mask], labels[labeled_mask])
                total_class_loss += class_loss.item()

            # Domain loss
            domain_loss = criterion_domain(domain_logits, domains)
            total_domain_loss += domain_loss.item()

            # Classification accuracy (labeled only)
            if labeled_mask.sum() > 0:
                _, predicted = class_logits[labeled_mask].max(1)
                total_labeled += labeled_mask.sum().item()
                correct_class += predicted.eq(labels[labeled_mask]).sum().item()

                # Per-domain classification accuracy
                for domain_id in [0, 1]:
                    domain_mask = labeled_mask & (domains == domain_id)
                    if domain_mask.sum() > 0:
                        total_by_domain[domain_id] += domain_mask.sum().item()
                        correct_by_domain[domain_id] += predicted.eq(labels[labeled_mask])[domains[labeled_mask] == domain_id].sum().item()

            # Domain accuracy
            _, predicted_domain = domain_logits.max(1)
            total_samples += domains.size(0)
            correct_domain += predicted_domain.eq(domains).sum().item()

    avg_class_loss = total_class_loss / len(loader) if len(loader) > 0 else 0
    avg_domain_loss = total_domain_loss / len(loader) if len(loader) > 0 else 0
    class_accuracy = 100. * correct_class / total_labeled if total_labeled > 0 else 0
    domain_accuracy = 100. * correct_domain / total_samples if total_samples > 0 else 0

    # Per-domain accuracies
    source_acc = 100. * correct_by_domain[0] / total_by_domain[0] if total_by_domain[0] > 0 else 0
    target_acc = 100. * correct_by_domain[1] / total_by_domain[1] if total_by_domain[1] > 0 else 0

    return avg_class_loss, avg_domain_loss, class_accuracy, domain_accuracy, source_acc, target_acc


def train_dann(
    cwru_root: str = "data/CWRU",
    paderborn_root: str = "data/Paderborn",
    checkpoint_dir: str = "checkpoints_dann",
    output_dir: str = "outputs_dann",
    num_epochs: int = 100,
    batch_size: int = 8,
    learning_rate: float = 0.0005,
    weight_decay: float = 5e-5,
    dropout: float = 0.3,
    labeled_ratio: float = 0.1,
    patience: int = 20,
    seed: int = 42,
):
    """Train DANN model."""

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
    print("DANN TRAINING - Cross-Dataset Domain Adaptation")
    print("="*70)

    print("\nLoading datasets...")
    train_loader, val_loader, test_loader = create_dann_dataloaders(
        cwru_root=cwru_root,
        paderborn_root=paderborn_root,
        batch_size=batch_size,
        labeled_ratio=labeled_ratio,
        seed=seed,
    )

    print("\nCreating DANN model...")
    sample_batch = next(iter(train_loader))
    model = create_dann_model(
        sample_batch=sample_batch,
        hidden_dim=64,
        fusion_dim=128,
        num_classes=3,  # 3 classes: Healthy, Inner, Outer
        dropout=dropout
    ).to(device)

    # Loss functions
    criterion_class = nn.CrossEntropyLoss()
    criterion_domain = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)

    # Training history
    history = {
        'train_loss': [],
        'train_class_loss': [],
        'train_domain_loss': [],
        'train_class_acc': [],
        'train_domain_acc': [],
        'val_class_loss': [],
        'val_domain_loss': [],
        'val_class_acc': [],
        'val_domain_acc': [],
        'val_source_acc': [],
        'val_target_acc': [],
    }

    best_target_acc = 0
    patience_counter = 0

    print("\nStarting training...")
    print("="*70)

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 70)

        # Train
        train_loss, train_class_loss, train_domain_loss, train_class_acc, train_domain_acc = train_epoch(
            model, train_loader, criterion_class, criterion_domain, optimizer, device, epoch, num_epochs
        )

        print(f"Train - Loss: {train_loss:.4f} (Cls: {train_class_loss:.4f}, Dom: {train_domain_loss:.4f})")
        print(f"        Class Acc: {train_class_acc:.2f}%, Domain Acc: {train_domain_acc:.2f}%")

        # Validate
        val_class_loss, val_domain_loss, val_class_acc, val_domain_acc, val_source_acc, val_target_acc = validate(
            model, val_loader, criterion_class, criterion_domain, device
        )

        print(f"Val   - Class Loss: {val_class_loss:.4f}, Domain Loss: {val_domain_loss:.4f}")
        print(f"        Overall Acc: {val_class_acc:.2f}%")
        print(f"        Source (CWRU) Acc: {val_source_acc:.2f}%")
        print(f"        Target (Paderborn) Acc: {val_target_acc:.2f}%")

        # Update history
        history['train_loss'].append(train_loss)
        history['train_class_loss'].append(train_class_loss)
        history['train_domain_loss'].append(train_domain_loss)
        history['train_class_acc'].append(train_class_acc)
        history['train_domain_acc'].append(train_domain_acc)
        history['val_class_loss'].append(val_class_loss)
        history['val_domain_loss'].append(val_domain_loss)
        history['val_class_acc'].append(val_class_acc)
        history['val_domain_acc'].append(val_domain_acc)
        history['val_source_acc'].append(val_source_acc)
        history['val_target_acc'].append(val_target_acc)

        # Learning rate schedule
        scheduler.step(val_target_acc)

        # Save best model (based on target domain accuracy)
        if val_target_acc > best_target_acc:
            best_target_acc = val_target_acc
            patience_counter = 0

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_target_acc': val_target_acc,
            }, f"{checkpoint_dir}/best_model.pth")

            print(f"OK New best target accuracy: {val_target_acc:.2f}%")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered after {patience} epochs without improvement")
            break

    # Load best model for testing
    print("\n" + "="*70)
    print("Loading best model for testing...")
    checkpoint = torch.load(f"{checkpoint_dir}/best_model.pth")
    model.load_state_dict(checkpoint['model_state_dict'])

    # Test
    test_class_loss, test_domain_loss, test_class_acc, test_domain_acc, test_source_acc, test_target_acc = validate(
        model, test_loader, criterion_class, criterion_domain, device
    )

    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"Test Overall Accuracy:         {test_class_acc:.2f}%")
    print(f"Test Source (CWRU) Accuracy:   {test_source_acc:.2f}%")
    print(f"Test Target (Paderborn) Accuracy: {test_target_acc:.2f}%")
    print(f"Test Domain Accuracy:          {test_domain_acc:.2f}%")
    print("="*70)

    # Save results
    history['best_target_acc'] = best_target_acc
    history['test_class_acc'] = test_class_acc
    history['test_source_acc'] = test_source_acc
    history['test_target_acc'] = test_target_acc
    history['test_domain_acc'] = test_domain_acc

    with open(f"{output_dir}/dann_history.json", 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\nOK Results saved to {output_dir}/dann_history.json")
    print("="*70)

    return test_target_acc


if __name__ == "__main__":
    import sys
    sys.path.append(str(Path(__file__).parent))

    # Use absolute paths
    base_dir = Path(__file__).resolve().parent.parent.parent
    cwru_data = base_dir / "CWRU_Project" / "data"
    paderborn_data = base_dir / "data"

    test_target_acc = train_dann(
        cwru_root=str(cwru_data),
        paderborn_root=str(paderborn_data),
        checkpoint_dir="checkpoints_dann",
        output_dir="outputs_dann",
        num_epochs=100,
        batch_size=8,
        learning_rate=0.0005,
        weight_decay=5e-5,
        dropout=0.3,
        labeled_ratio=0.1,  # 10% of Paderborn labeled
        patience=20,
        seed=42
    )

    print(f"\nFinal Paderborn Accuracy: {test_target_acc:.2f}%")
