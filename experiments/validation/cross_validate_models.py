"""Cross-Validation Testing for Paderborn 3-Class and DANN Models."""
import json
import sys
from pathlib import Path
import numpy as np
import torch

# Add project paths
sys.path.append(str(Path(__file__).parent / "Paderborn_Supervised"))
sys.path.append(str(Path(__file__).parent / "DANN_CrossDataset"))


def test_paderborn_3class_with_seeds(seeds=[42, 123, 456, 789, 1011]):
    """Test Paderborn 3-class model with multiple seeds."""
    print("=" * 80)
    print("CROSS-VALIDATION: Paderborn 3-Class Supervised (Non-Overlapping)")
    print("=" * 80)

    from Paderborn_Supervised.dataset_paderborn_3class import Paderborn3ClassDataset, create_loader
    from Paderborn_Supervised.model_paderborn_3class import create_model_3class
    from Paderborn_Supervised.train_paderborn_3class import validate
    import torch.nn as nn

    base_dir = Path(__file__).resolve().parent.parent
    data_root = str(base_dir / "data")

    results = {
        'seeds': seeds,
        'test_acc': [],
        'test_balanced_acc': [],
        'val_acc': [],
        'val_balanced_acc': [],
    }

    for seed in seeds:
        print(f"\n{'─'*80}")
        print(f"Testing with seed={seed}")
        print(f"{'─'*80}")

        # Create datasets
        val_dataset = Paderborn3ClassDataset(
            data_root=data_root,
            split='val',
            seed=seed,
        )

        test_dataset = Paderborn3ClassDataset(
            data_root=data_root,
            split='test',
            seed=seed,
        )

        val_loader = create_loader(val_dataset, batch_size=32, shuffle=False)
        test_loader = create_loader(test_dataset, batch_size=32, shuffle=False)

        # Create model
        sample_batch = next(iter(val_loader))
        model = create_model_3class(
            sample_batch=sample_batch,
            hidden_dim=64,
            fusion_dim=128,
            num_classes=3,
            dropout=0.3
        )

        # Load best checkpoint (trained with seed=42)
        checkpoint_path = Path(__file__).parent / "Paderborn_Supervised" / "checkpoints_3class" / "best_model.pth"
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            print(f"WARNING: Warning: Checkpoint not found at {checkpoint_path}")
            continue

        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()

        # Validate
        val_loss, val_acc, val_balanced_acc, _, _ = validate(model, val_loader, criterion, device)
        test_loss, test_acc, test_balanced_acc, _, _ = validate(model, test_loader, criterion, device)

        print(f"Val  - Acc: {val_acc:.2f}%, Balanced Acc: {val_balanced_acc:.2f}%")
        print(f"Test - Acc: {test_acc:.2f}%, Balanced Acc: {test_balanced_acc:.2f}%")

        results['val_acc'].append(val_acc)
        results['val_balanced_acc'].append(val_balanced_acc)
        results['test_acc'].append(test_acc)
        results['test_balanced_acc'].append(test_balanced_acc)

    # Compute statistics
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}")
    print(f"Validation Accuracy:         {np.mean(results['val_acc']):.2f}% ± {np.std(results['val_acc']):.2f}%")
    print(f"Validation Balanced Acc:     {np.mean(results['val_balanced_acc']):.2f}% ± {np.std(results['val_balanced_acc']):.2f}%")
    print(f"Test Accuracy:               {np.mean(results['test_acc']):.2f}% ± {np.std(results['test_acc']):.2f}%")
    print(f"Test Balanced Accuracy:      {np.mean(results['test_balanced_acc']):.2f}% ± {np.std(results['test_balanced_acc']):.2f}%")
    print(f"{'='*80}")

    # Save results
    output_dir = Path(__file__).parent / "Paderborn_Supervised" / "outputs_3class"
    with open(output_dir / "cross_validation_results.json", 'w') as f:
        json.dump({
            'seeds': results['seeds'],
            'val_acc': results['val_acc'],
            'val_balanced_acc': results['val_balanced_acc'],
            'test_acc': results['test_acc'],
            'test_balanced_acc': results['test_balanced_acc'],
            'summary': {
                'val_acc_mean': float(np.mean(results['val_acc'])),
                'val_acc_std': float(np.std(results['val_acc'])),
                'val_balanced_acc_mean': float(np.mean(results['val_balanced_acc'])),
                'val_balanced_acc_std': float(np.std(results['val_balanced_acc'])),
                'test_acc_mean': float(np.mean(results['test_acc'])),
                'test_acc_std': float(np.std(results['test_acc'])),
                'test_balanced_acc_mean': float(np.mean(results['test_balanced_acc'])),
                'test_balanced_acc_std': float(np.std(results['test_balanced_acc'])),
            }
        }, f, indent=2)

    print(f"\nOK Results saved to {output_dir / 'cross_validation_results.json'}")

    return results


def test_dann_with_seeds(seeds=[42, 123, 456, 789, 1011]):
    """Test DANN model with multiple seeds."""
    print("\n" + "=" * 80)
    print("CROSS-VALIDATION: DANN Cross-Dataset (CWRU→Paderborn)")
    print("=" * 80)

    from DANN_CrossDataset.dataset_dann import create_dann_dataloaders
    from DANN_CrossDataset.model_dann import DANNModel
    from DANN_CrossDataset.train_dann import test_dann

    base_dir = Path(__file__).resolve().parent.parent
    cwru_data = str(base_dir / "CWRU_Dataset")
    paderborn_data = str(base_dir / "data")

    results = {
        'seeds': seeds,
        'val_target_acc': [],
        'test_target_acc': [],
        'test_source_acc': [],
    }

    for seed in seeds:
        print(f"\n{'─'*80}")
        print(f"Testing with seed={seed}")
        print(f"{'─'*80}")

        # Create dataloaders
        loaders = create_dann_dataloaders(
            cwru_root=cwru_data,
            paderborn_root=paderborn_data,
            batch_size=16,
            num_labeled_source=44,
            num_unlabeled_target=64,
            seed=seed
        )

        # Get sample batch to determine input dims
        sample_batch = next(iter(loaders['train_source']))DANNModel(
            envelope_dim=sample_batch['envelope'].shape[1],
            fft_dims=[
                sample_batch['fft_scale1'].shape[1],
                sample_batch['fft_scale2'].shape[1],
                sample_batch['fft_scale3'].shape[1],
            ],
            stats_dim=sample_batch['stats'].shape[1],
            hidden_dim=64,
            fusion_dim=128,
            num_classes=3,
            dropout=0.3
        )

        # Load best checkpoint (trained with seed=42)
        checkpoint_path = Path(__file__).parent / "DANN_CrossDataset" / "checkpoints_dann" / "best_model.pth"
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            print(f"WARNING: Warning: Checkpoint not found at {checkpoint_path}")
            continue

        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        model = model.to(device)

        # Test
        test_results = test_dann(model, loaders, device)

        print(f"Val Target Acc:  {test_results['val_target_acc']:.2f}%")
        print(f"Test Target Acc: {test_results['test_target_acc']:.2f}%")
        print(f"Test Source Acc: {test_results['test_source_acc']:.2f}%")

        results['val_target_acc'].append(test_results['val_target_acc'])
        results['test_target_acc'].append(test_results['test_target_acc'])
        results['test_source_acc'].append(test_results['test_source_acc'])

    # Compute statistics
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}")
    print(f"Val Target Accuracy:         {np.mean(results['val_target_acc']):.2f}% ± {np.std(results['val_target_acc']):.2f}%")
    print(f"Test Target Accuracy:        {np.mean(results['test_target_acc']):.2f}% ± {np.std(results['test_target_acc']):.2f}%")
    print(f"Test Source Accuracy:        {np.mean(results['test_source_acc']):.2f}% ± {np.std(results['test_source_acc']):.2f}%")
    print(f"{'='*80}")

    # Save results
    output_dir = Path(__file__).parent / "DANN_CrossDataset" / "outputs_dann"
    with open(output_dir / "cross_validation_results.json", 'w') as f:
        json.dump({
            'seeds': results['seeds'],
            'val_target_acc': results['val_target_acc'],
            'test_target_acc': results['test_target_acc'],
            'test_source_acc': results['test_source_acc'],
            'summary': {
                'val_target_acc_mean': float(np.mean(results['val_target_acc'])),
                'val_target_acc_std': float(np.std(results['val_target_acc'])),
                'test_target_acc_mean': float(np.mean(results['test_target_acc'])),
                'test_target_acc_std': float(np.std(results['test_target_acc'])),
                'test_source_acc_mean': float(np.mean(results['test_source_acc'])),
                'test_source_acc_std': float(np.std(results['test_source_acc'])),
            }
        }, f, indent=2)

    print(f"\nOK Results saved to {output_dir / 'cross_validation_results.json'}")

    return results


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("CROSS-VALIDATION TESTING (Option 2: Multiple Seeds)")
    print("=" * 80)
    print("\nTesting existing best models with 5 different random seeds")
    print("to verify result stability and report mean ± std.")
    print("\nNote: Multi-seed approach provides robust validation of model stability.")
    print("=" * 80)

    seeds = [42, 123, 456, 789, 1011]

    # Test Paderborn 3-class
    paderborn_results = test_paderborn_3class_with_seeds(seeds)

    # Test DANN
    dann_results = test_dann_with_seeds(seeds)

    print("\n" + "=" * 80)
    print("CROSS-VALIDATION COMPLETE")
    print("=" * 80)
    print("\nAll results saved to respective outputs directories.")
    print("Check cross_validation_results.json files for detailed statistics.")
