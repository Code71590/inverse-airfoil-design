"""
Training Script for the Airfoil Surrogate Model
=================================================
Loads preprocessed data, trains the neural network surrogate,
and saves the trained model checkpoint.
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from surrogate_model import AirfoilSurrogate, AirfoilDataset, DataNormalizer, compute_r2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "processed_data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")

# Feature and target column names
N_CST = 6
FEATURE_COLS = (
    [f'cst_upper_{i}' for i in range(N_CST)] +
    [f'cst_lower_{i}' for i in range(N_CST)] +
    ['mach', 'alpha']
)
TARGET_COLS = ['CL', 'CD', 'CM']


def load_data():
    """Load preprocessed train/val/test datasets."""
    dfs = {}
    for split in ['train', 'val', 'test']:
        path = os.path.join(DATA_DIR, f"{split}.csv")
        if os.path.exists(path):
            dfs[split] = pd.read_csv(path)
            print(f"  Loaded {split}: {len(dfs[split])} samples")
        else:
            print(f"  ⚠ Missing {split}.csv")
    return dfs


def train_model(epochs=100, batch_size=512, lr=1e-3, patience=15, quick_test=False):
    """Train the surrogate model."""
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("  SURROGATE MODEL TRAINING")
    print("=" * 60)
    
    # ── Load data ──
    print("\n📂 Loading data...")
    dfs = load_data()
    
    if 'train' not in dfs:
        print("✗ No training data found. Run preprocess_data.py first.")
        return
    
    # Extract features and targets
    X_train = dfs['train'][FEATURE_COLS].values.astype(np.float32)
    y_train = dfs['train'][TARGET_COLS].values.astype(np.float32)
    
    X_val = dfs.get('val', dfs['train'])[FEATURE_COLS].values.astype(np.float32)
    y_val = dfs.get('val', dfs['train'])[TARGET_COLS].values.astype(np.float32)
    
    if quick_test:
        # Use a small subset for testing
        n = min(5000, len(X_train))
        X_train, y_train = X_train[:n], y_train[:n]
        X_val, y_val = X_val[:min(1000, len(X_val))], y_val[:min(1000, len(y_val))]
        epochs = min(epochs, 10)
        print(f"  ⚡ Quick test mode: {n} train, {len(X_val)} val samples")
    
    # ── Normalize ──
    print("\n📊 Normalizing data...")
    normalizer = DataNormalizer()
    normalizer.fit(X_train, y_train)
    
    X_train_norm = normalizer.transform_features(X_train)
    y_train_norm = normalizer.transform_targets(y_train)
    X_val_norm = normalizer.transform_features(X_val)
    y_val_norm = normalizer.transform_targets(y_val)
    
    # ── Create dataloaders ──
    train_dataset = AirfoilDataset(X_train_norm, y_train_norm)
    val_dataset = AirfoilDataset(X_val_norm, y_val_norm)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size * 2, shuffle=False,
                            num_workers=0)
    
    # ── Init model ──
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🔧 Device: {device}")
    
    model = AirfoilSurrogate(input_dim=len(FEATURE_COLS), output_dim=len(TARGET_COLS))
    model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Model parameters: {total_params:,}")
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
    )
    
    # ── Training loop ──
    print(f"\n🚀 Training for {epochs} epochs...")
    print(f"   Batch size: {batch_size}, LR: {lr}, Patience: {patience}")
    print("-" * 60)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    early_stop_counter = 0
    
    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        epoch_loss = 0
        n_batches = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        avg_train_loss = epoch_loss / n_batches
        train_losses.append(avg_train_loss)
        
        # Validate
        model.eval()
        val_loss = 0
        n_val_batches = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                pred = model(X_batch)
                val_loss += criterion(pred, y_batch).item()
                n_val_batches += 1
                all_preds.append(pred.cpu().numpy())
                all_targets.append(y_batch.cpu().numpy())
        
        avg_val_loss = val_loss / n_val_batches
        val_losses.append(avg_val_loss)
        
        # R² on validation
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        
        # Denormalize for R²
        preds_real = normalizer.inverse_transform_targets(all_preds)
        targets_real = normalizer.inverse_transform_targets(all_targets)
        r2_scores = compute_r2(targets_real, preds_real)
        
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        if epoch % max(1, epochs // 20) == 0 or epoch == 1:
            print(f"  Epoch {epoch:4d}/{epochs} | "
                  f"Train: {avg_train_loss:.6f} | "
                  f"Val: {avg_val_loss:.6f} | "
                  f"R²: CL={r2_scores[0]:.4f} CD={r2_scores[1]:.4f} CM={r2_scores[2]:.4f} | "
                  f"LR: {current_lr:.2e}")
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_counter = 0
            # Save best model
            checkpoint = {
                'model_state': model.state_dict(),
                'normalizer_state': normalizer.state_dict(),
                'epoch': epoch,
                'val_loss': best_val_loss,
                'r2_scores': r2_scores,
                'feature_cols': FEATURE_COLS,
                'target_cols': TARGET_COLS
            }
            torch.save(checkpoint, os.path.join(MODELS_DIR, 'best_surrogate.pth'))
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"\n  ⏹ Early stopping at epoch {epoch}")
                break
    
    print("-" * 60)
    print(f"  ✓ Best validation loss: {best_val_loss:.6f}")
    print(f"  ✓ Final R²: CL={r2_scores[0]:.4f}  CD={r2_scores[1]:.4f}  CM={r2_scores[2]:.4f}")
    
    # ── Plot training curves ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(train_losses, label='Train Loss', color='#4F46E5')
    axes[0].plot(val_losses, label='Val Loss', color='#EF4444')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('MSE Loss')
    axes[0].set_title('Training & Validation Loss')
    axes[0].legend()
    axes[0].set_yscale('log')
    axes[0].grid(True, alpha=0.3)
    
    # Parity plots
    labels = ['CL', 'CD', 'CM']
    colors = ['#4F46E5', '#10B981', '#F59E0B']
    for i, (lbl, col) in enumerate(zip(labels, colors)):
        axes[1].scatter(targets_real[:, i], preds_real[:, i],
                       alpha=0.1, s=2, color=col, label=f'{lbl} (R²={r2_scores[i]:.3f})')
    
    lims = [min(targets_real.min(), preds_real.min()),
            max(targets_real.max(), preds_real.max())]
    axes[1].plot(lims, lims, 'k--', alpha=0.5, linewidth=1)
    axes[1].set_xlabel('True')
    axes[1].set_ylabel('Predicted')
    axes[1].set_title('Parity Plot (Validation)')
    axes[1].legend(markerscale=5)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'training_results.png'), dpi=150)
    plt.close()
    print(f"  ✓ Plots saved to {PLOTS_DIR}/training_results.png")
    
    # ── Evaluate on test set ──
    if 'test' in dfs:
        print("\n📋 Test Set Evaluation:")
        X_test = dfs['test'][FEATURE_COLS].values.astype(np.float32)
        y_test = dfs['test'][TARGET_COLS].values.astype(np.float32)
        
        X_test_norm = normalizer.transform_features(X_test)
        test_dataset = AirfoilDataset(X_test_norm, np.zeros_like(y_test))
        test_loader = DataLoader(test_dataset, batch_size=batch_size * 2)
        
        model.eval()
        test_preds = []
        with torch.no_grad():
            for X_batch, _ in test_loader:
                X_batch = X_batch.to(device)
                pred = model(X_batch)
                test_preds.append(pred.cpu().numpy())
        
        test_preds = np.concatenate(test_preds)
        test_preds_real = normalizer.inverse_transform_targets(test_preds)
        test_r2 = compute_r2(y_test, test_preds_real)
        
        test_mse = np.mean((y_test - test_preds_real) ** 2, axis=0)
        
        print(f"   R²:  CL={test_r2[0]:.4f}  CD={test_r2[1]:.4f}  CM={test_r2[2]:.4f}")
        print(f"   MSE: CL={test_mse[0]:.6f}  CD={test_mse[1]:.8f}  CM={test_mse[2]:.6f}")
    
    print("\n✓ Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train airfoil surrogate model')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--quick-test', action='store_true',
                        help='Quick test with small subset')
    
    args = parser.parse_args()
    train_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        quick_test=args.quick_test
    )
