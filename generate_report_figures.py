"""
Validation Results & Report Figure Generator
==============================================
Generates publication-quality plots and metrics tables
for the surrogate-based inverse airfoil design project report.

Outputs are saved to: plots/report/
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
import torch
from surrogate_model import AirfoilSurrogate, DataNormalizer, compute_r2
from cst_module import cst_from_vector, reconstruct_airfoil

# ── Paths ──────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "processed_data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORT_DIR = os.path.join(BASE_DIR, "plots", "report")

N_CST = 6
FEATURE_COLS = (
    [f'cst_upper_{i}' for i in range(N_CST)] +
    [f'cst_lower_{i}' for i in range(N_CST)] +
    ['mach', 'alpha']
)
TARGET_COLS = ['CL', 'CD', 'CM']

# ── Style Configuration ───────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 200,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

COLORS = {
    'CL': '#4361EE',
    'CD': '#2EC4B6',
    'CM': '#FF9F1C',
    'accent': '#E71D36',
    'gray': '#6c757d'
}


def load_model_and_data():
    """Load trained model and all dataset splits."""
    print("📂 Loading model and data...")
    
    # Load checkpoint
    ckpt_path = os.path.join(MODELS_DIR, "best_surrogate.pth")
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    
    model = AirfoilSurrogate(
        input_dim=len(checkpoint['feature_cols']),
        output_dim=len(checkpoint['target_cols'])
    )
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    
    normalizer = DataNormalizer()
    normalizer.load_state_dict(checkpoint['normalizer_state'])
    
    # Load data splits
    data = {}
    for split in ['train', 'val', 'test']:
        path = os.path.join(DATA_DIR, f"{split}.csv")
        df = pd.read_csv(path)
        # Subsample train for speed (plots don't need 1.5M points)
        if split == 'train' and len(df) > 100000:
            df = df.sample(n=100000, random_state=42)
        data[split] = df
        print(f"   {split}: {len(df)} samples")
    
    info = {
        'epoch': checkpoint.get('epoch', '?'),
        'val_loss': checkpoint.get('val_loss', 0),
        'r2_train': checkpoint.get('r2_scores', [0, 0, 0])
    }
    
    return model, normalizer, data, info


def predict_split(model, normalizer, df):
    """Run model predictions on a dataframe split."""
    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df[TARGET_COLS].values.astype(np.float32)
    
    X_norm = normalizer.transform_features(X)
    
    with torch.no_grad():
        pred_norm = model(torch.FloatTensor(X_norm))
    
    pred = normalizer.inverse_transform_targets(pred_norm.numpy())
    return y, pred


# ═══════════════ FIGURE 1: PARITY PLOTS ═══════════════
def fig_parity_plots(model, normalizer, data):
    """Predicted vs Actual scatter plots for all three coefficients."""
    print("\n📊 Figure 1: Parity Plots...")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    
    for split, marker_alpha, label in [('test', 0.08, 'Test'), ('val', 0.05, 'Val')]:
        y_true, y_pred = predict_split(model, normalizer, data[split])
        r2 = compute_r2(y_true, y_pred)
        
        for i, (ax, name) in enumerate(zip(axes, TARGET_COLS)):
            ax.scatter(y_true[:, i], y_pred[:, i],
                      alpha=marker_alpha, s=3, color=COLORS[name],
                      label=f'{label} (R²={r2[i]:.4f})' if split == 'test' else None,
                      rasterized=True)
    
    for i, (ax, name) in enumerate(zip(axes, TARGET_COLS)):
        y_true, y_pred = predict_split(model, normalizer, data['test'])
        lims = [min(y_true[:, i].min(), y_pred[:, i].min()),
                max(y_true[:, i].max(), y_pred[:, i].max())]
        margin = (lims[1] - lims[0]) * 0.05
        ax.plot([lims[0]-margin, lims[1]+margin], [lims[0]-margin, lims[1]+margin],
                'k--', alpha=0.6, linewidth=1, label='Ideal')
        ax.set_xlabel(f'True {name}')
        ax.set_ylabel(f'Predicted {name}')
        ax.set_title(f'{name} Prediction Accuracy')
        r2 = compute_r2(y_true[:, i:i+1], y_pred[:, i:i+1])
        ax.legend(loc='upper left', framealpha=0.9)
        ax.set_xlim(lims[0]-margin, lims[1]+margin)
        ax.set_ylim(lims[0]-margin, lims[1]+margin)
        ax.set_aspect('equal')
    
    fig.suptitle('Surrogate Model: Predicted vs Actual (Test Set)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, '01_parity_plots.png'))
    plt.close()
    print("   ✓ Saved 01_parity_plots.png")


# ═══════════════ FIGURE 2: ERROR DISTRIBUTIONS ═══════════════
def fig_error_distributions(model, normalizer, data):
    """Histograms of prediction errors for each coefficient."""
    print("\n📊 Figure 2: Error Distributions...")
    
    y_true, y_pred = predict_split(model, normalizer, data['test'])
    errors = y_true - y_pred
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    
    for i, (ax, name) in enumerate(zip(axes, TARGET_COLS)):
        err = errors[:, i]
        mae = np.mean(np.abs(err))
        rmse = np.sqrt(np.mean(err**2))
        
        ax.hist(err, bins=100, color=COLORS[name], alpha=0.7, edgecolor='white',
                linewidth=0.5, density=True)
        ax.axvline(0, color='k', linewidth=1, linestyle='--', alpha=0.5)
        ax.axvline(np.mean(err), color=COLORS['accent'], linewidth=1.5,
                  linestyle='-', label=f'Mean={np.mean(err):.4f}')
        
        # Add ±1σ bands
        std = np.std(err)
        ax.axvline(std, color=COLORS['gray'], linewidth=1, linestyle=':', alpha=0.7)
        ax.axvline(-std, color=COLORS['gray'], linewidth=1, linestyle=':', alpha=0.7,
                  label=f'±1σ = ±{std:.4f}')
        
        ax.set_xlabel(f'{name} Error (True − Predicted)')
        ax.set_ylabel('Density')
        ax.set_title(f'{name} Error Distribution\nMAE={mae:.4f}, RMSE={rmse:.4f}')
        ax.legend(fontsize=9)
    
    fig.suptitle('Prediction Error Distributions (Test Set)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, '02_error_distributions.png'))
    plt.close()
    print("   ✓ Saved 02_error_distributions.png")


# ═══════════════ FIGURE 3: ERROR VS MACH ═══════════════
def fig_error_vs_mach(model, normalizer, data):
    """Box plots of error across different Mach numbers."""
    print("\n📊 Figure 3: Error vs Mach Number...")
    
    df = data['test'].copy()
    y_true, y_pred = predict_split(model, normalizer, df)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    
    mach_values = sorted(df['mach'].unique())
    
    for i, (ax, name) in enumerate(zip(axes, TARGET_COLS)):
        errors_by_mach = []
        labels = []
        for m in mach_values:
            mask = df['mach'].values == m
            err = np.abs(y_true[mask, i] - y_pred[mask, i])
            errors_by_mach.append(err)
            labels.append(f'{m:.1f}')
        
        bp = ax.boxplot(errors_by_mach, labels=labels, patch_artist=True,
                       showfliers=False, widths=0.6)
        for patch in bp['boxes']:
            patch.set_facecolor(COLORS[name])
            patch.set_alpha(0.6)
        for median in bp['medians']:
            median.set_color('black')
            median.set_linewidth(1.5)
        
        ax.set_xlabel('Mach Number')
        ax.set_ylabel(f'|{name} Error|')
        ax.set_title(f'{name} Absolute Error by Mach')
    
    fig.suptitle('Model Error Across Mach Numbers (Test Set)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, '03_error_vs_mach.png'))
    plt.close()
    print("   ✓ Saved 03_error_vs_mach.png")


# ═══════════════ FIGURE 4: RESIDUAL PLOTS ═══════════════
def fig_residuals(model, normalizer, data):
    """Residuals vs predicted value — checks for systematic bias."""
    print("\n📊 Figure 4: Residual Plots...")
    
    y_true, y_pred = predict_split(model, normalizer, data['test'])
    errors = y_true - y_pred
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    
    for i, (ax, name) in enumerate(zip(axes, TARGET_COLS)):
        ax.scatter(y_pred[:, i], errors[:, i], alpha=0.05, s=2,
                  color=COLORS[name], rasterized=True)
        ax.axhline(0, color='k', linewidth=1, linestyle='--', alpha=0.5)
        
        # Running mean
        sorted_idx = np.argsort(y_pred[:, i])
        window = max(len(sorted_idx) // 50, 100)
        pred_sorted = y_pred[sorted_idx, i]
        err_sorted = errors[sorted_idx, i]
        running_mean = np.convolve(err_sorted, np.ones(window)/window, mode='valid')
        pred_running = pred_sorted[window//2:window//2+len(running_mean)]
        ax.plot(pred_running, running_mean, color=COLORS['accent'],
               linewidth=2, label='Running mean')
        
        ax.set_xlabel(f'Predicted {name}')
        ax.set_ylabel(f'Residual (True − Pred)')
        ax.set_title(f'{name} Residuals')
        ax.legend()
    
    fig.suptitle('Residual Analysis (Test Set)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, '04_residual_plots.png'))
    plt.close()
    print("   ✓ Saved 04_residual_plots.png")


# ═══════════════ FIGURE 5: ERROR VS ALPHA ═══════════════
def fig_error_vs_alpha(model, normalizer, data):
    """Error variation with angle of attack."""
    print("\n📊 Figure 5: Error vs Alpha...")
    
    df = data['test'].copy()
    y_true, y_pred = predict_split(model, normalizer, df)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    
    alphas = df['alpha'].values
    alpha_bins = np.arange(alphas.min(), alphas.max() + 1, 1.0)
    
    for i, (ax, name) in enumerate(zip(axes, TARGET_COLS)):
        abs_err = np.abs(y_true[:, i] - y_pred[:, i])
        
        bin_means = []
        bin_stds = []
        bin_centers = []
        for j in range(len(alpha_bins) - 1):
            mask = (alphas >= alpha_bins[j]) & (alphas < alpha_bins[j + 1])
            if mask.sum() > 10:
                bin_means.append(np.mean(abs_err[mask]))
                bin_stds.append(np.std(abs_err[mask]))
                bin_centers.append((alpha_bins[j] + alpha_bins[j + 1]) / 2)
        
        ax.fill_between(bin_centers,
                        np.array(bin_means) - np.array(bin_stds),
                        np.array(bin_means) + np.array(bin_stds),
                        alpha=0.2, color=COLORS[name])
        ax.plot(bin_centers, bin_means, 'o-', color=COLORS[name],
               markersize=4, linewidth=1.5, label=f'Mean |error|')
        
        ax.set_xlabel('Angle of Attack α (°)')
        ax.set_ylabel(f'|{name} Error|')
        ax.set_title(f'{name} Error vs α')
        ax.legend()
    
    fig.suptitle('Error Variation with Angle of Attack (Test Set)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, '05_error_vs_alpha.png'))
    plt.close()
    print("   ✓ Saved 05_error_vs_alpha.png")


# ═══════════════ FIGURE 6: R² PER MACH-ALPHA ═══════════════
def fig_r2_heatmap(model, normalizer, data):
    """R² heatmap across Mach-alpha conditions."""
    print("\n📊 Figure 6: R² Heatmap...")
    
    df = data['test'].copy()
    y_true, y_pred = predict_split(model, normalizer, df)
    
    mach_vals = sorted(df['mach'].unique())
    alpha_bins = np.arange(-5, 16, 2)
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    
    for idx, (ax, name) in enumerate(zip(axes, TARGET_COLS)):
        r2_matrix = np.full((len(mach_vals), len(alpha_bins) - 1), np.nan)
        
        for mi, m in enumerate(mach_vals):
            for ai in range(len(alpha_bins) - 1):
                mask = ((df['mach'].values == m) &
                       (df['alpha'].values >= alpha_bins[ai]) &
                       (df['alpha'].values < alpha_bins[ai + 1]))
                if mask.sum() > 20:
                    r2 = compute_r2(y_true[mask, idx:idx+1], y_pred[mask, idx:idx+1])
                    r2_matrix[mi, ai] = r2[0]
        
        alpha_centers = [(alpha_bins[i] + alpha_bins[i+1]) / 2 for i in range(len(alpha_bins) - 1)]
        
        im = ax.imshow(r2_matrix, aspect='auto', cmap='RdYlGn', vmin=0.5, vmax=1.0,
                      origin='lower')
        ax.set_xticks(range(len(alpha_centers)))
        ax.set_xticklabels([f'{a:.0f}' for a in alpha_centers], fontsize=8)
        ax.set_yticks(range(len(mach_vals)))
        ax.set_yticklabels([f'{m:.1f}' for m in mach_vals])
        ax.set_xlabel('α (°)')
        ax.set_ylabel('Mach')
        ax.set_title(f'{name} R²')
        plt.colorbar(im, ax=ax, shrink=0.8, label='R²')
        
        # Annotate cells
        for mi in range(len(mach_vals)):
            for ai in range(len(alpha_centers)):
                if not np.isnan(r2_matrix[mi, ai]):
                    color = 'white' if r2_matrix[mi, ai] < 0.75 else 'black'
                    ax.text(ai, mi, f'{r2_matrix[mi, ai]:.2f}',
                           ha='center', va='center', fontsize=7, color=color)
    
    fig.suptitle('R² Score Across Operating Conditions (Test Set)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, '06_r2_heatmap.png'))
    plt.close()
    print("   ✓ Saved 06_r2_heatmap.png")


# ═══════════════ FIGURE 7: INVERSE DESIGN EXAMPLES ═══════════════
def fig_inverse_design_examples(model, normalizer):
    """Demonstrate inverse design with example cases."""
    print("\n📊 Figure 7: Inverse Design Examples...")
    
    from inverse_design import InverseDesigner, predict_aero
    designer = InverseDesigner()
    
    test_cases = [
        {'name': 'High Lift', 'targets': {'CL': 1.2, 'CD': 0.02}, 'mach': 0.3, 'alpha': 5.0},
        {'name': 'Low Drag', 'targets': {'CL': 0.5, 'CD': 0.008}, 'mach': 0.3, 'alpha': 2.0},
        {'name': 'Transonic', 'targets': {'CL': 0.6, 'CD': 0.015}, 'mach': 0.6, 'alpha': 3.0},
    ]
    
    fig = plt.figure(figsize=(16, 9))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.55, wspace=0.35,
                           height_ratios=[1.1, 1.0])
    
    for j, case in enumerate(test_cases):
        result = designer.design_evolutionary(
            case['targets'], case['mach'], case['alpha'],
            maxiter=300, popsize=40
        )
        
        if result is None:
            print(f"   ✗ Optimization failed for {case['name']}")
            continue
        
        # Top row: airfoil shapes
        ax_shape = fig.add_subplot(gs[0, j])
        ax_shape.fill(result['x_coords'], result['y_coords'],
                     alpha=0.15, color=COLORS['CL'])
        ax_shape.plot(result['x_coords'], result['y_coords'],
                     linewidth=2, color=COLORS['CL'])
        ax_shape.plot([0, 1], [0, 0], 'k--', alpha=0.3, linewidth=0.5)
        ax_shape.set_xlim(-0.05, 1.05)
        x_c = np.array(result['x_coords'])
        y_c = np.array(result['y_coords'])
        y_ext = max(abs(y_c.min()), abs(y_c.max())) * 1.35 + 0.04
        ax_shape.set_ylim(-y_ext, y_ext)
        ax_shape.set_aspect('equal', adjustable='datalim')
        ax_shape.set_xlabel('x/c', fontsize=9)
        ax_shape.set_ylabel('y/c', fontsize=9)
        ax_shape.tick_params(labelsize=8)
        
        # Title with targets and achieved — add ✓/≈/✗ symbols
        t = case['targets']
        p = result['predicted']
        cl_err = abs(p['CL'] - t['CL']) / max(abs(t['CL']), 1e-6) * 100
        cd_err = abs(p['CD'] - t['CD']) / max(abs(t['CD']), 1e-6) * 100
        cl_sym = '\u2713' if cl_err < 5 else ('\u2248' if cl_err < 20 else '\u2717')
        cd_sym = '\u2713' if cd_err < 5 else ('\u2248' if cd_err < 20 else '\u2717')
        ax_shape.set_title(
            f"{case['name']}  (M={case['mach']}, \u03b1={case['alpha']}\u00b0)\n"
            f"Target:    CL={t['CL']:.3f}  CD={t['CD']:.4f}\n"
            f"Achieved: CL={p['CL']:.3f} {cl_sym}  CD={p['CD']:.4f} {cd_sym}",
            fontsize=9.5, linespacing=1.5
        )
        
        # Bottom row: alpha sweeps
        ax_sweep = fig.add_subplot(gs[1, j])
        cst_vec = np.array(result['cst_upper'] + result['cst_lower'])
        sweep = designer.sweep_alpha(cst_vec, case['mach'])
        alphas = [s['alpha'] for s in sweep]
        cls = [s['CL'] for s in sweep]
        cds = [s['CD'] for s in sweep]
        
        ax_sweep.plot(alphas, cls, '-', color=COLORS['CL'], linewidth=2)
        ax_sweep.set_xlabel('\u03b1 (\u00b0)', fontsize=9)
        ax_sweep.set_ylabel('CL', color=COLORS['CL'], fontsize=9)
        ax_sweep.tick_params(axis='y', labelcolor=COLORS['CL'], labelsize=8)
        ax_sweep.tick_params(axis='x', labelsize=8)
        
        ax2 = ax_sweep.twinx()
        ax2.plot(alphas, cds, '-', color=COLORS['CD'], linewidth=2)
        ax2.set_ylabel('CD', color=COLORS['CD'], fontsize=9)
        ax2.tick_params(axis='y', labelcolor=COLORS['CD'], labelsize=8)
        ax2.spines['right'].set_visible(True)
        
        # Target reference lines (dotted)
        ax_sweep.axhline(t['CL'], color=COLORS['CL'], linestyle=':', alpha=0.7,
                         linewidth=1.4)
        ax2.axhline(t['CD'], color=COLORS['CD'], linestyle=':', alpha=0.7,
                    linewidth=1.4)
        # Design-point alpha marker
        ax_sweep.axvline(case['alpha'], color=COLORS['gray'], linestyle='--',
                         alpha=0.5, linewidth=1)
        ax_sweep.set_title('CL & CD  vs  \u03b1', fontsize=10)
    
    fig.suptitle('Inverse Design Demonstrations  [Evolutionary Optimizer — Robust]',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.savefig(os.path.join(REPORT_DIR, '07_inverse_design_examples.png'),
                bbox_inches='tight')
    plt.close()
    print("   ✓ Saved 07_inverse_design_examples.png")


# ═══════════════ FIGURE 8: COMBINED METRICS TABLE ═══════════════
def fig_metrics_table(model, normalizer, data):
    """Generate a metrics summary table as an image."""
    print("\n📊 Figure 8: Metrics Summary Table...")
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis('off')
    
    # Compute metrics for all splits
    rows = []
    for split in ['train', 'val', 'test']:
        y_true, y_pred = predict_split(model, normalizer, data[split])
        r2 = compute_r2(y_true, y_pred)
        mae = np.mean(np.abs(y_true - y_pred), axis=0)
        rmse = np.sqrt(np.mean((y_true - y_pred)**2, axis=0))
        
        for i, name in enumerate(TARGET_COLS):
            rows.append([
                split.capitalize(), name, len(data[split]),
                f'{r2[i]:.4f}', f'{mae[i]:.5f}', f'{rmse[i]:.5f}'
            ])
    
    col_labels = ['Split', 'Target', 'Samples', 'R²', 'MAE', 'RMSE']
    
    table = ax.table(cellText=rows, colLabels=col_labels,
                    loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)
    
    # Style header
    for j, label in enumerate(col_labels):
        cell = table[0, j]
        cell.set_facecolor('#2d3748')
        cell.set_text_props(color='white', fontweight='bold')
    
    # Alternate row colors
    for i in range(1, len(rows) + 1):
        for j in range(len(col_labels)):
            cell = table[i, j]
            if (i - 1) // 3 == 0:
                cell.set_facecolor('#EBF5FB')
            elif (i - 1) // 3 == 1:
                cell.set_facecolor('#FEF9E7')
            else:
                cell.set_facecolor('#E8F8F5')
    
    ax.set_title('Surrogate Model Performance Metrics', fontsize=14,
                fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, '08_metrics_table.png'))
    plt.close()
    print("   ✓ Saved 08_metrics_table.png")
    
    # Also save as CSV
    metrics_df = pd.DataFrame(rows, columns=col_labels)
    metrics_df.to_csv(os.path.join(REPORT_DIR, 'metrics_summary.csv'), index=False)
    print("   ✓ Saved metrics_summary.csv")


# ═══════════════ FIGURE 9: CST RECONSTRUCTION QUALITY ═══════════════
def fig_cst_reconstruction(data):
    """Show CST reconstruction quality for sample airfoils."""
    print("\n📊 Figure 9: CST Reconstruction Quality...")
    
    from cst_module import read_dat_file, fit_airfoil_cst, reconstruct_airfoil
    
    shape_dir = os.path.join(BASE_DIR, "Shape", "data", "airfoil", "cst_gen")
    
    # Pick a few airfoils from test set
    test_names = data['test']['airfoil_name'].unique()[:6]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    axes = axes.flatten()
    
    for idx, name in enumerate(test_names):
        ax = axes[idx]
        dat_path = os.path.join(shape_dir, f"{name}.dat")
        
        if not os.path.exists(dat_path):
            ax.text(0.5, 0.5, f'{name}\n(file not found)', transform=ax.transAxes,
                   ha='center', va='center')
            continue
        
        x_orig, y_orig = read_dat_file(dat_path)
        params = fit_airfoil_cst(x_orig, y_orig, n_weights=N_CST)
        x_recon, y_recon = reconstruct_airfoil(params, n_points=200)
        
        max_err = np.max(np.abs(np.interp(
            np.linspace(0, 1, 100),
            x_orig[np.argsort(x_orig)],
            y_orig[np.argsort(x_orig)]
        ) - np.interp(
            np.linspace(0, 1, 100),
            x_recon[np.argsort(x_recon)],
            y_recon[np.argsort(x_recon)]
        )))
        
        ax.plot(x_orig, y_orig, 'k.', markersize=1.5, alpha=0.6, label='Original')
        ax.plot(x_recon, y_recon, '-', color=COLORS['CL'], linewidth=1.5, label='CST (n=6)')
        ax.set_xlim(-0.05, 1.05)
        ax.set_aspect('equal')
        ax.set_title(f'{name}\nMax err: {max_err:.4f}', fontsize=9)
        if idx == 0:
            ax.legend(fontsize=8)
        ax.set_xlabel('x/c')
        ax.set_ylabel('y/c')
    
    fig.suptitle('CST Parameterization: Reconstruction Quality', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, '09_cst_reconstruction.png'))
    plt.close()
    print("   ✓ Saved 09_cst_reconstruction.png")


# ═══════════════ FIGURE 10: DATASET OVERVIEW ═══════════════
def fig_dataset_overview(data):
    """Visualize data distribution across the dataset."""
    print("\n📊 Figure 10: Dataset Overview...")
    
    df = data['train']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    # Row 1: Target distributions
    for i, (ax, name) in enumerate(zip(axes[0], TARGET_COLS)):
        ax.hist(df[name], bins=100, color=COLORS[name], alpha=0.7,
               edgecolor='white', linewidth=0.3)
        ax.set_xlabel(name)
        ax.set_ylabel('Count')
        ax.set_title(f'{name} Distribution (Train)')
        ax.axvline(df[name].mean(), color='k', linestyle='--', alpha=0.5,
                  label=f'μ={df[name].mean():.3f}')
        ax.legend()
    
    # Row 2: Mach distribution, alpha distribution, samples per airfoil
    ax = axes[1, 0]
    mach_counts = df['mach'].value_counts().sort_index()
    ax.bar(mach_counts.index.astype(str), mach_counts.values,
          color=COLORS['CL'], alpha=0.7, edgecolor='white')
    ax.set_xlabel('Mach Number')
    ax.set_ylabel('Count')
    ax.set_title('Samples per Mach')
    
    ax = axes[1, 1]
    ax.hist(df['alpha'], bins=50, color=COLORS['CD'], alpha=0.7,
           edgecolor='white', linewidth=0.3)
    ax.set_xlabel('Angle of Attack α (°)')
    ax.set_ylabel('Count')
    ax.set_title('Alpha Distribution')
    
    ax = axes[1, 2]
    samples_per_airfoil = df.groupby('airfoil_name').size()
    ax.hist(samples_per_airfoil, bins=50, color=COLORS['CM'], alpha=0.7,
           edgecolor='white', linewidth=0.3)
    ax.set_xlabel('Samples per Airfoil')
    ax.set_ylabel('Number of Airfoils')
    ax.set_title('Data Points per Airfoil')
    
    fig.suptitle('Dataset Overview', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, '10_dataset_overview.png'))
    plt.close()
    print("   ✓ Saved 10_dataset_overview.png")


# ═══════════════ MAIN ═══════════════════════════
def generate_all():
    """Generate all report figures."""
    os.makedirs(REPORT_DIR, exist_ok=True)
    
    print("=" * 60)
    print("  REPORT FIGURE GENERATOR")
    print("=" * 60)
    
    model, normalizer, data, info = load_model_and_data()
    
    print(f"\n   Model trained for {info['epoch']} epochs")
    print(f"   Best val loss: {info['val_loss']:.6f}")
    
    # Generate all figures
    fig_parity_plots(model, normalizer, data)
    fig_error_distributions(model, normalizer, data)
    fig_error_vs_mach(model, normalizer, data)
    fig_residuals(model, normalizer, data)
    fig_error_vs_alpha(model, normalizer, data)
    fig_r2_heatmap(model, normalizer, data)
    fig_inverse_design_examples(model, normalizer)
    fig_metrics_table(model, normalizer, data)
    fig_cst_reconstruction(data)
    fig_dataset_overview(data)
    
    print("\n" + "=" * 60)
    print(f"  ✓ All 10 figures saved to: {REPORT_DIR}")
    print("=" * 60)
    
    # Print summary
    y_true, y_pred = predict_split(model, normalizer, data['test'])
    r2 = compute_r2(y_true, y_pred)
    print(f"\n  Test Set R²:  CL={r2[0]:.4f}  CD={r2[1]:.4f}  CM={r2[2]:.4f}")


if __name__ == "__main__":
    generate_all()
