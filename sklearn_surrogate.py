"""
Scikit-learn Surrogate Model (Fallback)
========================================
Uses sklearn MLPRegressor when PyTorch is unavailable.
Provides the same interface as the PyTorch surrogate.
"""

import os
import numpy as np
import pandas as pd
import pickle
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "processed_data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")

N_CST = 6
FEATURE_COLS = (
    [f'cst_upper_{i}' for i in range(N_CST)] +
    [f'cst_lower_{i}' for i in range(N_CST)] +
    ['mach', 'alpha']
)
TARGET_COLS = ['CL', 'CD', 'CM']


class SklearnSurrogate:
    """Sklearn-based surrogate with the same prediction interface."""
    
    def __init__(self):
        self.model = None
        self.feature_scaler = MinMaxScaler()
        self.target_scaler = MinMaxScaler()
    
    def train(self, max_samples=200000, epochs=200):
        """Train the sklearn MLP surrogate."""
        os.makedirs(MODELS_DIR, exist_ok=True)
        os.makedirs(PLOTS_DIR, exist_ok=True)
        
        print("\n" + "=" * 60)
        print("  SKLEARN SURROGATE MODEL TRAINING")
        print("=" * 60)
        
        # Load data
        print("\n📂 Loading data...")
        train_path = os.path.join(DATA_DIR, "train.csv")
        val_path = os.path.join(DATA_DIR, "val.csv")
        test_path = os.path.join(DATA_DIR, "test.csv")
        
        if not os.path.exists(train_path):
            print("✗ No training data. Run preprocess_data.py first.")
            return
        
        df_train = pd.read_csv(train_path)
        print(f"  Train: {len(df_train)} samples")
        
        # Subsample for reasonable training time
        if len(df_train) > max_samples:
            df_train = df_train.sample(n=max_samples, random_state=42)
            print(f"  Subsampled to {max_samples} for training speed")
        
        X_train = df_train[FEATURE_COLS].values
        y_train = df_train[TARGET_COLS].values
        
        # Load validation
        if os.path.exists(val_path):
            df_val = pd.read_csv(val_path)
            val_n = min(len(df_val), 50000)
            df_val = df_val.sample(n=val_n, random_state=42)
            X_val = df_val[FEATURE_COLS].values
            y_val = df_val[TARGET_COLS].values
            print(f"  Val:   {val_n} samples")
        else:
            X_val, y_val = X_train[:5000], y_train[:5000]
        
        # Normalize
        print("\n📊 Normalizing data...")
        X_train_norm = self.feature_scaler.fit_transform(X_train)
        y_train_norm = self.target_scaler.fit_transform(y_train)
        X_val_norm = self.feature_scaler.transform(X_val)
        
        # Train MLP
        print(f"\n🚀 Training MLP (max_iter={epochs})...")
        print("   Architecture: 128→256→256→128")
        print("   This may take a few minutes...")
        
        self.model = MLPRegressor(
            hidden_layer_sizes=(128, 256, 256, 128),
            activation='relu',
            solver='adam',
            alpha=1e-4,
            batch_size=1024,
            learning_rate='adaptive',
            learning_rate_init=1e-3,
            max_iter=epochs,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15,
            verbose=True,
            random_state=42
        )
        
        self.model.fit(X_train_norm, y_train_norm)
        
        print(f"\n   ✓ Training complete in {self.model.n_iter_} iterations")
        
        # Evaluate
        print("\n📋 Evaluation:")
        y_val_pred_norm = self.model.predict(X_val_norm)
        y_val_pred = self.target_scaler.inverse_transform(y_val_pred_norm)
        
        r2 = r2_score(y_val, y_val_pred, multioutput='raw_values')
        mse = mean_squared_error(y_val, y_val_pred, multioutput='raw_values')
        
        for i, name in enumerate(TARGET_COLS):
            print(f"   {name}: R²={r2[i]:.4f}  MSE={mse[i]:.6f}")
        
        # Save model
        model_data = {
            'model': self.model,
            'feature_scaler': self.feature_scaler,
            'target_scaler': self.target_scaler,
            'feature_cols': FEATURE_COLS,
            'target_cols': TARGET_COLS,
            'r2_scores': r2
        }
        model_path = os.path.join(MODELS_DIR, 'sklearn_surrogate.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"\n   ✓ Model saved to {model_path}")
        
        # Plot
        self._plot_results(y_val, y_val_pred, r2)
        
        # Test set
        if os.path.exists(test_path):
            df_test = pd.read_csv(test_path)
            test_n = min(len(df_test), 50000)
            df_test = df_test.sample(n=test_n, random_state=42)
            X_test = df_test[FEATURE_COLS].values
            y_test = df_test[TARGET_COLS].values
            X_test_norm = self.feature_scaler.transform(X_test)
            y_test_pred_norm = self.model.predict(X_test_norm)
            y_test_pred = self.target_scaler.inverse_transform(y_test_pred_norm)
            test_r2 = r2_score(y_test, y_test_pred, multioutput='raw_values')
            print(f"\n📋 Test Set:")
            for i, name in enumerate(TARGET_COLS):
                print(f"   {name}: R²={test_r2[i]:.4f}")
        
        print("\n✓ Training complete!")
    
    def _plot_results(self, y_true, y_pred, r2):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        colors = ['#4F46E5', '#10B981', '#F59E0B']
        labels = TARGET_COLS
        
        for i, (ax, lbl, col) in enumerate(zip(axes, labels, colors)):
            ax.scatter(y_true[:, i], y_pred[:, i], alpha=0.1, s=2, color=col)
            lims = [min(y_true[:, i].min(), y_pred[:, i].min()),
                    max(y_true[:, i].max(), y_pred[:, i].max())]
            ax.plot(lims, lims, 'k--', alpha=0.5)
            ax.set_xlabel(f'True {lbl}')
            ax.set_ylabel(f'Predicted {lbl}')
            ax.set_title(f'{lbl} (R²={r2[i]:.4f})')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, 'training_results.png'), dpi=150)
        plt.close()
        print(f"   ✓ Plots saved to {PLOTS_DIR}/training_results.png")
    
    def predict(self, cst_upper, cst_lower, mach, alpha):
        """Predict aero coefficients."""
        features = np.concatenate([cst_upper, cst_lower, [mach, alpha]])
        features = features.reshape(1, -1)
        features_norm = self.feature_scaler.transform(features)
        pred_norm = self.model.predict(features_norm)
        pred = self.target_scaler.inverse_transform(pred_norm)
        return {
            'CL': float(pred[0, 0]),
            'CD': float(pred[0, 1]),
            'CM': float(pred[0, 2])
        }


def load_sklearn_model(model_path=None):
    """Load a trained sklearn surrogate."""
    if model_path is None:
        model_path = os.path.join(MODELS_DIR, 'sklearn_surrogate.pkl')
    
    with open(model_path, 'rb') as f:
        data = pickle.load(f)
    
    surrogate = SklearnSurrogate()
    surrogate.model = data['model']
    surrogate.feature_scaler = data['feature_scaler']
    surrogate.target_scaler = data['target_scaler']
    return surrogate


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-samples', type=int, default=200000)
    parser.add_argument('--epochs', type=int, default=200)
    args = parser.parse_args()
    
    surrogate = SklearnSurrogate()
    surrogate.train(max_samples=args.max_samples, epochs=args.epochs)
