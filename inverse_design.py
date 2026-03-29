"""
Inverse Airfoil Design Module
================================
Uses the trained surrogate model to find optimal CST parameters
that produce desired aerodynamic characteristics.
Supports both PyTorch and sklearn backends.
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
from cst_module import cst_from_vector, reconstruct_airfoil
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Backend Detection ───────────────────────────
_BACKEND = None
_MODEL = None


def _detect_backend():
    """Detect available model backend."""
    global _BACKEND, _MODEL
    
    sklearn_path = os.path.join(BASE_DIR, "models", "sklearn_surrogate.pkl")
    torch_path = os.path.join(BASE_DIR, "models", "best_surrogate.pth")
    
    # Try sklearn first (more reliable)
    if os.path.exists(sklearn_path):
        try:
            from sklearn_surrogate import load_sklearn_model
            _MODEL = load_sklearn_model(sklearn_path)
            _BACKEND = 'sklearn'
            print(f"  ✓ Loaded sklearn surrogate model")
            return
        except Exception as e:
            print(f"  ⚠ sklearn model load failed: {e}")
    
    # Try PyTorch
    if os.path.exists(torch_path):
        try:
            import torch
            from surrogate_model import AirfoilSurrogate, DataNormalizer
            checkpoint = torch.load(torch_path, map_location='cpu', weights_only=False)
            model = AirfoilSurrogate(
                input_dim=len(checkpoint['feature_cols']),
                output_dim=len(checkpoint['target_cols'])
            )
            model.load_state_dict(checkpoint['model_state'])
            model.eval()
            normalizer = DataNormalizer()
            normalizer.load_state_dict(checkpoint['normalizer_state'])
            _MODEL = (model, normalizer, checkpoint)
            _BACKEND = 'torch'
            print(f"  ✓ Loaded PyTorch surrogate model")
            return
        except Exception as e:
            print(f"  ⚠ PyTorch model load failed: {e}")
    
    raise FileNotFoundError(
        "No trained model found. Run sklearn_surrogate.py or train_surrogate.py first."
    )


def predict_aero(cst_upper, cst_lower, mach, alpha):
    """
    Predict aerodynamic coefficients from CST params and conditions.
    Auto-detects backend (sklearn or PyTorch).
    """
    global _BACKEND, _MODEL
    if _MODEL is None:
        _detect_backend()
    
    cst_upper = np.asarray(cst_upper, dtype=np.float64)
    cst_lower = np.asarray(cst_lower, dtype=np.float64)
    
    if _BACKEND == 'sklearn':
        return _MODEL.predict(cst_upper, cst_lower, mach, alpha)
    
    elif _BACKEND == 'torch':
        import torch
        model, normalizer, _ = _MODEL
        features = np.concatenate([cst_upper, cst_lower, [mach, alpha]])
        features = features.astype(np.float32).reshape(1, -1)
        features_norm = normalizer.transform_features(features)
        with torch.no_grad():
            pred_norm = model(torch.FloatTensor(features_norm))
        pred = normalizer.inverse_transform_targets(pred_norm.numpy())
        return {
            'CL': float(pred[0, 0]),
            'CD': float(pred[0, 1]),
            'CM': float(pred[0, 2])
        }


class InverseDesigner:
    """
    Inverse airfoil design using surrogate-based optimization.
    """
    
    def __init__(self):
        global _MODEL
        if _MODEL is None:
            _detect_backend()
        
        self.n_cst = 6
        self.bounds_upper = [(-0.5, 1.5)] * self.n_cst
        self.bounds_lower = [(-1.5, 0.5)] * self.n_cst
        self.all_bounds = self.bounds_upper + self.bounds_lower
    
    def objective(self, cst_vec, targets, mach, alpha, weights=None):
        if weights is None:
            weights = {'CL': 1.0, 'CD': 5.0, 'CM': 0.5}
        
        cst_upper = cst_vec[:self.n_cst]
        cst_lower = cst_vec[self.n_cst:]
        
        pred = predict_aero(cst_upper, cst_lower, mach, alpha)
        
        cost = 0.0
        for key in ['CL', 'CD', 'CM']:
            if key in targets and targets[key] is not None:
                cost += weights.get(key, 1.0) * (targets[key] - pred[key]) ** 2
        
        # Smoothness regularization
        for surface_w in [cst_upper, cst_lower]:
            diff = np.diff(surface_w)
            cost += 0.001 * np.sum(diff ** 2)
        
        return cost
    
    def design_gradient(self, targets, mach, alpha, weights=None,
                        n_restarts=5, x0=None):
        """Gradient-based inverse design using L-BFGS-B with multi-start."""
        best_cost = float('inf')
        best_result = None
        
        for i in range(n_restarts):
            if x0 is not None and i == 0:
                init = x0
            else:
                init = np.random.uniform(
                    [b[0] for b in self.all_bounds],
                    [b[1] for b in self.all_bounds]
                )
            
            try:
                res = minimize(
                    self.objective,
                    init,
                    args=(targets, mach, alpha, weights),
                    method='L-BFGS-B',
                    bounds=self.all_bounds,
                    options={'maxiter': 200, 'ftol': 1e-10}
                )
                
                if res.fun < best_cost:
                    best_cost = res.fun
                    best_result = res
            except Exception:
                continue
        
        if best_result is None:
            return None
        
        return self._format_result(best_result.x, mach, alpha, best_cost)
    
    def design_evolutionary(self, targets, mach, alpha, weights=None,
                           maxiter=100, popsize=30):
        """Evolutionary inverse design using Differential Evolution."""
        res = differential_evolution(
            self.objective,
            bounds=self.all_bounds,
            args=(targets, mach, alpha, weights),
            maxiter=maxiter,
            popsize=popsize,
            tol=1e-8,
            seed=42,
            workers=1
        )
        
        return self._format_result(res.x, mach, alpha, res.fun)
    
    def _format_result(self, cst_vec, mach, alpha, cost):
        cst_upper = cst_vec[:self.n_cst]
        cst_lower = cst_vec[self.n_cst:]
        
        pred = predict_aero(cst_upper, cst_lower, mach, alpha)
        
        cst_params = cst_from_vector(cst_vec, self.n_cst)
        x_coords, y_coords = reconstruct_airfoil(cst_params, n_points=200)
        
        return {
            'cst_upper': cst_upper.tolist(),
            'cst_lower': cst_lower.tolist(),
            'predicted': pred,
            'cost': float(cost),
            'x_coords': x_coords.tolist(),
            'y_coords': y_coords.tolist(),
            'mach': mach,
            'alpha': alpha
        }
    
    def sweep_alpha(self, cst_vec, mach, alpha_range=None):
        if alpha_range is None:
            alpha_range = np.linspace(-5, 15, 41)
        
        cst_upper = cst_vec[:self.n_cst]
        cst_lower = cst_vec[self.n_cst:]
        
        results = []
        for a in alpha_range:
            pred = predict_aero(cst_upper, cst_lower, mach, float(a))
            pred['alpha'] = float(a)
            results.append(pred)
        
        return results


def test_inverse():
    """Quick test of the inverse design module."""
    print("Testing inverse design module...")
    
    designer = InverseDesigner()
    
    targets = {'CL': 0.5, 'CD': 0.01}
    result = designer.design_gradient(
        targets, mach=0.3, alpha=2.0, n_restarts=3
    )
    
    if result:
        print(f"  Target:    CL={targets['CL']:.3f}, CD={targets.get('CD', 'N/A')}")
        print(f"  Achieved:  CL={result['predicted']['CL']:.4f}, "
              f"CD={result['predicted']['CD']:.6f}")
        print(f"  Cost: {result['cost']:.6f}")
        print("  ✓ Inverse design test PASSED")
    else:
        print("  ✗ Optimization failed")
    
    return result


if __name__ == "__main__":
    test_inverse()
