"""
Class Shape Transformation (CST) Parameterization Module
=========================================================
Implements CST method for airfoil shape representation.
CST uses Bernstein polynomials to decompose an airfoil into
a class function * shape function, enabling compact parametric
representation with typically 12 coefficients (6 upper + 6 lower).
"""

import numpy as np
from scipy.optimize import least_squares
from math import comb


def bernstein_poly(n, k, x):
    """Compute the k-th Bernstein basis polynomial of degree n at x."""
    return comb(n, k) * (x ** k) * ((1.0 - x) ** (n - k))


def class_function(x, N1=0.5, N2=1.0):
    """
    CST class function for airfoil shapes.
    C(x) = x^N1 * (1-x)^N2
    Default N1=0.5, N2=1.0 gives round leading edge, sharp trailing edge.
    """
    return (x ** N1) * ((1.0 - x) ** N2)


def shape_function(x, weights):
    """
    CST shape function using Bernstein polynomial basis.
    S(x) = sum_{k=0}^{n} w_k * B_{k,n}(x)
    """
    n = len(weights) - 1
    S = np.zeros_like(x)
    for k, w in enumerate(weights):
        S += w * bernstein_poly(n, k, x)
    return S


def cst_curve(x, weights, dz_te=0.0):
    """
    Compute the y-coordinates of a CST curve.
    y(x) = C(x) * S(x) + x * dz_te
    
    Parameters
    ----------
    x : array-like, x-coordinates (0 to 1)
    weights : array-like, CST weight coefficients
    dz_te : float, trailing edge thickness offset
    
    Returns
    -------
    y : array, y-coordinates
    """
    x = np.asarray(x, dtype=np.float64)
    C = class_function(x)
    S = shape_function(x, weights)
    return C * S + x * dz_te


def fit_cst_weights(x, y, n_weights=6, dz_te=None):
    """
    Fit CST weights to match given (x, y) coordinates.
    
    Parameters
    ----------
    x : array-like, x-coordinates (0 to 1, normalized)
    y : array-like, y-coordinates
    n_weights : int, number of CST weights to fit
    dz_te : float or None, trailing edge offset (auto-detected if None)
    
    Returns
    -------
    weights : array, fitted CST weight coefficients
    dz_te : float, trailing edge thickness offset used
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    
    # Avoid division by zero at x=0 and x=1
    eps = 1e-10
    x = np.clip(x, eps, 1.0 - eps)
    
    if dz_te is None:
        dz_te = y[-1] if len(y) > 0 else 0.0
    
    C = class_function(x)
    
    # Target shape function values: S(x) = (y - x*dz_te) / C(x)
    target_S = (y - x * dz_te) / (C + eps)
    
    # Build Bernstein basis matrix
    n = n_weights - 1
    B = np.zeros((len(x), n_weights))
    for k in range(n_weights):
        B[:, k] = bernstein_poly(n, k, x)
    
    # Least squares fit
    weights, _, _, _ = np.linalg.lstsq(B, target_S, rcond=None)
    
    return weights, dz_te


def split_airfoil(x_coords, y_coords):
    """
    Split airfoil coordinates into upper and lower surfaces.
    Assumes standard Selig format (TE→upper→LE→lower→TE).
    
    Returns
    -------
    x_upper, y_upper, x_lower, y_lower
    """
    x_coords = np.asarray(x_coords)
    y_coords = np.asarray(y_coords)
    
    # Find leading edge (minimum x)
    le_idx = np.argmin(x_coords)
    
    # Upper surface: from LE going backwards (reverse to get LE→TE order)
    x_upper = x_coords[:le_idx + 1][::-1]
    y_upper = y_coords[:le_idx + 1][::-1]
    
    # Lower surface: from LE going forward
    x_lower = x_coords[le_idx:]
    y_lower = y_coords[le_idx:]
    
    return x_upper, y_upper, x_lower, y_lower


def fit_airfoil_cst(x_coords, y_coords, n_weights=6):
    """
    Fit CST parameters to a complete airfoil.
    
    Parameters
    ----------
    x_coords : array, x-coordinates of the full airfoil
    y_coords : array, y-coordinates of the full airfoil
    n_weights : int, number of CST weights per surface
    
    Returns
    -------
    cst_params : dict with 'upper_weights', 'lower_weights', 
                 'upper_dz_te', 'lower_dz_te'
    """
    x_u, y_u, x_l, y_l = split_airfoil(x_coords, y_coords)
    
    w_upper, dz_upper = fit_cst_weights(x_u, y_u, n_weights)
    w_lower, dz_lower = fit_cst_weights(x_l, y_l, n_weights)
    
    return {
        'upper_weights': w_upper,
        'lower_weights': w_lower,
        'upper_dz_te': dz_upper,
        'lower_dz_te': dz_lower
    }


def reconstruct_airfoil(cst_params, n_points=200):
    """
    Reconstruct airfoil coordinates from CST parameters.
    
    Parameters
    ----------
    cst_params : dict, CST parameters from fit_airfoil_cst
    n_points : int, number of points per surface
    
    Returns
    -------
    x_full, y_full : arrays of full airfoil coordinates (Selig order)
    """
    # Cosine spacing for better LE resolution
    beta = np.linspace(0, np.pi, n_points)
    x = 0.5 * (1.0 - np.cos(beta))
    
    y_upper = cst_curve(x, cst_params['upper_weights'], cst_params['upper_dz_te'])
    y_lower = cst_curve(x, cst_params['lower_weights'], cst_params['lower_dz_te'])
    
    # Selig format: upper TE→LE then lower LE→TE
    x_full = np.concatenate([x[::-1], x[1:]])
    y_full = np.concatenate([y_upper[::-1], y_lower[1:]])
    
    return x_full, y_full


def cst_vector(cst_params):
    """Flatten CST params into a single parameter vector."""
    return np.concatenate([
        cst_params['upper_weights'],
        cst_params['lower_weights']
    ])


def cst_from_vector(vec, n_weights=6):
    """Reconstruct CST params dict from a flat parameter vector."""
    return {
        'upper_weights': vec[:n_weights],
        'lower_weights': vec[n_weights:2 * n_weights],
        'upper_dz_te': 0.0,
        'lower_dz_te': 0.0
    }


def read_dat_file(filepath):
    """
    Read an airfoil .dat coordinate file.
    
    Returns
    -------
    x, y : arrays of coordinates
    """
    x_coords = []
    y_coords = []
    with open(filepath, 'r') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            # Skip header line (airfoil name)
            parts = line.split()
            if len(parts) == 2:
                try:
                    xv = float(parts[0])
                    yv = float(parts[1])
                    x_coords.append(xv)
                    y_coords.append(yv)
                except ValueError:
                    continue
    return np.array(x_coords), np.array(y_coords)


def test_cst_roundtrip():
    """Test CST fit and reconstruction accuracy."""
    # Generate a known NACA 0012-like shape
    beta = np.linspace(0, np.pi, 100)
    x = 0.5 * (1.0 - np.cos(beta))
    y_upper = 0.6 * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 
                       + 0.2843 * x**3 - 0.1015 * x**4)
    y_lower = -y_upper
    
    x_full = np.concatenate([x[::-1], x[1:]])
    y_full = np.concatenate([y_upper[::-1], y_lower[1:]])
    
    # Fit and reconstruct
    params = fit_airfoil_cst(x_full, y_full, n_weights=6)
    x_recon, y_recon = reconstruct_airfoil(params, n_points=100)
    
    max_err = np.max(np.abs(y_full - y_recon))
    print(f"CST Roundtrip Test: max error = {max_err:.6f}")
    assert max_err < 0.005, f"CST roundtrip error too large: {max_err:.6f}"
    print("✓ CST roundtrip test PASSED")
    
    return params


if __name__ == "__main__":
    test_cst_roundtrip()
