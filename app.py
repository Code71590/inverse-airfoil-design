"""
Flask Web Application for Inverse Airfoil Design
===================================================
Premium interactive dashboard for the surrogate-based
inverse airfoil design system.
"""

import os
import json
import numpy as np
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Lazy-load the model
_designer = None


def get_designer():
    global _designer
    if _designer is None:
        model_path = os.path.join(BASE_DIR, "models", "best_surrogate.pth")
        if not os.path.exists(model_path):
            return None
        from inverse_design import InverseDesigner
        _designer = InverseDesigner()
    return _designer


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/predict', methods=['POST'])
def predict():
    """Forward prediction: CST params → aero coefficients."""
    designer = get_designer()
    if designer is None:
        return jsonify({'error': 'Model not trained yet. Run train_surrogate.py first.'}), 503
    
    data = request.json
    try:
        cst_upper = np.array(data['cst_upper'], dtype=float)
        cst_lower = np.array(data['cst_lower'], dtype=float)
        mach = float(data['mach'])
        alpha = float(data['alpha'])
        
        from inverse_design import predict_aero
        result = predict_aero(designer.model, designer.normalizer,
                             cst_upper, cst_lower, mach, alpha)
        
        from cst_module import cst_from_vector, reconstruct_airfoil
        cst_vec = np.concatenate([cst_upper, cst_lower])
        cst_params = cst_from_vector(cst_vec)
        x_coords, y_coords = reconstruct_airfoil(cst_params)
        
        return jsonify({
            'predicted': result,
            'x_coords': x_coords.tolist(),
            'y_coords': y_coords.tolist()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/inverse_design', methods=['POST'])
def inverse_design():
    """Inverse design: target aero → optimal CST params."""
    designer = get_designer()
    if designer is None:
        return jsonify({'error': 'Model not trained yet. Run train_surrogate.py first.'}), 503
    
    data = request.json
    try:
        targets = {}
        if data.get('target_CL') is not None and data.get('target_CL') != '':
            targets['CL'] = float(data['target_CL'])
        if data.get('target_CD') is not None and data.get('target_CD') != '':
            targets['CD'] = float(data['target_CD'])
        if data.get('target_CM') is not None and data.get('target_CM') != '':
            targets['CM'] = float(data['target_CM'])
        
        mach = float(data.get('mach', 0.3))
        alpha = float(data.get('alpha', 2.0))
        method = data.get('method', 'gradient')
        
        weights = {
            'CL': float(data.get('weight_CL', 1.0)),
            'CD': float(data.get('weight_CD', 5.0)),
            'CM': float(data.get('weight_CM', 0.5))
        }
        
        if method == 'evolutionary':
            result = designer.design_evolutionary(
                targets, mach, alpha, weights,
                maxiter=50, popsize=20
            )
        else:
            result = designer.design_gradient(
                targets, mach, alpha, weights,
                n_restarts=5
            )
        
        if result is None:
            return jsonify({'error': 'Optimization failed to converge'}), 500
        
        # Alpha sweep for performance plots
        cst_vec = np.array(result['cst_upper'] + result['cst_lower'])
        sweep = designer.sweep_alpha(cst_vec, mach)
        result['alpha_sweep'] = sweep
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/alpha_sweep', methods=['POST'])
def alpha_sweep():
    """Generate CL/CD vs alpha curves."""
    designer = get_designer()
    if designer is None:
        return jsonify({'error': 'Model not trained yet.'}), 503
    
    data = request.json
    try:
        cst_upper = np.array(data['cst_upper'], dtype=float)
        cst_lower = np.array(data['cst_lower'], dtype=float)
        mach = float(data.get('mach', 0.3))
        
        cst_vec = np.concatenate([cst_upper, cst_lower])
        sweep = designer.sweep_alpha(cst_vec, mach)
        
        return jsonify({'sweep': sweep})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("  INVERSE AIRFOIL DESIGN SYSTEM")
    print("=" * 60)
    print("  Open http://localhost:5000 in your browser")
    print("=" * 60 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
