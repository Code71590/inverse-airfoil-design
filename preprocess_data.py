"""
Data Preprocessing Pipeline
============================
Parses XFOIL polar files and airfoil .dat coordinate files,
fits CST parameters, and creates a consolidated dataset.
"""

import os
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from cst_module import read_dat_file, fit_airfoil_cst, cst_vector

# ── Paths ──────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SHAPE_DIR = os.path.join(BASE_DIR, "Shape", "data", "airfoil", "cst_gen")
AERO_DIR = os.path.join(BASE_DIR, "Aerodynamic", "aerodynamic_label", "cst_gen")
SPLIT_DIR = os.path.join(BASE_DIR, "Shape", "data", "airfoil")
OUTPUT_DIR = os.path.join(BASE_DIR, "processed_data")

N_CST_WEIGHTS = 6  # per surface → 12 total


def parse_xfoil_polar(filepath):
    """
    Parse an XFOIL polar file and extract aerodynamic data.
    
    Returns
    -------
    list of dicts with keys: alpha, CL, CD, CDp, CM
    Returns empty list if no data found.
    """
    records = []
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
    except Exception:
        return records
    
    data_started = False
    for line in lines:
        line = line.strip()
        if line.startswith('------'):
            data_started = True
            continue
        if data_started and line:
            parts = line.split()
            if len(parts) >= 5:
                try:
                    rec = {
                        'alpha': float(parts[0]),
                        'CL': float(parts[1]),
                        'CD': float(parts[2]),
                        'CDp': float(parts[3]),
                        'CM': float(parts[4])
                    }
                    records.append(rec)
                except ValueError:
                    continue
    return records


def extract_mach_from_polar(filepath):
    """Extract Mach number from XFOIL polar file header."""
    try:
        with open(filepath, 'r') as f:
            for line in f:
                if 'Mach' in line and 'Re' in line:
                    match = re.search(r'Mach\s*=\s*([\d.]+)', line)
                    if match:
                        return float(match.group(1))
    except Exception:
        pass
    return None


def get_airfoil_names(split_file):
    """Read airfoil names from a split file."""
    names = []
    with open(split_file, 'r') as f:
        for line in f:
            name = line.strip()
            if name:
                names.append(name)
    return names


def process_single_airfoil(name):
    """
    Process a single airfoil: fit CST + collect aero data.
    
    Returns
    -------
    list of dicts, one per (mach, alpha) condition
    """
    # Read shape file
    dat_path = os.path.join(SHAPE_DIR, f"{name}.dat")
    if not os.path.exists(dat_path):
        return []
    
    try:
        x_coords, y_coords = read_dat_file(dat_path)
        if len(x_coords) < 10:
            return []
        cst_params = fit_airfoil_cst(x_coords, y_coords, N_CST_WEIGHTS)
        cst_vec = cst_vector(cst_params)
    except Exception:
        return []
    
    # Read aerodynamic data from all Mach conditions
    aero_dir = os.path.join(AERO_DIR, name)
    if not os.path.isdir(aero_dir):
        return []
    
    records = []
    for fname in os.listdir(aero_dir):
        # Skip non-polar files
        if fname == 'input_file.in' or fname.endswith('.dat'):
            continue
        
        fpath = os.path.join(aero_dir, fname)
        if not os.path.isfile(fpath):
            continue
        
        # Mach number is the filename
        try:
            mach = float(fname)
        except ValueError:
            continue
        
        # Parse polar
        polar_data = parse_xfoil_polar(fpath)
        for entry in polar_data:
            row = {'airfoil_name': name}
            # CST coefficients
            for i in range(N_CST_WEIGHTS):
                row[f'cst_upper_{i}'] = cst_vec[i]
            for i in range(N_CST_WEIGHTS):
                row[f'cst_lower_{i}'] = cst_vec[N_CST_WEIGHTS + i]
            # Operating conditions
            row['mach'] = mach
            row['alpha'] = entry['alpha']
            # Aerodynamic coefficients
            row['CL'] = entry['CL']
            row['CD'] = entry['CD']
            row['CM'] = entry['CM']
            records.append(row)
    
    return records


def preprocess_dataset():
    """Main preprocessing pipeline."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Process each split
    for split_name in ['train', 'val', 'test']:
        split_file = os.path.join(SPLIT_DIR, f"cst_gen_{split_name}.txt")
        if not os.path.exists(split_file):
            print(f"⚠ Split file not found: {split_file}")
            continue
        
        airfoil_names = get_airfoil_names(split_file)
        print(f"\n{'='*60}")
        print(f"Processing {split_name} split: {len(airfoil_names)} airfoils")
        print(f"{'='*60}")
        
        all_records = []
        failed = 0
        
        for name in tqdm(airfoil_names, desc=f"  {split_name}"):
            records = process_single_airfoil(name)
            if records:
                all_records.extend(records)
            else:
                failed += 1
        
        if all_records:
            df = pd.DataFrame(all_records)
            
            # Filter out invalid data
            df = df.dropna()
            df = df[df['CD'] > 0]  # Physical constraint
            df = df[df['CD'] < 1.0]  # Remove outliers
            
            output_path = os.path.join(OUTPUT_DIR, f"{split_name}.csv")
            df.to_csv(output_path, index=False)
            
            print(f"  ✓ {split_name}: {len(df)} samples from "
                  f"{len(df['airfoil_name'].unique())} airfoils "
                  f"({failed} failed)")
            print(f"    CL range: [{df['CL'].min():.3f}, {df['CL'].max():.3f}]")
            print(f"    CD range: [{df['CD'].min():.5f}, {df['CD'].max():.5f}]")
            print(f"    Mach values: {sorted(df['mach'].unique())}")
            print(f"    Saved to: {output_path}")
        else:
            print(f"  ✗ No valid data for {split_name} split")
    
    print("\n✓ Preprocessing complete!")


if __name__ == "__main__":
    preprocess_dataset()
