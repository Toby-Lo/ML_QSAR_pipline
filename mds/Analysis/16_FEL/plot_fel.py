#!/usr/bin/env python3
#python3 ../../Analysis/16_FEL/plot_fel.py -i analysis/FEL_input.xvg -o analysis/plots/FEL_plot.png


import argparse
import numpy as np
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="Generate Free Energy Landscape (FEL) from PCA modes.")
    parser.add_argument("-i", "--input", required=True, help="Input XVG/DAT file with PCA modes (e.g., FEL_input.xvg)")
    parser.add_argument("-o", "--output", default="FEL_plot.png", help="Output plot filename")
    parser.add_argument("-t", "--temperature", type=float, default=300.0, help="Temperature in Kelvin")
    parser.add_argument("-b", "--bins", type=int, default=50, help="Number of bins for 2D histogram")
    args = parser.parse_args()

    R = 0.008314  # Ideal gas constant in kJ/(mol·K)

    print(f"[INFO] Reading data from {args.input}...")
    # Load data; comments=['#', '@'] ignores headers in typical xvg/dat files
    data = np.loadtxt(args.input, comments=['#', '@'])
    
    # Extract columns: Frame (0), Mode1 (1), Mode2 (2)
    pca1 = data[:, 1]
    pca2 = data[:, 2]

    print(f"[INFO] Calculating probability distribution with {args.bins} bins...")
    hist, xedges, yedges = np.histogram2d(pca1, pca2, bins=args.bins, density=True)
    hist = hist.T  # Transpose to align axes for matplotlib contourf

    # Avoid log(0) by replacing zeros with a tiny fractional value
    hist[hist == 0] = np.min(hist[hist > 0]) * 0.01

    print("[INFO] Calculating Gibbs Free Energy (ΔG)...")
    free_energy = -R * args.temperature * np.log(hist)
    free_energy -= np.min(free_energy)  # Normalize so the global energy minimum is 0

    print(f"[INFO] Rendering FEL plot to {args.output}...")
    X, Y = np.meshgrid(xedges[:-1], yedges[:-1])
    
    plt.figure(figsize=(8, 6))
    cp = plt.contourf(X, Y, free_energy, levels=20, cmap='nipy_spectral')
    cbar = plt.colorbar(cp)
    cbar.set_label('Gibbs Free Energy $\Delta G$ (kJ/mol)')
    
    plt.xlabel('PC1 (Mode 1)')
    plt.ylabel('PC2 (Mode 2)')
    plt.title('Free Energy Landscape (FEL)')
    plt.tight_layout()
    plt.savefig(args.output, dpi=300)
    print("[OK] Done!")

if __name__ == "__main__":
    main()