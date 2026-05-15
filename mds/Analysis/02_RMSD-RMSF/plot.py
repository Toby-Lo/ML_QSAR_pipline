#!/usr/bin/env python3
"""Quick plotting entry for this analysis folder.

Typical usage (run from your simulation directory):
  - RMSD Time series:
      python3 ../../Analysis/02_RMSD-RMSF/plot_timeseries.py \
        analysis/Calpha_RMSD.dat \
        -o analysis/plots/02_Calpha_RMSD.png
  - RMSF :
      python3 ../../Analysis/02_RMSD-RMSF/plot_timeseries.py \
        analysis/RMSF.dat \
        -o analysis/plots/02_RMSF_Profile.png
        
  - Heatmaps (2D-RMSD/DCCM):
      python3 ../../Analysis/02_RMSD-RMSF/plot_heatmap.py \
        analysis/Calpha_RMSD.dat \
        -o analysis/plots/02_Calpha_RMSD_heatmap.png
"""
print("See plot_timeseries.py / plot_heatmap.py for CLI usage.")
