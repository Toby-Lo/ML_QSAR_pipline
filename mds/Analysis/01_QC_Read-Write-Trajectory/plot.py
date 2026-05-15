#!/usr/bin/env python3
"""Quick plotting entry for this analysis folder.

Typical usage (run from your simulation directory):
cd runs/a1aom/
  - Time series:
      python3 ../../Analysis/01_QC_Read-Write-Trajectory/plot_timeseries.py \
        analysis/QC_Calpha_RMSD.dat \
        -o analysis/plots/01_QC_Calpha_RMSD_timeseries_a1a0m.png

  - Heatmaps (2D-RMSD/DCCM):
      python3 ../../Analysis/01_QC_Read-Write-Trajectory/plot_heatmap.py <matrix_file> -o analysis/plots/01_QC_Calpha_RMSD_heatmap_a_18198196.png

      python3 ../../Analysis/01_QC_Read-Write-Trajectory/plot_heatmap.py \
        analysis/QC_Calpha_RMSD.dat \
        -o analysis/plots/01_QC_Calpha_RMSD_heatmap_a_200ns.png \
        --zlabel "Distance (Å)" \
        --dt 0.002 --ntwx 5000 --stride 10
""" 
print("See plot_timeseries.py / plot_heatmap.py for CLI usage.")
