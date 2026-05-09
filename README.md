# plvq-svi-interpretability
Reference code for the study:  "The Impact of Visual Elements in Street View on Plant Landscape Visual Quality: A Quantitative Study Based on Deep Learning, GWPCA, and SHAP"

Step 1: Before running, ensure that the mmcv and mmengine libraries are correctly installed. Modify the data root directory and model weight storage paths in the script to match your local environment.

Step 2: This script relies on the timm and optuna libraries. It is recommended to use a GPU to accelerate the Bayesian optimization process when running in --mode optimize. In prediction mode, ensure the model weight filename matches the one defined in the script.

Step 3: An R environment is required. Please install spatial analysis packages such as GWmodel and sp beforehand. The input Excel data must include X and Y coordinate columns to enable geographically weighted calculations.

Step 4: This script compares eight machine learning algorithms. The script will terminate if libraries such as catboost, xgboost, or lightgbm are missing. Using a conda environment for centralized package management is highly recommended.

Step 5: SHAP analysis is computationally intensive. When generating dependency plots, the script automatically calculates LOWESS smoothing curves and SHAP-zero crossing points. Ensure sufficient memory is available when processing large datasets.

Step 6: This step requires the installation of the PyALE library. The generated ALE 1D plots include 95% confidence intervals and rug plots to indicate sample density.

Step 7: For second-order interaction analysis, the ALEPlot and fields packages must be installed in the R environment. 
