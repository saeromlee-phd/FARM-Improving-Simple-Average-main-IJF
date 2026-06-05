# FARM-Improving-Simple-Average-main-IJF
Reproducibility Package: Improving the Simple Average Combined Forecast via Factor-Adjusted Regularization

1. GENERAL INFORMATION
----------------------
* Date of Assembly: June 4, 2026
* Author of the reproducibility package: Saerom Lee
* Contact Information: slee839@ucr.edu

2. COMPUTING ENVIRONMENT
------------------------
* Language/Software: MATLAB
* Version: '24.2.0.2923080 (R2024b) Update 6'
* Required Toolboxes: Statistics and Machine Learning Toolbox 
* Hardware/OS Used for Testing:
  - OS: Windows 11 Pro
  - CPU: AMD Ryzen 5 4500U with Radeon Graphics (2.38 GHz)
  - RAM: 8.00 GB

3. REPOSITORY STRUCTURE
-----------------------
├── Combined Forecasts_Main Paper.m        (Main script for Table 1)
├── Combined Forecasts_Online Appendix.m    (Main script for Table A1)
├── README.txt                              (This file)
│
├── [Data & Output Files]
│   ├── 2024-12-no[TARGET].xlsx             (Predictors; e.g., 2024-12-noCPIAUCSL.xlsx)
│   ├── 2024-12-[TARGET].xlsx               (Targets; e.g., 2024-12-CPIAUCSL.xlsx)
│   ├── Results_[TARGET]1.xlsx              (Output for Table 1; e.g., Results_CPIAUCSL1.xlsx)
│   └── Appendix_Results_[TARGET]1.xlsx     (Output for Table A1; e.g., Appendix_Results_CPIAUCSL1.xlsx)
│       * Note: Covers 5 targets (CPIAUCSL, RPI, PCEPI, INDPRO, UNRATE)
│
└── [Helper Functions] (Auxiliary MATLAB functions called by main scripts)
    └── compute_msfe.m, compute_rmsfe.m, forecasts_eBoosting_withc.m, 
        forecasts_epostALASSO_withc.m, forecasts_epostLASSO_withc.m, 
        forecasts_eRidge_withc.m, forecasts_FARM_withc.m, forecasts_FARM1_withc.m, 
        kfoldcv_SPCA.m, mtrans.m, ol1.m, ol2.m, panelFactorNew.m, pathl1.m, 
        pathl1_l.m, pathl1_ns.m, pathl2.m, pathl2_l.m, sel_reg_b_new.m, 
        select_lambda_lasso.m, select_lambda_ridge.m, supervisedPCA.m, ul1.m


4. DATA DESCRIPTION
-------------------
This reproducibility package contains the datasets and code required to replicate the empirical application of forecasting macroeconomic variables presented in the paper.

[Included Data Files]
The repository includes Excel files for 5 different forecast targets (CPIAUCSL, RPI, PCEPI, INDPRO, UNRATE):
* Predictor Files: 2019-12-No[TARGET].xlsx (e.g., 2019-12-NoCPIAUCSL.xlsx)
  - Contains the predictor variables.
* Target Files: 2019-12-[TARGET].xlsx (e.g., 2019-12-CPIAUCSL.xlsx)
  - Contains the forecast target variable.

[Data Origin & Access]
* Source: The raw datasets are sourced from the FRED-MD database provided by the Federal Reserve Bank of St. Louis.
* Alternative Access: Data can be downloaded directly from the St. Louis Fed 
  FRED-MD Repository (https://research.stlouisfed.org/econ/mccracken/fred-databases/).
* Usage Restrictions: Publicly available for research purposes.


5. CODE & REPLICATION INSTRUCTIONS
----------------------------------

[Main Paper Results (Table 1)]
* Script to Run: Combined Forecasts_Main Paper.m
* Process:
  1. Loads the dataset (Default: 2019-12-NoCPIAUCSL.xlsx for predictors and 2019-12-CPIAUCSL.xlsx for the target).
  2. Computes the forecast results.
  3. Saves the resulting forecasts, relative MSFE, and DM test p-values into an Excel file.
* Expected Runtime: Approximately 4 hours per variable for each forecast horizon (h) given the baseline hardware specifications.
* Replicating Other Targets: 
  To replicate the results for other target variables (RPI, PCEPI, INDPRO, UNRATE), modify the input file names at the beginning of the script and update the output file  
  name at the end of the script before execution.

[Online Supplementary Appendix Results (Table A1)]
* Script to Run: Combined Forecasts_Online Appendix.m
* Process:
  1. Loads the same predictor and target data files.
  2. Computes the results specifically for the FARM2 and FARM3 models.
  3. Saves the forecasts, relative MSFE, and p-values into an Excel file.
* Expected Runtime: Approximately 3 hours per variable for each forecast horizon (h).

NOTE: Both scripts contain detailed inline comments describing each algorithmic step for ease of verification.
================================================================================
