This repository contains the Isca source code modifications for the simulations described in the manuscript "Impact of Atmospheric Cloud Radiative Effects on Annular Mode Persistence in Idealized Simulations" , as well as  Python scripts used to analyze the data.

Description of files:

LW_ACRE_regression_SAM.mat - File containing regressions of ACRE on SAM index based on ERA5 and CloudSat data (Matlab data file)

heating_rates.py — Script used to generate the ACRE field based on LW_ACRE_regression_SAM.mat (Python)

CW_SAM_heating.nc —  File containing the diabatic heating field R which mimics ACRE anomalies associated with SAM (NetCDF)

run_Held_Suarez.py — Namelist file used to configure Isca according to the setup of Held-Suarez '94  (Python)

hs_forcing.F90 — Modified Isca source code that adds the time-varying ACRE heating field (Fortran '90)

Isca_output_analysis.py — Analysis script used to read the simulation output and produce the figures presented in the paper (Python)

Eliassen_response.py — Script used to calculate the Eliassen response to a diabatic forcing (Python)



