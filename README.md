This repository contains the Isca source code modifications for the simulations described in the manuscript "Impact of Atmospheric Cloud Radiative Effects on Annular Mode Persistence in Idealized Simulations" , as well as  Python scripts used to analyze the data.

Description of files:

CW_SAM_heating.nc —  NetCDF file containing the diabatic heating field R which mimics ACRE anomalies associated with SAM

run_Held_Suarez.py — Namelist file used to configure Isca according to the setup of Held-Suarez '94

Held_Suarez_forcing.py — Modified Isca source code that adds the time-varying ACRE heating field 

Isca_output_analysis.py — Analysis script used to read the simulation output and produce the figures presented in the paper


