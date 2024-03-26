This repository contains the Isca source code modifications for the simulations described in the manuscript "Impact of Atmospheric Cloud Radiative Effects on Annular Mode Persistence in Idealized Simulations" , as well as  Python scripts used to analyze the data.

Description of files:


heating_rates.py — Script used to generate the ACRE field based on ERA5 and CloudSat data

CW_SAM_heating.nc —  NetCDF file containing the diabatic heating field R which mimics ACRE anomalies associated with SAM

run_Held_Suarez.py — Namelist file used to configure Isca according to the setup of Held-Suarez '94  (Python)

hs_forcing.F90 — Modified Isca source code that adds the time-varying ACRE heating field (Fortran '90)

Isca_output_analysis.py — Analysis script used to read the simulation output and produce the figures presented in the paper (Python)



