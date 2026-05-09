Author: Evan Sims (ems452@cornell.edu)
Instructor: Dr. Stefano Galelli
CEE 6400 Final Project

This is a repository for the code from my final project called "Comparing Algorithms for Battery Energy Arbitrage".

Data:
The data used is from the GridStatus library in Python. All the necessary data is pulled into each code file at the beginning, so there's no
separate data files to include. 

Code:
The code is split into three files. 
1) final_proj_model.py --> This is the main file. It contains the baseline algorithms and pulls in the results from DPS and MPC for the
   ultimate comparison/visualization of all five algorithms. 
2) mpc.py --> This is where the perfect and imperfect MPC algorithms are written and run. It produces full timeseries outputs of the
   cumulative profits for each MPC run on 2025 data. These outputs are saved locally to be pulled into the main file for comparison to
   the other algorithms. 
3) dps_optimizer.py --> This is where the DPS occurs offline to calculate the threshold parameters. These values are saved locally for
   import into the main file for comparison. These parameters are then run in the simulation within the main file to get the cumulative
   profits for DPS.

All locally stored parameters or data are in .csv format, with figures in .png format. The file pathnames have been edited to be follow the 
relative path method, so take note the directory the code in run in to avoid errors.
