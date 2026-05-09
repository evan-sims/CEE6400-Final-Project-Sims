#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 22:53:29 2026

@author: evansims
*Based on code provided in Homeworks by Stefano Galelli

This is the optimization script for the Policy 3 - DPS for my
CEE6400 final project comparing BESS control algorithms
"""

#%% Import libraries

import gridstatus as gs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize


#%% Electricity Price Data
"""
Pulls in the electricity price data from the isodata library, and selects stores the NYISO data:
    
>isodata provides a uniform API to access energy data from the major Independent System Operators (ISOs) in the United States.
>Currently supports fuel mix, load, supply, load forecast, and LMP pricing data for CAISO, SPP, ISONE, MISO, Ercot, NYISO, and PJM.

>Use nyiso.markets to see available markets
"""

# Import NYISO data from the gridstatus library
nyiso = gs.NYISO()

# Set dates for data
start_train = "Jan 1, 2024"
end_train = "Dec 31, 2024"
end_test = "Dec 31, 2025"

# Get historical data for real time Locational Marginal Price (LMP) at
# 5-minute intervals from the NYC location zones in NYISO from 2024-2025
price_data = nyiso.get_lmp(start= start_train, 
                                      end= end_test, 
                                      market="REAL_TIME_5_MIN", 
                                      locations=["N.Y.C."])

# Subset the data set to include only time, location and LMP columns
filtered_data = price_data[['Time', 'Location', 'LMP']]
print(filtered_data)

initial_soc = 50
#%% Battery Model
"""
This is a class that define the specs of the battery and holds the step transition
function. The step transition function also enforces battery limit constraints
In this project, I will use the following battery parameters:

Battery size = 100 MWh
Charging efficiency = 96%
Storage efficiency = 100%
Max power (dis)charge rate to grid (P_max) = 25 MW
Min power (dis)charge rate to grid (P_min) = 0 MW
Max operational power state (S_max) = 90 MWh
Min operational power state (S_min) = 10 MWh
Timestep length (L) = 5 minutes = 5/60 hours

"""

class BESS:
    def __init__(self, size=100, charge_eff=0.96, storage_eff=1, max_power=25):
        self.e = charge_eff #%
        self.se = storage_eff #%
        self.P_max = max_power #MW
        self.P_min = 0 #MW
        self.S_max = 0.9 * size #MW
        self.S_min = 0.1 * size #MW
        self.L = 5/60 #Hours

    def step_trans(self, current_state, action):
        # Map action to separate discharge (d_t) and charge (c_t) variables
        if action > 0:
            d_t = 0
            c_t = action
        elif action < 0:
            c_t = 0
            d_t = abs(action)
        else:
            d_t = 0
            c_t = 0

        # Compute the next battery state of charge
        next_s = (self.se * current_state) + self.L * ((self.e * c_t) - (1/self.e * d_t))
        
        # Secondary insurance that battery state of charge limits aren't exceeded 
        next_s = min(self.S_max, max(next_s, self.S_min))
    
        # Return the next battery state of charge    
        return next_s

    # Calculate the max possible charge rate given effiency losses and 
    # current soc for each given timestep
    def max_charge(self, soc):
        # Max available energy (MWh)
        available = self.S_max - soc
        # Max possible charge rate during this timestep (MW)
        c_lim = available / self.L / self.e
        # If the charge limit is greater than battery power constraint (P_max), use P_max
        return min(c_lim, self.P_max)
        
    # Calculate the max possible discharge rate given effiency losses and 
    # current soc for each given timestep
    def max_discharge(self, soc):
        # Max available energy (MWh)
        available = max(0, soc - self.S_min)
        # Max possible charge rate during this timestep (MW)
        d_lim = available / self.L * self.e
        # If the discharge limit is greater than battery power constraint (P_max), use P_max
        return min(d_lim, self.P_max)

#%% Calibration Simulation
"""
Inputs: 
    - Charging Threshold (T1)
    - Discharging Threshold (T2)

Returns:
    - Average daily BESS operation profits

"""

# Filter calibration period data 
calib_data = filtered_data[filtered_data["Time"] <= end_train].copy()

def calibration_simulation(T1, T2):
    bess = BESS()
    soc_sim = [initial_soc]
    obj_sim = []
    time_sim = []
    price_sim = []
    profit = 0
    
    # For each timestep in the data
    for row in calib_data.itertuples():
        
        # Update the current state
        current_soc = soc_sim[-1]
        
        # Identify the electricity price at the present timestep
        price = row.LMP
        
        # Identify the datetime at present timestep
        time = row.Time
        
        # If the price is above T1 $/MWh, attempt to discharge at full power
        if price < T1:
            action = bess.max_charge(current_soc)
                
        # If the price is above T2 $/MWh, attempt to discharge at full power
        elif price >= T2:
            action = -bess.max_discharge(current_soc)
                
        # Otherwise hold the battery idle
        else:
            action = 0
                            
        # Calculate the next state, battery limits enforced within step_trans()
        next_soc = bess.step_trans(current_soc, action)
        
        # Calculate the reward ($/MWh * timestep length in hours * -power rate)
        # Note the negative is becasue + action was defined as charing the battery
        # which becomes a cost to the operator, discharging is revenue
        reward = price * bess.L * -action

        # Adjust total profit so far with current timestep reward
        profit += reward
        
        # Add SOC, cumulative profit and time to their lists
        soc_sim.append(next_soc)
        obj_sim.append(profit)
        time_sim.append(time)
        price_sim.append(price)  
    
    
    # Calculate total days in the simulation
    # (24 hours/day * 60 minutes/hour) / 5 minutes/step = 288 steps per day
    ints_per_day = 288
    total_days = len(calib_data) / ints_per_day
    
    # Calculate the average daily profits
    avg_daily_profit = profit / total_days
    
    return profit, avg_daily_profit


#%% DPS problem definition
"""
This is the optimization for the Direct Policy Search algorithm that will 
output the thershold parameters defining how the battery should be operated
"""
 
class DPS_Problem(ElementwiseProblem):
    def __init__(self, **kwargs):
        # 2 decision variables--> T1: max charge thresh, T2: max discharge thresh
        xl = [0, 0] # set lower limit at $0 /MWh
        xu = [300, 300] # set upper limit at $300 /MWh
        super().__init__(n_var=2, n_obj=1, xl=xl, xu=xu, **kwargs)
 
    def _evaluate(self, x, out, *args, **kwargs):
        # x[0] = T1, x[1] = T2
        if x[0] >= x[1]:  # Ensure T1 < T2
            out["F"] = [1e12]  # Penalize infeasible solutions
            return
        # Calculate the cumulative and average daily profits given T1, T2
        profit, avg_daily_profit = calibration_simulation(x[0],x[1])
        
        # Set the objectives (minimization)
        out["F"] = [-profit] # Invert for the minimizer        
        
         
#%% DPS optimization

# algorithm paramets adjusted through trial and error to effectively 
# search the thershold space while eventually converging on a solution
algorithm = NSGA2(
    pop_size=50,
    sampling=FloatRandomSampling(),
    crossover=SBX(prob=0.9, eta=15),
    mutation=PM(prob=0.7, eta=10),
    eliminate_duplicates=True
)

# Create an instance of the problem
problem_inst = DPS_Problem()

# Optimize using calibration data
res = minimize(problem_inst,
               algorithm,
               ('n_gen', 100),
               save_history=True,
               verbose=True)

# Extract Solutions
calibration_profits = res.F

#%% Report the optimal solution from calibration data

# Check how many optimal T1,T2 pairs were found
# If more than one pair found, take the first one
if res.X.ndim > 1:
    
    # Find the index of the  minimum value in the array
    best_idx = np.argmin(res.F)
    
    # Extract the T1 and T2 thresholds using the min index
    best_calib_T1 = res.X[best_idx][0].item()
    best_calib_T2 = res.X[best_idx][1].item()
    
    # Extract the profit and invert it back to positive
    # np.ravel needed in case it returns non-1D shape
    best_calib_profit = -1 * np.ravel(res.F)[best_idx].item()
    
# If only one optimal threshold pair was found, extract it
else:
    best_calib_T1 = res.X[0].item()
    best_calib_T2 = res.X[1].item()
    
    # Extract the profit and invert it back to positive
    best_calib_profit = -1 * res.F[0].item()

# Store the optimal parameter in a dataframe for import into main project script
optimal_df = pd.DataFrame({"T1": [best_calib_T1], "T2": [best_calib_T2]})

# Save it to a CSV
optimal_df.to_csv("/Users/evansims/Documents/05_School/Senior/CEE 6400/Final Project/optimal_dps_params.csv", index=False)

print(f"Optimal Charging Threshold (T1): ${best_calib_T1:.2f}/MWh")
print(f"Optimal Discharging Threshold (T2): ${best_calib_T2:.2f}/MWh")
print(f"Maximum Calibration Profit: ${best_calib_profit:,.2f}")

#%% Validation Simulation
"""
Inputs: 
    - Charging Threshold (T1)
    - Discharging Threshold (T2)

Returns:
    - Average daily BESS operation profits

"""

# Filter calibration period data to be 
valid_data = filtered_data[filtered_data["Time"] > end_train].copy()

def validation_simulation(T1, T2):
    bess = BESS()
    soc_sim = [initial_soc]
    obj_sim = []
    time_sim = []
    price_sim = []
    profit = 0
    
    # For each timestep in the data
    for row in valid_data.itertuples():
        
        # Update the current state
        current_soc = soc_sim[-1]
        
        # Identify the electricity price at the present timestep
        price = row.LMP
        
        # Identify the datetime at present timestep
        time = row.Time
        
        # If the price is above T1 $/MWh, attempt to discharge at full power
        if price < T1:
            action = bess.max_charge(current_soc)
                
        # If the price is above T2 $/MWh, attempt to discharge at full power
        elif price >= T2:
            action = -bess.max_discharge(current_soc)
                
        # Otherwise hold the battery idle
        else:
            action = 0
                            
        # Calculate the next state, battery limits enforced within step_trans()
        next_soc = bess.step_trans(current_soc, action)
        
        # Calculate the reward ($/MWh * timestep length in hours * -power rate)
        # Note the negative is becasue + action was defined as charing the battery
        # which becomes a cost to the operator, discharging is revenue
        reward = price * bess.L * -action

        # Adjust total profit so far with current timestep reward
        # Subtract to fit pymoo minimization
        profit += reward
        
        # Add SOC, cumulative profit and time to their lists
        soc_sim.append(next_soc)
        obj_sim.append(profit)
        time_sim.append(time)
        price_sim.append(price)  
    
    # Calculate total days in the simulation
    ints_per_day = 288
    total_days = len(valid_data) / ints_per_day
    
    # Calculate the average daily profits
    avg_daily_profit = profit / total_days
    
    return profit, avg_daily_profit

# Run the validation simulation
valid_profits, valid_avg_daily_profits = validation_simulation(best_calib_T1, best_calib_T2)

# Print the results
print(f"Validation Cumulative Profit: ${valid_profits:,.2f}")
print(f"Validation Avg Daily Profit: ${valid_avg_daily_profits:,.2f}")
            

#%% Visualization

# Lists to hold the history
t1_all = []
t2_all = []
profits_all = []

# Loop through every generation in the algorithm's history
for step in res.history:
    # Extract the variables (X) and objectives (F) from each generation
    X = step.pop.get("X")
    F = step.pop.get("F")
    
    # Loop through all individuals in this generation
    for i in range(len(X)):
        # Only plot feasible solutions (where T1 < T2)
        if X[i][0] < X[i][1]:
            t1_all.append(X[i][0])
            t2_all.append(X[i][1])
            profits_all.append(-1 * F[i][0])

# Create heatmap of attempted T1, T2
plt.figure(figsize=(10, 8))

# Scatter plot using color scheme to identify profits
scatter = plt.scatter(t1_all, t2_all, c=profits_all, cmap='viridis', alpha=0.7, edgecolors='none')
cbar = plt.colorbar(scatter)
cbar.set_label('Cumulative Profit ($)', rotation=270, labelpad=20)

# Add the infeasible boundary line (where T1 = T2)
plt.plot([0, 300], [0, 300], 'r--', linewidth=2, label="Infeasible Boundary (T1 = T2)")
plt.fill_between([0, 300], [0, 300], 0, color='red', alpha=0.05) # Lightly shade the penalty zone

# Labels and Formatting
plt.title("Search Landscape: BESS Profit by Charge/Discharge Thresholds")
plt.xlabel("Charging Threshold (T1) [$/MWh]")
plt.ylabel("Discharging Threshold (T2) [$/MWh]")
plt.xlim(0, 300)
plt.ylim(0, 300)
plt.grid(True, alpha=0.3)
plt.legend(loc='upper left')

plt.savefig('/Users/evansims/Documents/05_School/Senior/CEE 6400/Final Project/Plots/DPS_heatmap.png')
plt.show()
        
        