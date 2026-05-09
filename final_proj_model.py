#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 21:04:18 2026

@author: evansims

CEE 6400 Final Project:
Control Algorithm Comparison for a Battery Energy Storage System
"""

#%% Imports

import gridstatus as gs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
price_data = nyiso.get_lmp(start=start_train, 
                                      end=end_test, 
                                      market="REAL_TIME_5_MIN", 
                                      locations=["N.Y.C."])

# Subset the data set to include only time, location and LMP columns
filtered_data = price_data[['Time', 'Location', 'LMP']]
print(filtered_data)

# Filter training period data
calib_data = filtered_data[filtered_data["Time"] <= end_train].copy()

# Filter test period data
test_data = filtered_data[filtered_data["Time"] > end_train].copy()

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
        self.S_max = 0.9 * size #MWh
        self.S_min = 0.1 * size #MWh
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
    
        
#%% Policy 1 - Fixed Schedule
"""
Battery will follow a fixed charging schedule as follows:
    Discharge from 5-9pm each day
    Charge from 12-4am each day

Returns: an action for how much to charge or discharge to the grid
    
This is meant to be a baseline policy that is supposed to be the simplest way
you could operate a battery for energy arbitrage.
"""
class Constant_Schedule:    
    def action(self, battery, soc, price, datetime):        
        # If the time is between midnight and 5am, charge at full possible power
        if datetime.time().hour < 4:
            action = battery.max_charge(soc)
            
        # If the time is between 5pm and 10pm, discharge at full possible power
        elif datetime.time().hour >= 17 and datetime.time().hour < 21:
            action = -battery.max_discharge(soc)
        # Otherwise hold idle
        else:
            action = 0
            
        return action

#%% Policy 2 - Greedy Heuristic
"""
Battery will charge according to the price of electricity as follows:
    - If LMP is less than 45th percentile price, then charge at full power
    - If LMP is greater than or equal to 55th percentile price, then discharge at full power
    
This is meant to be another baseline policy that is quite simple but slightly 
more dynamic than the static schedule in policy #1
"""

# Calculate the percentiles of calibration LMP data to get price setpoints
lower_T = calib_data['LMP'].quantile(0.45)
median_price = calib_data['LMP'].median()
upper_T = calib_data['LMP'].quantile(0.55)

print(f"45th Percentile: ${lower_T:.2f}")
print(f"Median (50th Percentile): ${median_price:.2f}")
print(f"55th Percentile: ${upper_T:.2f}")

class Greedy_Heuristic:    
    def action(self, battery, soc, price, datetime):
        # If the price is below the 45th percentile LMP value of training data
        # attempt to charge at full power
        if price < lower_T:
            action = battery.max_charge(soc)
            
        # If the price is above the 55th percentile LMP value of training data
        # attempt to discharge at full power
        elif price >= upper_T:
            action = -battery.max_discharge(soc)
            
        #Otherwise hold the battery idle
        else:
            action = 0
                
        return action


#%% Policy 3 - Direct Policy Search
"""
This DPS algorithm will look to set the optimal charge and discharge price
threshold for the battery, based on the NYISO data

The optimization for this DPS occurs in the file dps_optimizer.py
"""
class DPS_Policy:
    def __init__(self, T1, T2):
        self.T1 = T1
        self. T2 = T2
        
    def action(self, battery, soc, price, datetime):
        # If the price is below T1, attempt to discharge at full power
        if price < self.T1:
            action = battery.max_charge(soc)
            
        # If the price is above T2, attempt to discharge at full power
        elif price >= self.T2:
            action = -battery.max_discharge(soc)
            #action = max(0, action)
            
        #Otherwise hold the battery idle
        else:
            action = 0
                
        return action
    
# Import results from DPS optimization script
opt_dps_params = pd.read_csv('/Users/evansims/Documents/05_School/Senior/CEE 6400/Final Project/optimal_dps_params.csv')

# Extract values from csv
optimal_T1 = opt_dps_params["T1"][0]
optimal_T2 = opt_dps_params["T2"][0]
    

#%% Policy 4 - MPC
"""
This MPC algorithm will look to set the optimal charge and discharge price
threshold for the battery, based on the NYISO data

The optimization for this MPC occurs in the file mpc.py
"""
# Import the MPC results
obj_perf_mpc = pd.read_csv("/Users/evansims/Documents/05_School/Senior/CEE 6400/Final Project/perf_mpc_cum_profits.csv")        
obj_imperf_mpc = pd.read_csv("/Users/evansims/Documents/05_School/Senior/CEE 6400/Final Project/imperf_mpc_cum_profits.csv")        


   
#%% Simulation
"""
Inputs: 
    - LMP lectiricty price data formulated above
    - An initial battery state (say 50%)
    - A given policy (CLASS) defined above to govern action selection
        - (It is expected that all feasibility constraints and limits on the 
           action are already applied within the policy class, returning only a
           feasible, final action.)

Returns: a set of arrays with values for each timestep in timeseries including:
    - State of charge
    - Cumulative profits
    - Time
    - Price timeseries

"""

def run_simulation(data, initial_soc, policy):
    bess = BESS()
    policy = policy
    soc_sim = [initial_soc]
    obj_sim = []
    time_sim = []
    price_sim = []
    profit = 0

    # For each timestep in the data
    for i, row in data.iterrows():
        #print("Iteration: ", i)
        # Update the current state
        current_soc = soc_sim[-1]
        
        # Identify the electricity price at the present timestep
        price = row["LMP"]
        # Identify the datetime at present timestep
        time = row["Time"]
        
        # Determine the action that should be taken according to a given policy
        # Returned action is expected to already have constraints/limits applied within policy
        action = policy.action(bess, current_soc, price, time)
        #print("action = ", action)
        
        # Calculate the next state, battery limits enforced within step_trans()
        next_soc = bess.step_trans(current_soc, action)
        #print("soc =", next_soc)
        
        # Calculate the reward ($/MWh * timestep length in hours * -power rate)
        # Note the negative is becasue + action was defined as charing the battery
        # which becomes a cost to the operator, discharging is revenue
        reward = price * bess.L * -action
        #print("reward = ", reward)

        
        # Adjust total profit so far with current timestep reward
        profit += reward
        
        # Add SOC, cumulative profit and time to their lists
        soc_sim.append(next_soc)
        obj_sim.append(profit)
        time_sim.append(time)
        price_sim.append(price)
        
    return soc_sim[1:], obj_sim, time_sim, price_sim

soc_const, obj_const, time, prices = run_simulation(test_data, 50, Constant_Schedule())
soc_greedy, obj_greedy, time, prices = run_simulation(test_data, 50, Greedy_Heuristic())
soc_dps, obj_dps, time, prices = run_simulation(test_data, 50, DPS_Policy(T1=optimal_T1, T2=optimal_T2))

#%% Visualization and Comparison

# Create a 3 panel plot to comapre algorithm results
fig, (ax_price, ax_profit) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# Plot 1: LMP - Electricity Price ($/MWh)
ax_price.plot(time, prices, color='black', label='NYC LMP ($/MWh)')
ax_price.set_ylabel("Price ($/MWh)")
ax_price.set_title("Control Algorithm Comparison: Market Price and Cumulative Profit")
ax_price.grid(axis='y', alpha=0.3)
ax_price.legend(loc='upper left')

# Plot 2: State of Charge (SoC) %
#ax_soc.plot(time, soc_const, label="Constant Schedule")
#ax_soc.plot(time, soc_greedy, label="Greedy Heuristic")
#ax_soc.plot(time, soc_dps, label="DPS")

#ax_soc.set_ylabel("SoC (%)")
#ax_soc.set_ylim(-5, 105)
#ax_soc.grid(True, alpha=0.3)
#ax_soc.legend(loc='upper left')

# Plot 3: Cumulative Profits ($)
ax_profit.plot(time, obj_const, label="Constant Schedule")
ax_profit.plot(time, obj_greedy, label="Greedy Heuristic")
ax_profit.plot(time, obj_dps, label="DPS")
ax_profit.plot(time[:len(obj_perf_mpc)], obj_perf_mpc, label="Perfect MPC")
ax_profit.plot(time[:len(obj_imperf_mpc)], obj_imperf_mpc, label="Imperfect MPC")

ax_profit.set_ylabel("Cumulative Profits ($)")
ax_profit.set_xlabel("Time (YYYY-MM)")
ax_profit.grid(True, alpha=0.3)
ax_profit.legend(loc='upper left')

plt.tight_layout()
plt.savefig('/Users/evansims/Documents/05_School/Senior/CEE 6400/Final Project/Plots/algo_comparison.png', bbox_inches='tight', dpi=150)

plt.show()

print("Cumulative Profits after 1 year by Algorithm:\n")
print(f"Constant Schedule: {obj_const[-1]}")
print(f"Greedy Heuristic: {obj_greedy[-1]}")
print(f"Direct Policy Search {obj_dps[-1]}")
print(f"MPC Imperfect Forecast: {obj_imperf_mpc.iloc[-1]}")
print(f"MPC Perfect Forecast: {obj_perf_mpc.iloc[-1]}")

#%% Zoomed in Visualization

# Define the 2-week slice indices (Mid-January)
start_idx = 3820
end_idx = 7975
#start_idx = 0
#end_idx = 8000

# Create a 2 panel plot to compare algorithm results over the zoomed period
fig, (ax_price, ax_profit) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Plot 1: LMP - Electricity Price
ax_price.plot(time[start_idx:end_idx], prices[start_idx:end_idx], color='black', label='NYC LMP ($/MWh)')
ax_price.set_ylabel("Price ($/MWh)", fontsize=11)
ax_price.set_title("Algorithm Comparison: Mid-January Volatility (2-Week Period)", fontsize=14, fontweight='bold')
ax_price.grid(axis='y', alpha=0.3)
ax_price.legend(loc='upper left')

# Plot 2: Cumulative Profits
ax_profit.plot(time[start_idx:end_idx], obj_const[start_idx:end_idx], label="Constant Schedule")
ax_profit.plot(time[start_idx:end_idx], obj_greedy[start_idx:end_idx], label="Greedy Heuristic")
ax_profit.plot(time[start_idx:end_idx], obj_dps[start_idx:end_idx], label="DPS")
ax_profit.plot(time[start_idx:end_idx], obj_perf_mpc[start_idx:end_idx], label="Perfect MPC")
ax_profit.plot(time[start_idx:end_idx], obj_imperf_mpc[start_idx:end_idx], label="Imperfect MPC")

ax_profit.set_ylabel("Cumulative Profits ($)", fontsize=11)
ax_profit.set_xlabel("Time (YYYY-MM-DD)", fontsize=11)
ax_profit.grid(True, alpha=0.3)
ax_profit.legend(loc='upper left')

plt.tight_layout()
plt.savefig('/Users/evansims/Documents/05_School/Senior/CEE 6400/Final Project/Plots/zoomed_algo_comparison.png', bbox_inches='tight', dpi=150)
plt.show()



#%% troubleshooting

