#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 17:05:33 2026

@author: evansims
*Based on code provided in lecture by Stefano Galelli

This is the optimization script for the Policy 4 - MPC for my
CEE6400 final project comparing BESS control algorithms

"""

#%% Import libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from numpy.random import default_rng
from statsmodels.tsa.seasonal import STL
import gridstatus as gs
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.arima.model import ARIMA



#%% Import LMP data
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
calib_lmp = calib_data['LMP']

# Filter test period data
test_data = filtered_data[filtered_data["Time"] > end_train].copy()
test_lmp = test_data['LMP']

#%% Parameters

# Battery size, same as assumption from final_proj_model.py
size = 100 #MWh

# Time horizon is 24 hours (12 steps/hour * 24 hrs)
T = 12 * 24

# Number of horizons solved 
# The maximum number of horions that can be solved (total data - horizon length)
N = len(test_data) - T

# SOC constaints, same as assumption from final_proj_model.py
soc_min = 0.1 * size
soc_max = 0.9 * size

# Power (dis)charge constraint
P_max = 25

# Initial conditions
x0 = 50

# Import BESS class for running MPC
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



#%% Helper function: lag matrix
def create_lag_matrix(y, lags):
    n = len(y)
    X = np.zeros((n - lags, lags))
    for i in range(1, lags + 1):
        X[:, i - 1] = y[lags - i:n - i]
    y_out = y[lags:]
    return X, y_out


#%% Set up the MPC 

def run_perfect_MPC(forecasted_lmp, actual_lmp, battery, T, N, x0):
    # Battery info from input, to be consistent with other algorithms
    e = battery.e
    se = battery.se
    P_max = battery.P_max
    P_min = battery.P_min
    S_max = battery.S_max
    S_min = battery.S_min
    L = battery.L
    
    # Store simulated state and applied control
    x_vals = [x0]
    c_vals = []
    d_vals = []
    
    # Store the cumulative profits at each timestep for comparison with other algos
    step_profits = []

    # Define variables outside loop
    x = cp.Variable(T + 1)
    c = cp.Variable(T, nonneg=True)
    d = cp.Variable(T, nonneg=True)
    
    # Create parameters for values that change every timestep
    lmp_param = cp.Parameter(T)
    x_init_param = cp.Parameter()
    
    # Define objective
    terminal_reward = 30
    cost = (lmp_param @ c) - (lmp_param @ d) - (terminal_reward * x[T])
    
    # Build constraints
    constraints = [x[0] == x_init_param]
    for k in range(T):
        constraints += [
            x[k+1] == x[k] + L * ((e * c[k]) - (1/e * d[k])),
            S_max >= x[k],
            S_min <= x[k],
            P_max >= c[k],
            P_max >= d[k],
        ]
    constraints += [S_max >= x[T], S_min <= x[T]]
    
    # Create the problem
    prob = cp.Problem(cp.Minimize(cost), constraints)

    # Loop over the forecast horizon
    for t in range(N):
        # Update the parameters with the current data
        lmp_param.value = forecasted_lmp[t:t+T].values
        x_init_param.value = x_vals[-1]

        # Solve the problem
        prob.solve() 

        # Safety check in case the solver fails to converge
        if prob.status != cp.OPTIMAL:
            print(f"Solver failed at timestep {t}!")
            break

        # Apply only the first optimal control move
        c_apply = c.value[0]
        d_apply = d.value[0]
        
        # Record this control move
        c_vals.append(c_apply)
        d_vals.append(d_apply)

        # Update the real system using (dis)charge action
        x_next = x_vals[-1] + L * ((e * c_apply) - (1/e * d_apply))
        x_vals.append(x_next)
        
        # Calculate actual profit for this 5-min step
        # Grab the TRUE price that actually occurred at this exact moment
        current_true_price = actual_lmp.iloc[t] 
        
        # Profit = (Power Sold - Power Bought) * (Fraction of an Hour) * Price
        cash_flow = (d_apply - c_apply) * L * current_true_price
        
        step_profits.append(cash_flow)

    # Turn the list of individual cash flows into a running total
    cumulative_profits = np.cumsum(step_profits)

    return np.array(x_vals[:-1]), np.array(c_vals), np.array(d_vals), cumulative_profits


#%% Run Perfect Foresight MPC
# Create battery problem
bess = BESS()

# Use the actual LMP data for perfect forecast
perfect_forecast = test_lmp

# True LMP data
actual_lmp = test_lmp

# N used for testing
#N=10

# Run the perfect forecast MPC
x_perf, c_perf, d_perf, perf_cum_profits = run_perfect_MPC(perfect_forecast, actual_lmp, bess, T, N, x0)


np.savetxt("./perf_mpc_cum_profits.csv", perf_cum_profits, delimiter=",", fmt="%d")

#%% Time-series analysis

# Initialize and fit the STL
# period=12*24=288 is the number of 5-minute samples to complete a daily cycle
# robust=True helps prevent overfitting to major price spikes
# seasonal=31 was found to be the sweet spot for finding a consistent, daily 
# pattern without too much overfitting. Lower loses the pattern, higher way overfits
result = STL(calib_lmp, period=288, seasonal=31, robust=True).fit()

# Extract seasonal trend, daily cyclical trend and residuals
seasonal_trend = result.trend
daily_cycle = result.seasonal
residual = result.resid

# Detrend the data of the daily cycle and long-term seasonal changes
detrended_lmp = calib_lmp - daily_cycle - seasonal_trend

# Visualization
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

ax1.plot(calib_lmp, label='Original LMP', color='black', alpha=0.6)
ax1.plot(seasonal_trend, label='Long-term Trend', color='red', linewidth=2)
ax1.legend(loc='upper left')
ax1.set_ylim(-20,100)
ax1.set_xlim(40000,48016)

ax2.plot(daily_cycle+seasonal_trend, label='Combined Trend (Long-term Trend + Residual)', color='blue')
ax2.legend(loc='upper left')
ax2.set_ylim(-20,100)
ax2.set_xlim(40000,48016)

ax3.plot(daily_cycle, label='Pure Daily Cycle', color='green')
ax3.legend(loc='upper left')
ax3.set_ylim(-20,20)
ax3.set_xlim(40000,48016)

ax4.plot(residual, label='Detrended Data (Residuals)', color='purple', alpha=0.5)
ax4.legend(loc='upper left')
ax4.set_ylim(-20,100)
ax4.set_xlim(40000,48016)

plt.tight_layout()
plt.show()


#%% Creating Forecasts with AR()

# List to store resulting AR coefficients
AR_values = []

# Calculate each from AR(1)-AR(3)
for i in np.arange(1,4):
    # Calculate the lagged matrix (A) for a given order for each price value (b) 
    A, b = create_lag_matrix(detrended_lmp.values, i)
    
    # Add a constant to be fit
    A_with_const = sm.add_constant(A) 
    
    # Fit OLS
    OLS_model = sm.OLS(b, A_with_const)
    fit_OLS = OLS_model.fit()
    
    print(f"AR({i}) results: {fit_OLS.params}")
    print(f"AR({i}) results: {fit_OLS.summary()}")

    AR_values.append(fit_OLS.params)
    
    # Calculate residuals
    residuals = fit_OLS.resid

    # Plot the ACF of residuals
    acf_resid = plot_acf(residuals, lags=24)
    plt.title(f"AR{i} ACF of Residuals")
    plt.xlabel("Lag Hours")
    plt.ylabel("Residual Auto-correlation")
    plt.show()


#%% Plot AR for troubleshooting

# Time array for plotting
time = calib_lmp.index.values
time_data = filtered_data["Time"]


# Arrays to hold predictions for each AR(n)
AR1_pred = []
AR2_pred = []
AR3_pred = []


# Calculate the one-step predictions
for j in range(3, len(detrended_lmp)):  
    AR1_pred.append(AR_values[0][0] +
                    detrended_lmp[j-1] * AR_values[0][1])
    AR2_pred.append(AR_values[1][0] +
                    detrended_lmp[j-1] * AR_values[1][1] +
                    detrended_lmp[j-2] * AR_values[1][2])
    AR3_pred.append(AR_values[2][0] +
                    detrended_lmp[j-1] * AR_values[2][1] + 
                    detrended_lmp[j-2] * AR_values[2][2] + 
                   detrended_lmp[j-3] * AR_values[2][3])


# Plot actual vs each of the AR models
plt.figure(figsize=(15,4))
plt.plot(time[3:], calib_lmp[3:], label = "Actual LMP")
#plt.plot(time[3:], AR1_pred, label = "AR1")
#plt.plot(time[3:], AR2_pred, label = "AR2")
plt.plot(time[3:], AR2_pred+daily_cycle[3:]+seasonal_trend[3:], label = "AR2+Trends")
#plt.plot(time[3:], AR2_pred+daily_cycle[3:], label = "AR2+Trends")
#plt.plot(time[3:], AR3_pred+daily_cycle[3:], label = "AR3")
plt.ylabel("Detrended LMP Data")
plt.xlabel("Time")
plt.xlim(10000, 13000)
plt.ylim(0, 200)
plt.title("Step-Ahead Predictions vs Actual")
plt.legend()
plt.show()

# Selecting AR(2) based on the above tests, showing little improvement beyond 2
AR2_const = AR_values[1][0]
AR2_phi1 = AR_values[1][1]
AR2_phi2 = AR_values[1][2]

# Recombine into one array
AR_coeffs = [AR2_const, AR2_phi1, AR2_phi2]


#%% Run Imperfect prediction MPC

def run_imperfect_MPC(actual_lmp, daily_cycle, AR_coeffs, battery, T, N, x0):
    
    # Battery info from input, to be consistent with other algorithms
    e = battery.e
    se = battery.se
    P_max = battery.P_max
    P_min = battery.P_min
    S_max = battery.S_max
    S_min = battery.S_min
    L = battery.L
    
    # Store simulated state and applied control
    x_vals = [x0]
    c_vals = []
    d_vals = []
    
    # Store the cumulative profits at each timestep for comparison with other algos
    step_profits = []

    # Define variables outside loop
    x = cp.Variable(T + 1)
    c = cp.Variable(T, nonneg=True)
    d = cp.Variable(T, nonneg=True)
    
    # Create parameters for values that change every timestep
    lmp_param = cp.Parameter(T)
    x_init_param = cp.Parameter()
    
    # Define objective
    terminal_reward = 30
    cost = (lmp_param @ c) - (lmp_param @ d) - (terminal_reward * x[T])
    
    # Build constraints
    constraints = [x[0] == x_init_param]
    for k in range(T):
        constraints += [
            x[k+1] == x[k] + L * ((e * c[k]) - (1/e * d[k])),
            S_max >= x[k],
            S_min <= x[k],
            P_max >= c[k],
            P_max >= d[k],
        ]
    constraints += [S_max >= x[T], S_min <= x[T]]
    
    # Create the problem
    prob = cp.Problem(cp.Minimize(cost), constraints)

    # Create the imperfect forecast
    # Start the loop at 288 so the battery has 24 hours of history to look at
    for t in range(288, N): 
                
        # Calculate Trend with a 24-hour moving average of actual prices
        current_trend = np.mean(actual_lmp.iloc[t-288 : t])
        
        # Extract the newest actual residuals
        # Residual = Actual - Lont-term Trend - Daily Cycle
        # t-1 is 5 minutes ago, t-2 is 10 minutes ago
        prev1_residual = actual_lmp.iloc[t-1] - current_trend - daily_cycle[(t-1) % 288]
        prev2_residual = actual_lmp.iloc[t-2] - current_trend - daily_cycle[(t-2) % 288]

        # Build the forecast horizon of length T      
        lmp_horizon = np.zeros(T)
        
        # Use AR coefficients
        const = AR_coeffs[0]
        phi1 = AR_coeffs[1]
        phi2 = AR_coeffs[2]
        
        for k in range(T):
            # Predict the next fading residual
            next_res = const + (phi1 * prev1_residual) + (phi2 * prev2_residual)
            
            # Calculate the yesterday's daily cycle for this future step expectation
            # Use % 288 to loop back to the start of the daily profile
            future_daily = daily_cycle[(t + k) % 288]
            
            # Recombine Forecast = Long-term Trend + Daily Cycle + Predicted Residual
            lmp_horizon[k] = current_trend + future_daily + next_res
            
            # Step forward to next loop iteration
            prev2_residual = prev1_residual
            prev1_residual = next_res

        # Update the parameters with the current data
        lmp_param.value = lmp_horizon
        x_init_param.value = x_vals[-1]        

        # Solve the problem
        prob.solve() 
    
        # Safety check in case the solver fails to converge
        if prob.status != cp.OPTIMAL:
            print(f"Solver failed at timestep {t}!")
            break
        
        # Apply only the first optimal control move
        c_apply = c.value[0]
        d_apply = d.value[0]
            
        # Record this control move
        c_vals.append(c_apply)
        d_vals.append(d_apply)

        # Update the real system using (dis)charge action
        x_next = x_vals[-1] + L * ((e * c_apply) - (1/e * d_apply))
        x_vals.append(x_next)
            
        # Calculate actual profit for this 5-min step
        # Grab the true price that actually occurred at this  moment
        current_true_price = actual_lmp.iloc[t]
            
        # Profit = (Power Sold - Power Bought) * (Fraction of an Hour) * Price
        cash_flow = (d_apply - c_apply) * L * current_true_price
        step_profits.append(cash_flow)

    # Turn the list of individual cash flows into a running total
    cumulative_profits = np.cumsum(step_profits)
        
    return np.array(x_vals[:-1]), np.array(c_vals), np.array(d_vals), cumulative_profits


#%% Simulate Imperfect MPC
# Create new battery problem
bess = BESS()

# True LMP data
actual_lmp = test_lmp

# Extract one day of the daily cycle to use in the MPC model
# Select day for index 46080 (day 160) as the best daily cycle fit (visually selected)
one_daily_cycle = np.array(daily_cycle)[46080 : 46080 + 288]


# Run imperfect forecast MPC
x_imperf, c_imperf, d_imperf, imperf_cum_profits = run_imperfect_MPC(actual_lmp, one_daily_cycle, AR_coeffs, bess, T, N, x0)

np.savetxt("./imperf_mpc_cum_profits.csv", imperf_cum_profits, delimiter=",", fmt="%d")


# AR(2) + STL predictive forecast, Must start at timestep 3 or later!

def get_lmp_prediction(detrended_lmp, t):
    AR_pred = []
    for j in range(timestep, timestep+25):
        lmp_ar = (AR_values[1][0] +
                    detrended_lmp[j-1] * AR_values[1][1] +
                    detrended_lmp[j-2] * AR_values[1][2])
        AR_pred.append(lmp_ar)
            
    
        # Long-term Trend - Assume the current trend stays perfectly flat
        current_trend = trend_data[t-1]
        trend_forecast = np.full(T, current_trend)

        # Daily Cycles - Copy the exact curve from 24 hours ago
        seasonal_forecast = seasonal_data[t-288 : t-288+T].values

        # AR(2) Residuals
        ar_forecast = np.zeros(T)
        resid_lag1 = detrended_lmp[t-1]  # 5 minutes ago residual
        resid_lag2 = detrended_lmp[t-2]  # 10 minutes ago residual
        
        # Pull your AR coefficients
        c = AR2_const
        phi1 = AR2_phi1
        phi2 = AR2_phi2
        
        for i in T:
            next_t_pred = c + (phi1 * resid_lag1) + (phi2 * resid_lag2)
            ar_forecast[i] = next_t_pred
            resid_lag1 = next_t_pred
            resid_lag2 = resid_lag1
            
        lmp_pred = trend_forecast + seasonal_forecast +  seasonal_forecast

    
#x_pred, c_pred, d_pred = run_MPC(, true_demand)
