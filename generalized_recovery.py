'''

Applied Risk and Assetmanagement
Generalized Recovery Model

Andreas Foitzik
'''


import numpy as np
import pandas as pd
import statsmodels.api as sm

from scipy.optimize import minimize

import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')

# ---------------- Import Data ----------------
eurostoxx50_qvar_riskfree   = pd.read_csv("eurostoxx50_qvar_riskfree.csv", sep = ';')
eurostoxx50_qvar_riskfree.head()

eurostoxx50_qvar_riskfree_all   = pd.read_csv("eurostoxx50_qvar_riskfree_all.csv", sep = ';')
eurostoxx50_qvar_riskfree_all.head()

eurostoxx50_rnd              = pd.read_csv("eurostoxx50_rnd.csv",  sep = ';')
eurostoxx50_rnd              = eurostoxx50_rnd[eurostoxx50_rnd['daystomaturity'] <= 365]
eurostoxx50_rnd['moneyness'] = eurostoxx50_rnd['implStrike'] / eurostoxx50_rnd['underlyingforwardprice']
eurostoxx50_rnd['moneyness'] = eurostoxx50_rnd['moneyness'].round(decimals = 3)
eurostoxx50_rnd.head()

# ---------------- Prepare Data ----------------
current_VIX                 = np.sqrt(eurostoxx50_qvar_riskfree['bakshiVariance'][0] * (365/30))

# as state we use the current return, which is 1
lowest_State        = 1 - 2.5 * current_VIX
highest_State       = 1 + 4 * current_VIX
teta_amount         = 11

print("\n")
print("------------ DATA SETUP ------------")
print("Current VIX:     ", current_VIX)
print("Lowest State:    ", lowest_State)
print("Highest State:   ", highest_State)
print("------------------------------------")
print("\n")

# ---------------- Arrow-Debreu-Prices ----------------
eurostoxx50_rnd             = eurostoxx50_rnd[(eurostoxx50_rnd['moneyness'] >= lowest_State)
                                              & (eurostoxx50_rnd['moneyness'] <= highest_State)
                                              & (eurostoxx50_rnd['daystomaturity'] <= 365)]

eurostoxx50_rnd             = pd.merge(eurostoxx50_rnd, eurostoxx50_qvar_riskfree_all[['daystomaturity', 'riskfree']], on = 'daystomaturity')
eurostoxx50_rnd['ad']       = eurostoxx50_rnd['rndStrike'] * np.exp(-eurostoxx50_rnd['riskfree'] * eurostoxx50_rnd['daystomaturity'] / 365)
eurostoxx50_rnd['maturity'] = eurostoxx50_rnd['daystomaturity'] / 365

days_to_maturity            = np.array(eurostoxx50_rnd['daystomaturity'].unique())
days_to_maturity            = np.sort(days_to_maturity)
maturities                  = np.matrix(eurostoxx50_rnd['maturity'].unique())
maturities                  = np.sort(maturities)

states                      = np.matrix(eurostoxx50_rnd['moneyness'].unique())
riskfree_rates              = np.matrix(eurostoxx50_rnd['riskfree'].unique())

amount_of_time_horizons     = maturities.shape[1]
amount_of_states            = states.shape[1]

pi                          = np.zeros(shape = (amount_of_time_horizons, amount_of_states))

for time_horizon in range(0, amount_of_time_horizons):
    pi_maturities = np.matrix(eurostoxx50_rnd[(eurostoxx50_rnd['maturity'] == maturities[0, time_horizon])]['ad'])
    if pi_maturities.shape[1] == 351:
        pi_maturities = np.append(pi_maturities, [[0]], 1)
    elif pi_maturities.shape[1] == 352:
        pi_maturities[0, 351] = 0
    pi[time_horizon, :] = pi_maturities[0]

print("------------- ARROW-DEBREU-PRICE MATRIX -------------")
print("Arrow-Debreu-Price Matrix")
print("Amount of States:            ", amount_of_states)
print("Amount of Time Horizons:     ", amount_of_time_horizons)
print("Arrwow-Debreu-Price Matrix:  ", pi)
print("-----------------------------------------------------")

# ---------------- Closed-Form Recovery ----------------
alpha                       = np.zeros(amount_of_time_horizons)
beta                        = np.zeros(amount_of_time_horizons)
delta                       = np.zeros(amount_of_time_horizons)

for time_horizon in range (0, amount_of_time_horizons):
    maturity                = maturities[0, time_horizon]
    riskfree_rate           = np.matrix(eurostoxx50_rnd[(eurostoxx50_rnd['maturity'] == maturity)]['riskfree'])[0,0]
    delta_zero              = 1 - riskfree_rate
    alpha[time_horizon]     = -(maturity - 1) * (delta_zero ** maturity)
    beta[time_horizon]      = maturity * (delta_zero ** (maturity - 1))
    delta[time_horizon]     = alpha[time_horizon] + beta[time_horizon] * delta_zero

print("Alpha:", alpha)
print("Beta:", beta)
print("Delta:", delta)

# ---------------- 10 State-Spaces ----------------
state_spaces       = []
state_space        = np.zeros((10, 2))

state_space[0, 0]               = 1 - 2.5 * current_VIX
state_space[0, 1]               = 1 - 2 * current_VIX
state_spaces.append(np.matrix(states[(states >= state_space[0, 0]) & (states < state_space[0, 1])]))

state_space_equal_portion_from  = 1 - 2 * current_VIX
state_space_equal_portion_to    = 1 + 2 * current_VIX
portion                         = (state_space_equal_portion_to - state_space_equal_portion_from) / 8

print("\n")
print("--- STATE SPACE --- ")
print("0 FROM ", state_space[0, 0], " TO ", state_space[0, 1])

for i in range (1, 9):
    state_space[i, 0]           = state_space[i-1, 1]
    state_space[i, 1]           = state_space[i, 0] + portion
    state_spaces.append(np.matrix(states[(states >= state_space[i, 0]) & (states < state_space[i, 1])]))
    print(i, " FROM ", state_space[i, 0], " TO ", state_space[i, 1])

state_space[9, 0]               = 1 + 2 * current_VIX
state_space[9, 1]               = 1 + 4 * current_VIX
state_spaces.append(np.matrix(states[(states >= state_space[9, 0]) & (states < state_space[9, 1])]))

print("9 FROM ", state_space[9, 0], " TO ", state_space[9, 1])
print("--------------------")
print("\n")

#amount of states still need to be 352

# ---------------- Design-Matrix B ----------------
B                       = np.zeros((states.shape[1], teta_amount))
amount_of_states_set    = 0

# Design Matrix Level
for i in range (0, amount_of_states):
    B[i, 0]     = 1.00

# Design Matrix State Space 1
for i in range (0, state_spaces[0].shape[1]):
    B[i, 1]     = (i+1) / state_spaces[0].shape[1]

amount_of_states_set   += state_spaces[0].shape[1]
for i in range (amount_of_states_set, amount_of_states):
    B[i, 1]     = 1

# Design Matrix for State Space 2 till 10
for a in range (1, len(state_spaces)):
    for i in range (0, amount_of_states_set):
        B[i, a+1]     = 0
    counter = 0
    amount_of_states_putting = (amount_of_states_set + state_spaces[a].shape[1])
    for i in range (amount_of_states_set, amount_of_states_putting):
        counter    += 1
        B[i, a+1]     = counter / state_spaces[a].shape[1]
    amount_of_states_set += state_spaces[a].shape[1]
    for i in range (amount_of_states_putting, amount_of_states):
        B[i, a+1]     = 1

print("\n")
print("--- DESIGN-MATRIX B ---")
print("Design-Matrix B - DATA:  ", B)
print("Design-Matrix B - SHAPE: ", B.shape)
print("-----------------------")
print("\n")

# ---------------- Minimization-Problem ----------------
b1              = (0, None)
b2              = (0.96, 1.0)
bnds            = (b1,b1,b1,b1,b1,b1,b1,b1,b1,b1,b1,b2,b2,b2,b2,b2,b2,b2,b2,b2,b2,b2,b2)

con1            = {'type':'ineq', 'fun': lambda x: x[1]}
con2            = {'type':'ineq', 'fun': lambda x: x[2]}
con3            = {'type':'ineq', 'fun': lambda x: x[3]}
con4            = {'type':'ineq', 'fun': lambda x: x[4]}
con5            = {'type':'ineq', 'fun': lambda x: x[5]}
con6            = {'type':'ineq', 'fun': lambda x: x[6]}
con7            = {'type':'ineq', 'fun': lambda x: x[7]}
con8            = {'type':'ineq', 'fun': lambda x: x[8]}
con9            = {'type':'ineq', 'fun': lambda x: x[9]}
con10           = {'type':'ineq', 'fun': lambda x: x[10]}
con11           = {'type':'ineq', 'fun': lambda x: x[11]}
constraints     = (con1,con2,con2,con3,con4,con5,con6,con7,con8,con9,con10,con11)

teta            = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
x0              = [*teta, *delta]
additional      = {'delta': delta, 'PI': pi, 'B': B, 'alpha': alpha, 'beta': beta}

def objective (x0, args):
    x1 = x0[0:11]
    x2 = x0[11:23]

    tmp_1 = args['PI'].dot(args['B'])
    tmp_2 = tmp_1.dot(x1)
    tmp_3 = args['beta'].dot(x2)
    tmp_4 = np.add(args['alpha'], tmp_3)

    tmp_5 = np.subtract(tmp_2, tmp_4)
    return np.sum(tmp_5)

res         = minimize(objective, x0, additional, method='SLSQP', bounds=bnds, constraints=con1)

teta_min    = res.x[0:11]
delta_min   = res.x[11:23]

print("\n")
print("--- MINIMIZATION PROBLEM ---")
print("TETA - Data:     " , teta_min)
print("TETA - Shape:    " , teta_min.shape)
print("DELTA - Data:    " , delta_min)
print("DELTA - Shape:   " , delta_min.shape)
print("----------------------------")
print("\n")

# ---------------- Multi-Period Physical Prob. ----------------
tmp_pd_1            = np.array(B.dot(teta_min))
tmp_pd_2            = np.diag(tmp_pd_1)
delta_diag          = np.diag(delta_min)
delta_diag_invers   = np.linalg.inv(delta_diag)

tmp_pd_3            = delta_diag_invers.dot(pi)
mp_pd               = tmp_pd_3.dot(tmp_pd_2)

# normalize P to have row sums of one
mp_pd = mp_pd / mp_pd.sum(axis=1)[:, None]

print("\n")
print("--- MULTI-PERIOD PHYSICAL PROBABILITIES ---")
print("Multi-Period physical probabilities    - Shape:", mp_pd.shape)
print("Multi-Period physical probabilities    - Data: ", mp_pd)
print("-------------------------------------------")
print("\n")

# ---------------- Phy. Exp. of one month returns ----------------
expected_return = np.zeros(12)
for a in range (0, mp_pd.shape[0]):
    for b in range(0, mp_pd.shape[1]):
        r                   = (states[0, b] / 1) - 1
        expected_return[a] += mp_pd[a, b] * r

print("\n")
print("--- Time t physical expectation of one month returns ---")
print("Expected Return    - Data:  ", expected_return)
print("Expected Return    - Shape: ", expected_return.shape)
print("--------------------------------------------------------")
print("\n")

# ---------------- Conditional expected one month excess returns ----------------
expected_excess_return  = np.zeros(12)
for a in range (0, mp_pd.shape[0]):
    expected_excess_return[a] = expected_return[a] - riskfree_rates[0, a]

print("\n")
print("--- Conditional expected one month excess returns ---")
print("Expected excess return    - Data:  ", expected_excess_return)
print("Expected excess return    - Shape: ", expected_excess_return.shape)
print("-----------------------------------------------------")
print("\n")

y = expected_excess_return
x = np.array(days_to_maturity)
plt.figure('Conditional Expected excess return')
plt.plot(x,y)
plt.xlabel('Days to maturity')
plt.ylabel('Conditional expected excess return')

# ---------------- one month conditional volatility ----------------
volatility      = np.zeros(12)
for a in range (0, expected_return.shape[0]):
    volatility[a] = np.sqrt(np.var(expected_return))

print("\n")
print("--- One month conditional volatility ---")
print("Volatility    - Data:  ", volatility)
print("Volatility    - Shape: ", volatility.shape)
print("----------------------------------------")
print("\n")

y = volatility
x = np.array(days_to_maturity)
plt.figure('Conditional volatility')
plt.plot(x,y)
plt.xlabel('Days to maturity')
plt.ylabel('Conditional volatility')
plt.show()

# ---------------- AR(1) process ----------------
Y = expected_excess_return[1:, ]

X = expected_excess_return[:-1, ]
X = sm.add_constant(X)

model       = sm.OLS(Y, X)
result      = model.fit()

# -------------------------------- RESULTS --------------------------------
# --------------------------- CORRELATION-MATRIX --------------------------
cor = np.corrcoef(expected_excess_return, volatility)
print("--- Correlation between expected Excess Return and Volatility ---")
print("Correlation    - Data:  ", cor)
print("-----------------------------------------------------------------")