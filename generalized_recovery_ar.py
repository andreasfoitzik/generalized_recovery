#
#========================================================================================
#   Applied Risk and Asset Management
#   Generalized Recovery Model
#   based on Christian Skov Jensen, David Lando, and Lasse Heje Pedersen; version of December 21, 2016
#
#   done by Andreas Foitzik
#========================================================================================
#
#
#

import numpy as np
import pandas as pd
import statsmodels.api as sm

from scipy.optimize import minimize

import scipy.integrate as integrate
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')

#
#====================================================
#                       DATA
#====================================================
#

# ---------------- Load Data ----------------
data_qvar_all               = pd.read_csv("data/FiglewskiStandardizationEOD_DE0009652396D1_Qmoments.csv", sep = ';')
data_rnd_all                = pd.read_csv("data/FiglewskiStandardizationEOD_DE0009652396D1_rnd.csv",  sep = ';')
data_rf_all                 = pd.read_csv("data/riskfree_rate.csv", sep = ';')

# ---------------- Reduce Data ----------------
data_qvar_all               = data_qvar_all.loc[data_qvar_all['daystomaturity'] <= 365]
data_rnd_all                = data_rnd_all.loc[data_rnd_all['daystomaturity'] <= 365]
data_rf_all                 = data_rf_all.loc[data_rf_all['daystomaturity'] <= 365]

# ---------------- Process Data ----------------
data_rnd_all['moneyness']   = (data_rnd_all['implStrike'] / data_rnd_all['underlyingforwardprice']).round(decimals=3)
data_rnd_all['maturity']    = data_rnd_all['daystomaturity'] / 365

# ---------------- Declare Variables ----------------
teta                        = np.zeros(11)
expectations                = pd.DataFrame(columns=['daystomaturity', 'loctimestamp', 'exp_r', 'exp_vol', 'exp_skew'])
plot                        = False
cmap = plt.cm.get_cmap("hsv", 12)

#
#====================================================
#              GENERALIZED RECOVERY
#====================================================
#

# ---------------- For each date in .csv ----------------
dates                       = data_rnd_all['loctimestamp'].unique()
dates                       = dates[0:30]

for date in dates:

    print("Date: ", date)

    # ---------------- Reduce Data ----------------
    data_rnd                = data_rnd_all.loc[data_rnd_all['loctimestamp'] == date].copy()
    data_qvar               = data_qvar_all.loc[data_qvar_all['loctimestamp'] == date].copy()
    data_rf                 = data_rf_all.loc[data_rf_all['loctimestamp'] == date].copy()

    # ---------------- Prepare Data ----------------
    data_qvar               = pd.merge(data_qvar, data_rf[['daystomaturity', 'riskfree']], on='daystomaturity')
    data_rnd                = pd.merge(data_rnd, data_qvar[['daystomaturity', 'riskfree']], on='daystomaturity')
    data_rnd['ad']          = data_rnd['rndStrike'] * np.exp( -data_rnd['riskfree'] * data_rnd['maturity'])

    maturities              = np.array(data_rnd['maturity'].unique())
    days_to_maturity        = np.array(data_rnd['daystomaturity'].unique())

    # ---------------- State-Space ----------------
    variances            = np.array(data_qvar['Q_variance'].unique())
    variances            = np.sort(variances)
    current_VIX          = np.sqrt(variances[0] * (365/30))

    # as state we use the current return, which is 1
    lowest_State            = 1 - (2.5 * current_VIX)
    highest_State           = 1 + (4 * current_VIX)

    # ---------------- Arrow-Debreu-Prices ----------------
    data = data_rnd.loc[(data_rnd['maturity'] == maturities[0]) & (data_rnd['moneyness'] >= lowest_State) & (data_rnd['moneyness'] <= highest_State), ['moneyness', 'ad']]
    data.columns.values[1] = days_to_maturity[0]

    for a in range(1, len(maturities)):
        x       = data_rnd.loc[(data_rnd['maturity'] == maturities[a]) & (data_rnd['moneyness'] >= lowest_State) & (data_rnd['moneyness'] <= highest_State),['moneyness', 'ad']]
        data    = pd.merge(data, x, on='moneyness')
        data.columns.values[a + 1] = days_to_maturity[a]

    states                  = np.array(data['moneyness'].unique())
    pi                      = np.zeros((len(maturities), len(states)))

    for b in range(0, len(maturities)):
        pi[b] = data.iloc[:, (b+1)].values

    if plot:
        plt.figure('ARROW-DEBREU-PRICES')
        plt.suptitle('ARROW-DEBREU-PRICES')
        for i in range(0, pi.shape[0]):
            plt.plot(states, pi[i, :])

        plt.legend(maturities)
        plt.xlabel('States')
        plt.ylabel('Arrow-Debreu-Prices')

    # ---------------- Discount-Factor Closed-Form Recovery ----------------
    alpha                       = np.zeros(len(maturities))
    beta                        = np.zeros(len(maturities))
    delta                       = np.zeros(len(maturities))

    for i in range (0, len(maturities)):
        riskfree_rate   = data_rnd[(data_rnd['maturity'] == maturities[i])]['riskfree'].unique()[0]
        delta_zero      = 1 - riskfree_rate
        alpha[i]        = -(maturities[i] - 1) * (delta_zero ** maturities[i])
        beta[i]         = maturities[i] * (delta_zero ** (maturities[i] - 1))
        delta[i]        = alpha[i] + beta[i] * delta_zero

    # ---------------- 10 State-Spaces ----------------
    state_spaces       = []
    state_space        = np.zeros((10, 2))

    state_space[0, 0]               = 1 - 2.5 * current_VIX
    state_space[0, 1]               = 1 - 2 * current_VIX
    state_spaces.append(np.matrix(states[(states >= state_space[0, 0]) & (states < state_space[0, 1])]))

    state_space_equal_portion_from  = 1 - 2 * current_VIX
    state_space_equal_portion_to    = 1 + 2 * current_VIX
    portion                         = (state_space_equal_portion_to - state_space_equal_portion_from) / 8

    for i in range (1, 9):
        state_space[i, 0]           = state_space[i-1, 1]
        state_space[i, 1]           = state_space[i, 0] + portion
        state_spaces.append(np.matrix(states[(states >= state_space[i, 0]) & (states < state_space[i, 1])]))

    state_space[9, 0]               = 1 + 2 * current_VIX
    state_space[9, 1]               = 1 + 4 * current_VIX
    state_spaces.append(np.matrix(states[(states >= state_space[9, 0]) & (states < state_space[9, 1])]))

    # ---------------- Design-Matrix B ----------------
    B                       = np.zeros((len(states), len(teta)))
    amount_of_states_set    = 0

    # Design Matrix Level
    for i in range (0, len(states)):
        B[i, 0]     = 1.00

    # Design Matrix State Space 1
    for i in range (0, state_spaces[0].shape[1]):
        B[i, 1]     = (i+1) / state_spaces[0].shape[1]

    amount_of_states_set   += state_spaces[0].shape[1]
    for i in range (amount_of_states_set, len(states)):
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
        for i in range (amount_of_states_putting, len(states)):
            B[i, a+1]     = 1

    if plot:
        plt.figure('DESIGN-MATRIX')
        plt.suptitle('Design-Matrix')
        for i in range(0, B.shape[1]):
            plt.plot(states, B[:,i], label='Column %s'% (i+1))

        plt.legend()
        plt.xlabel('States')
        plt.ylabel('Column')

    # ---------------- Minimization-Problem ----------------
    amount_of_bounds = (len(teta)+len(maturities))
    b1              = (0, None)
    b2              = (0, 1.0)
    bnds            = [b1,b1,b1,b1,b1,b1,b1,b1,b1,b1,b1]

    for i in range (len(teta), amount_of_bounds):
        bnds.append(b2)

    con             = {'type':'ineq', 'fun': lambda x: x > 0}
    constraints     = [con]

    teta_seed       = 0.1
    rf_seed         = np.sum(delta) / len(delta)
    x0_seed         = [teta_seed, teta_seed, teta_seed, teta_seed, teta_seed, teta_seed, teta_seed, teta_seed, teta_seed, teta_seed, teta_seed]

    for i in range (0, len(maturities)):
        x0_seed.append(rf_seed)

    additional      = {'PI': pi, 'B': B, 'alpha': alpha, 'beta': beta}

    def objective (x0, args):
        x1 = x0[0:len(teta)]
        x2 = x0[len(teta):amount_of_bounds]

        x = -1 *((args['PI'].dot(args['B'])).dot(x1)) - (args['alpha'] + args['beta'].dot(x2))

        return np.sum(x)

    res         = minimize(objective, x0_seed, additional, method='SLSQP', bounds=bnds, constraints=constraints)

    teta        = res.x[0:len(teta)]
    delta_min   = res.x[len(teta):amount_of_bounds]

    inv_pricing_kernel = B.dot(teta)

    # TODO - include initial level of the inverse pricing kernel
    if plot:
        plt.figure('Inverse PRICING-KERNEL')
        plt.suptitle('Inverse PRICING-KERNEL')
        plt.plot(states, inv_pricing_kernel, color='r')
        for i in range (0, len(state_space)):
            plt.axvline(x=state_space[i,0], color='b')

        plt.axvline(x=state_space[len(state_space)-1, 1], color='b')
        plt.legend()
        plt.xlabel('States')
        plt.ylabel('Inverse PRICING-KERNEL')

    #TODO - include pricing kernel

    # ---------------- Physical Probabilities ----------------
    delta_diag          = np.diag(delta_min)

    #TODO - check whether this the right approach
    if np.linalg.det(delta_diag) == 0:
        continue

    delta_diag_invers   = np.linalg.inv(delta_diag)

    P_init              = delta_diag_invers.dot(pi).dot(np.diag( B.dot(teta)))

    # normalize P to have row sums of one
    P = P_init / P_init.sum(axis=1)[:, None]

    if plot:
        plt.figure('PHYSICAL PROBABILITY DISTRIBUTION')
        plt.suptitle('PHYSICAL PROBABILITY DISTRIBUTION')
        for i in range(0, P.shape[0]):
            plt.plot(states, P[i])

        plt.legend(days_to_maturity)
        plt.xlabel('States')
        plt.ylabel('Physical Probability')

    #
    # =========================================================================
    #     Computing Statistics under the Physical Probability Distribution
    # =========================================================================
    #

    # ---------------- conditional Phy. Exp. for next day ----------------
    exp_r = np.empty(P.shape[0])
    for i in range (0, P.shape[0]):
        exp_r[i] = integrate.trapz(np.array(P[i, :]).tolist(), states, 0.01)

    if plot:
        plt.figure('Conditional Phy. Exp. for the next day')
        plt.suptitle('Conditional Phy. Exp. of one month returns')
        for i in range(0, len(exp_r)):
            plt.plot(days_to_maturity[i], exp_r[i]*100)

        plt.legend(days_to_maturity)
        plt.xlabel('Maturities')
        plt.ylabel('Expected Return in %')

    # ---------------- conditional volatility ----------------
    exp_r_2 = np.empty(P.shape[0])
    for i in range (0, P.shape[0]):
        exp_r_2[i] = integrate.trapz(np.power(np.array(P[i, :]).tolist(), 2), states, 0.01)

    var = exp_r_2 - np.power(exp_r, 2)
    vol = np.sqrt(var)

    if plot:
        plt.figure('Conditional Volatility')
        plt.suptitle('Conditional Volatility')
        for i in range(0, len(vol)):
            plt.plot(maturities, vol)

        plt.legend(days_to_maturity)
        plt.xlabel('Maturities')
        plt.ylabel('Conditional Volatility')

    # ---------------- conditional Skewness ----------------
    exp_r_3 = np.empty(P.shape[0])
    for i in range (0, P.shape[0]):
        exp_r_3[i] = integrate.trapz(np.power(np.array(P[i, :]).tolist(), 3), states, 0.01)

    skewness = (exp_r_3 - (3 * exp_r.dot(exp_r_2)) + 2 * np.power(exp_r, 3)) / np.power((exp_r_2 - np.power(exp_r, 2)), 3/2)

    if plot:
        plt.figure('Conditional Skewness')
        plt.suptitle('Conditional Skewness')
        for i in range(0, len(skewness)):
            plt.plot(maturities, skewness)

        plt.legend(days_to_maturity)
        plt.xlabel('Maturities')
        plt.ylabel('Skewness')

    # ---------------- conditional Kurtosis ----------------
    kurtosis = np.zeros(P.shape[0])
    for i in range (0, P.shape[0]):
        x = 0
        for a in range(0, P.shape[1]):
            x += (P[i,a] - exp_r[i]) ** 4
        kurtosis[i] = x / P.shape[1]

    if plot:
        plt.figure('Conditional Kurtosis')
        plt.suptitle('Conditional Kurtosis')
        for i in range(0, len(kurtosis)):
            plt.plot(maturities, kurtosis)
        plt.legend(days_to_maturity)
        plt.xlabel('Maturities')
        plt.ylabel('Kurtosis')

    # ---------------- Expected values ----------------
    for i in range(0, len(maturities)):
        expectations.loc[len(expectations.index)] = [days_to_maturity[i], date, exp_r[i], vol[i], skewness[i]]

#
# =========================================================================
#                           Autoregression
# =========================================================================
#

# ---------------- Expectations ----------------
print("\n")
print("--- Expectations ---")
for i in range (0, len(maturities)):
    print("Expected values:\n", expectations.loc[expectations['daystomaturity'] == days_to_maturity[i]])

print("----------------------------------------")
print("\n")

plt.figure('Expected Return')
plt.suptitle('Expected Return')
for i in range(0, len(maturities)):
    plt.plot(expectations.loc[expectations['daystomaturity'] == days_to_maturity[i]]['exp_r'])

plt.legend(days_to_maturity)
plt.xlabel('Date')
plt.ylabel('Expected Return in %')

# ---------------- Calculate Returns ----------------
plt.figure('Return')
plt.suptitle('Return')
expected_returns            = expectations.loc[expectations['daystomaturity'] == days_to_maturity[0]]['exp_r']
expected_returns_shifted    = expected_returns.shift(1)
returns                     = (expected_returns /  expected_returns_shifted)
plt.plot(returns)

plt.legend(days_to_maturity)
plt.xlabel('Date')
plt.ylabel('Expected Return')

# ---------------- AR(1) ----------------
returns = returns.dropna(axis=0)
returns = returns.values

Y = returns[1:len(returns)]
X = returns[0:len(returns)-1]
X = sm.add_constant(X)

model   = sm.OLS(Y, X)
results = model.fit()    # Fit the model
print(results.summary())

plt.show()