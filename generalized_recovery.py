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

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')

import scipy.stats as stats
import scipy.integrate as integrate

from scipy.interpolate import interp1d
from scipy.optimize import minimize

#
#====================================================
#                       DATA
#====================================================
#

# ---------------- variable declarations ----------------
MATURITY_365                = 365
TETAS_AMOUNT                = 11
STATE_SPACES_AMOUNT         = 10

teta                        = np.zeros(TETAS_AMOUNT)
expectations                = pd.DataFrame(columns=['daystomaturity', 'loctimestamp', 'exp_r', 'exp_vol', 'exp_skew', 'exp_kurt'])

# ---------------- Load Data ----------------
data_qvar_rf                = pd.read_csv("data/eurostoxx50_qvar_riskfree_all.csv", sep = ';')
data_rnd                    = pd.read_csv("data/eurostoxx50_rnd.csv",  sep = ';')

# ---------------- Reduce Data ----------------
# only use options with maturity less than 365 days for stability
data_rnd                    = data_rnd.loc[data_rnd['daystomaturity'] <= MATURITY_365]

# ---------------- Prepare Data ----------------
data_rnd['loctimestamp']    = pd.to_datetime(data_rnd['loctimestamp'], format = '%Y-%m-%d')
data_rnd                    = pd.merge(data_rnd, data_qvar_rf[['daystomaturity', 'riskfree', 'bakshiVariance']], on = 'daystomaturity')

# ---------------- Calculate Data ----------------
data_rnd['ad']              = data_rnd['rndStrike'] * np.exp(-data_rnd['riskfree'] * data_rnd['daystomaturity'] / MATURITY_365)
data_rnd['moneyness']       = (data_rnd['implStrike'] / data_rnd['underlyingforwardprice']).round(decimals = 3)
data_rnd['maturity']        = data_rnd['daystomaturity'] / MATURITY_365

# ------------------- Get Data ----------------
# use days_to_maturity for visualization
days_to_maturity            = np.sort(np.array(data_rnd['daystomaturity'].unique()))
# use annualized maturities for calculus
maturities                  = np.sort(np.array(data_rnd['maturity'].unique()))

# ------------------- TXT-Format -------------------
txt_days_to_maturity        = "Days to maturity"
SEPERATOR                   = "--------------------------------------------------------"
cmap                        = plt.cm.get_cmap("hsv", 12)

#
#====================================================
#              GENERALIZED RECOVERY
#====================================================
#

# ---------------- For each given day ----------------

for date in data_rnd['loctimestamp'].unique():

    # ---------------- State-Spaces ----------------
    variances            = np.array(data_rnd['bakshiVariance'].unique())
    variances            = np.sort(variances)
    # get highest variance, for biggest state space
    current_VIX          = np.sqrt(variances[-1] * (MATURITY_365/30))

    # as state we use the current return, which is 1
    lowest_State         = 1 - 0.25 * current_VIX
    highest_State        = 1 + 0.4 * current_VIX

    print("\n")
    print("------------ DATA SETUP ------------")
    print("Current VIX:     ", current_VIX)
    print("Lowest State:    ", lowest_State)
    print("Highest State:   ", highest_State)
    print(SEPERATOR)

    # ---------------- Arrow-Debreu-Prices ----------------
    # collect all states
    data = pd.DataFrame()
    for maturity in maturities:
        data = pd.concat([data, data_rnd.loc[(data_rnd['maturity'] == maturity) & (data_rnd['moneyness'] >= lowest_State) & (data_rnd['moneyness'] <= highest_State)]['moneyness']])

    states = np.sort(np.array(data[0].unique()))

    # get Arrow-Debreu prices of all to define state price matrix
    pi = np.zeros((len(maturities), len(states)))
    for index in range(0, len(maturities)):
        y       = data_rnd.loc[(data_rnd['maturity'] == maturities[index])]['ad'].values
        x       = data_rnd.loc[(data_rnd['maturity'] == maturities[index])]['moneyness'].values
        values  = data_rnd.loc[(data_rnd['maturity'] == maturities[index])]

        for state in range(0, len(states)):
            ad = values[data_rnd['moneyness'] == states[state]]['ad'].values
            if ad:
                pi[index, state] = ad[0]
            elif index > 0 and state + 1 < len(states):
                # interpolate results of all non defined states
                f = interp1d(x, y)
                pi[index, state] = f(states[state])
                print("Interpolate for ", pi[index, state])

    plt.figure('RISK NEUTRAL DENSITY')
    plt.suptitle('RISK NEUTRAL DENSITY')
    for i in range(0, pi.shape[0]):
        plt.plot(states, pi[i, :], label='Days to maturity %s'%(days_to_maturity[i]), color=cmap(i))

    plt.legend(days_to_maturity)
    plt.xlabel('States')
    plt.ylabel('Arrow-Debreu-Prices')

    print("------------- ARROW-DEBREU-PRICE MATRIX -------------")
    print("Arrow-Debreu-Price Matrix")
    print("Amount of States:            ", len(states))
    print("Amount of Maturities:        ", len(maturities))
    print(SEPERATOR)

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

    print("\n")
    print("------------- DISCOUNT-FACTOR CLOSED-FORM-RECOVERY -------------")
    print("Discount-factor approximation:")
    print("Alpha:   ")
    print(alpha)
    print(alpha.shape)
    print("Beta:    ")
    print(beta)
    print(beta.shape)
    print("Delta:   ")
    print(delta)
    print(delta.shape)
    print(SEPERATOR)

    # ---------------- 10 State-Spaces ----------------
    state_spaces       = []
    state_space        = np.zeros((STATE_SPACES_AMOUNT, 2))

    state_space[0, 0]               = 1 - 0.25 * current_VIX
    state_space[0, 1]               = 1 - 0.2 * current_VIX
    state_spaces.append(np.matrix(states[(states >= state_space[0, 0]) & (states < state_space[0, 1])]))

    state_space_equal_portion_from  = 1 - 0.2 * current_VIX
    state_space_equal_portion_to    = 1 + 0.2 * current_VIX
    portion                         = (state_space_equal_portion_to - state_space_equal_portion_from) / 8

    print("\n")
    print("--------------- STATE SPACE --------------- ")
    print("0  FROM ", state_space[0, 0], " TO ", state_space[0, 1])

    for i in range (1, 9):
        state_space[i, 0]           = state_space[i-1, 1]
        state_space[i, 1]           = state_space[i, 0] + portion
        state_spaces.append(np.matrix(states[(states >= state_space[i, 0]) & (states < state_space[i, 1])]))
        print(i, " FROM ", state_space[i, 0], " TO ", state_space[i, 1])

    state_space[9, 0]               = 1 + 0.2 * current_VIX
    state_space[9, 1]               = 1 + 0.4 * current_VIX
    state_spaces.append(np.matrix(states[(states >= state_space[9, 0]) & (states < state_space[9, 1])]))

    print("9  FROM ", state_space[9, 0], " TO ", state_space[9, 1])
    print(SEPERATOR)

    # ---------------- Design-Matrix B ----------------
    B                       = np.zeros((len(states), len(teta)))
    # counting the rank of the matrix
    amount_of_states_set    = 0

    # Design Matrix Level
    for i in range (0, len(states)):
        B[i, 0]     = 1.00

    # Design Matrix State Space 1
    for i in range (0, state_spaces[0].shape[1]):
        B[i, 1]     = (i) / state_spaces[0].shape[1]

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
            counter      += 1
            B[i, a+1]     = counter / state_spaces[a].shape[1]
        amount_of_states_set += state_spaces[a].shape[1]
        for i in range (amount_of_states_putting, len(states)):
            B[i, a+1]     = 1

    plt.figure('DESIGN-MATRIX')
    plt.suptitle('DESIGN-MATRIX')
    for i in range(0, B.shape[1]):
        plt.plot(states, B[:,i], label='Column %s'% (i+1), color=cmap(i))

    plt.legend()
    plt.xlabel('States')
    plt.ylabel('Column')

    print("\n")
    print("--------------- DESIGN-MATRIX B ---------------")
    print("Design-Matrix B - DATA:  ")
    print(B)
    print("Design-Matrix B - SHAPE: ")
    print(B.shape)
    print(SEPERATOR)

    # ---------------- Minimization-Problem ----------------
    # boundaries
    b1              = (0, None)
    b2              = (0, 1.0)
    bnds            = [b1,b1,b1,b1,b1,b1,b1,b1,b1,b1,b1,b2]

    teta_seed       = 1
    rf_seed         = np.sum(delta) / len(delta)

    x0_seed = [teta_seed, teta_seed, teta_seed, teta_seed, teta_seed, teta_seed, teta_seed, teta_seed, teta_seed, teta_seed, teta_seed, rf_seed]

    def objective (x0):
        x1 = x0[0:len(teta)]
        x2 = x0[len(teta):len(teta)+1]

        x = pi.dot(B).dot(x1) - (alpha + beta*x2)

        return np.linalg.norm(x)

    res         = minimize(objective, x0_seed, method='L-BFGS-B', bounds=bnds)

    teta        = res.x[0:len(teta)]
    delta_min   = res.x[len(teta):len(teta)+1]

    print("\n")
    print("--- MINIMIZATION PROBLEM ---")
    print("TETA - Data:     ")
    print(teta)
    print(teta.shape)
    print("DELTA - Data:    ")
    print(delta_min)
    print(delta_min.shape)
    print(SEPERATOR)

    # ---------------- Pricing-Kernel ----------------
    inv_pricing_kernel = B.dot(teta)

    plt.figure('INVERSE PRICING-KERNEL')
    plt.suptitle('INVERSE PRICING-KERNEL')
    plt.plot(states, inv_pricing_kernel, color='r')
    for i in range (0, len(state_space)):
        plt.axvline(x=state_space[i,1], color='b')

    plt.axvline(x=state_space[len(state_space)-1, 1], color='b')
    plt.legend()
    plt.xlabel('States')
    plt.ylabel('Inverse PRICING-KERNEL')

    plt.figure('PRICING-KERNEL')
    plt.suptitle('PRICING-KERNEL')
    plt.plot(states, (1/inv_pricing_kernel), color='r')
    for i in range (0, len(state_space)):
        plt.axvline(x=state_space[i,1], color='b')

    plt.axvline(x=state_space[len(state_space)-1, 1], color='b')
    plt.legend()
    plt.xlabel('States')
    plt.ylabel('PRICING-KERNEL')

    # ---------------- Physical Probabilities ----------------
    deltas = np.matrix([delta_min ** 1, delta_min ** 2, delta_min ** 3, delta_min ** 4, delta_min ** 5,
                                        delta_min ** 6, delta_min ** 7, delta_min ** 8, delta_min ** 9, delta_min ** 10,
                                        delta_min ** 11, delta_min ** 12])

    delta_diag          = np.diagflat(deltas)
    delta_diag_invers   = np.linalg.inv(delta_diag)

    P_init              = delta_diag_invers.dot(pi).dot(np.diag( B.dot(teta)))

    # normalize P to have row sums of one
    P = np.asarray(P_init / P_init.sum(axis=1))

    plt.figure('PHYSICAL PROBABILITY DISTRIBUTION')
    plt.subplot(211).set_title('PHYSICAL PROBABILITY DISTRIBUTION')
    for i in range(0, P.shape[0]):
        plt.plot(states, P[i], label='Days to maturity %s'%(days_to_maturity[i]), color=cmap(i))

    plt.legend()
    plt.xlabel('States')
    plt.ylabel('Physical Probability')

    print("\n")
    print("--- MULTI-PERIOD PHYSICAL PROBABILITIES ---")
    print("Multi-Period physical probabilities    - Data: ")
    print(P)
    print("Multi-Period physical probabilities    - Shape:", P.shape)
    print(SEPERATOR)

    #
    # =========================================================================
    #     Computing Statistics under the Physical Probability Distribution
    # =========================================================================
    #

    # ---------------- calculating moments under P-density ----------------
    exp_r   = np.zeros(len(P))
    exp_r_2 = np.zeros(len(P))
    exp_r_3 = np.zeros(len(P))
    exp_r_4 = np.zeros(len(P))

    for i in range (0, len(P)):
        p_list      = np.array(P[i, :]).tolist()
        exp_r[i]    = integrate.trapz(p_list, states, 0.01)
        exp_r_2[i]  = integrate.trapz(np.power(p_list, 2), states, 0.01)
        exp_r_3[i]  = integrate.trapz(np.power(p_list, 3), states, 0.01)
        exp_r_4[i]  = integrate.trapz(np.power(p_list, 4), states, 0.01)

    print("\n")
    print("--- Conditional expected moments under P-DENSITY ---")

    # ---------------- conditional Phy. Exp. for next day ----------------
    plt.subplot(212)
    for i in range(0, len(exp_r)):
        plt.plot(days_to_maturity[i], exp_r[i]*100, 'ro', color=cmap(i), label="Days to Maturity %s"%(days_to_maturity[i]))

    plt.legend()
    plt.xlabel(txt_days_to_maturity)
    plt.ylabel('Expected Return in %')

    print("Expected Return    - Data:  \n", exp_r)
    print(SEPERATOR)

    # ---------------- conditional Volatility ----------------
    var = exp_r_2 - np.power(exp_r, 2)
    vol = np.sqrt(var)

    plt.figure('CONDITIONAL EXPECTATIONS')
    plt.subplot(311).set_title('EXPECTED VOLATILITY')
    for i in range(0, len(vol)):
        plt.plot(days_to_maturity[i], vol[i], 'ro', color=cmap(i), label='Days to maturity %s'%(days_to_maturity[i]))

    plt.legend(days_to_maturity)
    plt.xlabel(txt_days_to_maturity)
    plt.ylabel('Volatility')

    print("Expected Volatility    - Data:  \n", vol)
    print(SEPERATOR)

    # ---------------- conditional Skewness ----------------
    skewness = np.zeros(len(P))
    for i in range(0, len(P)):
        skewness[i] = stats.skew(P[i])

    plt.subplot(312).set_title('EXPECTED SKEWNESS')
    for i in range(0, len(skewness)):
        plt.plot(days_to_maturity[i], skewness[i], 'ro', color=cmap(i), label='Days to maturity %s'%(days_to_maturity[i]))

    plt.legend(days_to_maturity)
    plt.xlabel(txt_days_to_maturity)
    plt.ylabel('Skewness')

    print("Expected Skewness    - Data:  \n", skewness)
    print(SEPERATOR)

    # ---------------- conditional Kurtosis ----------------
    kurtosis = np.zeros(len(P))
    for i in range(0, len(P)):
        kurtosis[i] = stats.kurtosis(P[i])

    plt.subplot(313).set_title('EXPECTED KURTOSIS')
    for i in range(0, len(kurtosis)):
        plt.plot(days_to_maturity[i], kurtosis[i], 'ro', color=cmap(i), label='Days to maturity %s'%(days_to_maturity[i]))

    plt.legend(days_to_maturity)
    plt.xlabel(txt_days_to_maturity)
    plt.ylabel('Kurtosis')

    print("Expected Kurtosis    - Data:  \n", kurtosis)
    print(SEPERATOR)

    # ---------------- Expected values ----------------
    for i in range(0, len(maturities)):
        expectations.loc[len(expectations.index)] = [days_to_maturity[i], date, exp_r[i], vol[i], skewness[i], kurtosis[i]]

plt.show()

print("\n")
print("--- One month conditional Expectations ---")
print("Expectations    - Data:  ", expectations)
print(SEPERATOR)
print("\n")