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
data_qvar_all                   = pd.read_csv("data/FiglewskiStandardizationEOD_DE0009652396D1_Qmoments.csv", sep = ';')
data_rnd_all                    = pd.read_csv("data/FiglewskiStandardizationEOD_DE0009652396D1_rnd.csv",  sep = ';')
data_rf_all                     = pd.read_csv("data/riskfree_rate.csv", sep = ';')

data_qvar_all                   = data_qvar_all.loc[data_qvar_all['loctimestamp'] > '2002-01-02 00:00:00']
data_rnd_all                    = data_rnd_all.loc[data_rnd_all['loctimestamp'] > '2002-01-02 00:00:00']
data_rf_all                     = data_rf_all.loc[data_rf_all['loctimestamp'] > '2002-01-02 00:00:00']

# ---------------- Reduce Data ----------------
data_qvar_all                   = data_qvar_all.loc[data_qvar_all['daystomaturity'] <= 365]
data_rnd_all                    = data_rnd_all.loc[data_rnd_all['daystomaturity'] <= 365]
data_rf_all                     = data_rf_all.loc[data_rf_all['daystomaturity'] <= 365]

# ---------------- Process Data ----------------
data_qvar_all                   = pd.merge(data_qvar_all, data_rf_all, how='left', left_on=['loctimestamp', 'daystomaturity'], right_on=['loctimestamp', 'daystomaturity'])
data_rnd_all                    = pd.merge(data_rnd_all, data_rf_all, how='left', left_on=['loctimestamp', 'daystomaturity'], right_on=['loctimestamp', 'daystomaturity'])

data_rnd_all['moneyness']       = (data_rnd_all['implStrike'] / data_rnd_all['underlyingforwardprice']).round(decimals=3)
data_rnd_all['maturity']        = data_rnd_all['daystomaturity'] / 365
data_rnd_all['ad']              = data_rnd_all['rndStrike'] * np.exp(-data_rnd_all['riskfree'] * data_rnd_all['maturity'])

data_rnd_all['loctimestamp']    = pd.to_datetime(data_rnd_all['loctimestamp'], format = '%Y-%m-%d')
data_rf_all['loctimestamp']     = pd.to_datetime(data_rf_all['loctimestamp'], format = '%Y-%m-%d')
data_qvar_all['loctimestamp']   = pd.to_datetime(data_qvar_all['loctimestamp'], format = '%Y-%m-%d')

maturities_all                  = np.sort(np.array(data_rnd_all['maturity'].unique()))
days_to_maturity_all            = np.sort(np.array(data_rnd_all['daystomaturity'].unique()))

# ---------------- Declare Variables ----------------
teta                            = np.zeros(11)

expectations                    = pd.DataFrame(columns=['daystomaturity', 'loctimestamp', 'exp_r', 'exp_vol', 'exp_skew', 'exp_kur'])
monthly_exp_excess_return       = pd.DataFrame(columns=days_to_maturity_all)
monthly_real_excess_return      = pd.DataFrame(columns=days_to_maturity_all)

plot                            = True
cmap                            = plt.cm.get_cmap("hsv", 12)

dates                           = data_rnd_all['loctimestamp'].unique()
dates                           = dates[7:8]

#
#====================================================
#              GENERALIZED RECOVERY
#====================================================
#

# ---------------- For each date in .csv ----------------

for index, date in enumerate(dates):

    print(index, " / ", len(dates), date)

    # ---------------- Reduce Data ----------------
    data_rnd                = data_rnd_all.loc[data_rnd_all['loctimestamp'].values == date].copy()
    data_qvar               = data_qvar_all.loc[data_qvar_all['loctimestamp'].values == date].copy()
    data_rf                 = data_rf_all.loc[data_rf_all['loctimestamp'].values == date].copy()

    # ---------------- Prepare Data ----------------
    maturities              = np.sort(np.array(data_rnd['maturity'].unique()))
    days_to_maturity        = np.sort(np.array(data_rnd['daystomaturity'].unique()))

    if len(days_to_maturity_all) != len(days_to_maturity):
        print("-- continue --")
        continue

    # ---------------- State-Space ----------------
    variances               = np.array(data_qvar['Q_variance'].unique())
    variances               = np.sort(variances)
    # to determine the current level of the VIX, we use the variance of the shortest maturity
    current_VIX             = np.sqrt(variances[0] * (365/30))

    # as state we use the current return, which is 1
    lowest_State            = 1 - (2.5 * current_VIX)
    highest_State           = 1 + (4 * current_VIX)

    # ---------------- Arrow-Debreu-Prices ----------------
    data = pd.DataFrame(columns=['moneyness', days_to_maturity_all[0], days_to_maturity_all[1], days_to_maturity_all[2], days_to_maturity_all[3], days_to_maturity_all[4], days_to_maturity_all[5],])
    data.loc[:, 'moneyness'] = data_rnd.loc[(data_rnd['daystomaturity'] == days_to_maturity_all[0]) & (data_rnd['moneyness'] >= lowest_State) & (data_rnd['moneyness'] <= highest_State)]['moneyness'].values

    for day in days_to_maturity_all:
        x                   = data_rnd.loc[(data_rnd['daystomaturity'] == day) & (data_rnd['moneyness'] >= lowest_State) & (data_rnd['moneyness'] <= highest_State), ['moneyness', 'ad']]
        data.loc[:, day]    = pd.merge(data, x, on='moneyness')['ad']

    states  = np.array(data['moneyness'].unique())
    pi      = np.zeros((len(maturities), len(states)))

    data = data.fillna(0)
    for b in range(0, len(maturities)):
        pi[b] = data.loc[:, days_to_maturity_all[b]].values

    if plot:
        plt.figure('ARROW-DEBREU-PRICES')
        plt.suptitle('ARROW-DEBREU-PRICES')
        for i in range(0, pi.shape[0]):
            plt.plot(states, pi[i, :])

        plt.legend(maturities)
        plt.xlabel('States')
        plt.ylabel('Arrow-Debreu-Prices')

    # ---------------- Discount-Factor Closed-Form Recovery ----------------
    alpha                       = np.zeros(len(maturities_all))
    beta                        = np.zeros(len(maturities_all))
    delta                       = np.zeros(len(maturities_all))

    for i in range (0, len(maturities_all)):
        riskfree_rate   = data_rf[(data_rf['daystomaturity'] == days_to_maturity_all[i])]['riskfree']
        delta_zero      = 1 - riskfree_rate
        alpha[i]        = -(maturities_all[i] - 1) * (delta_zero ** maturities_all[i])
        beta[i]         = maturities_all[i] * (delta_zero ** (maturities_all[i] - 1))
        delta[i]        = alpha[i] + beta[i] * delta_zero

    # ---------------- 10 State-Spaces ----------------
    state_spaces       = []
    state_space        = np.zeros((10, 2))

    state_space[0, 0]               = 1 - 2.5 * current_VIX
    state_space[0, 1]               = 1 - 2 * current_VIX
    state_spaces.append(states[(states >= state_space[0, 0]) & (states < state_space[0, 1])])

    state_space_equal_portion_from  = 1 - 2 * current_VIX
    state_space_equal_portion_to    = 1 + 2 * current_VIX
    portion                         = (state_space_equal_portion_to - state_space_equal_portion_from) / 8

    for i in range (1, 9):
        state_space[i, 0]           = state_space[i-1, 1]
        state_space[i, 1]           = state_space[i, 0] + portion
        state_spaces.append(states[(states >= state_space[i, 0]) & (states < state_space[i, 1])])

    state_space[9, 0]               = 1 + 2 * current_VIX
    state_space[9, 1]               = 1 + 4 * current_VIX
    state_spaces.append(states[(states >= state_space[9, 0]) & (states < state_space[9, 1])])

    if state_space[8] == []:
        state_space[8] = [1.2, 1.2, 1.2, 1.2, 1.2]
    if state_space[9] == []:
        state_space[9] = [1.4,1.4,1.4,1.4,1.4]
    # ---------------- Design-Matrix B ----------------
    B                       = np.zeros((len(states), len(teta)))
    amount_of_states_set    = 0

    # Design Matrix Level
    for i in range (0, len(states)):
        B[i, 0]     = 1.00

    # Design Matrix State Space 1
    for i in range (0, len(state_spaces[0])):
        B[i, 1]     = i / len(state_spaces[0])

    amount_of_states_set   += len(state_spaces[0])
    for i in range (amount_of_states_set, len(states)):
        B[i, 1]     = 1

    # Design Matrix for State Space 2 till 10
    for a in range (1, len(state_spaces)):
        for i in range (0, amount_of_states_set):
            B[i, a+1]     = 0
        counter = 0
        amount_of_states_putting = (amount_of_states_set + len(state_spaces[a]))
        for i in range (amount_of_states_set, amount_of_states_putting):
            counter    += 1
            B[i, a+1]     = counter / len(state_spaces[a])
        amount_of_states_set += len(state_spaces[a])
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
    amount_of_bounds = (len(teta) + len(maturities_all))
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

    for i in range (0, len(maturities_all)):
        x0_seed.append(rf_seed)

    def objective (x0):
        x1 = x0[0:len(teta)]
        x2 = x0[len(teta):amount_of_bounds]

        x = ((pi.dot(B)).dot(x1)) - (alpha + beta.dot(x2))

        return - np.sum(x)

    res         = minimize(objective, x0_seed, method='SLSQP', bounds=bnds, constraints=constraints)

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
    if plot:
        plt.figure('PRICING-KERNEL')
        plt.suptitle('PRICING-KERNEL')
        plt.plot(states, (1/inv_pricing_kernel), color='r')
        for i in range(0, len(state_space)):
            plt.axvline(x=state_space[i, 0], color='b')

        plt.axvline(x=state_space[len(state_space) - 1, 1], color='b')
        plt.legend()
        plt.xlabel('States')
        plt.ylabel('Inverse PRICING-KERNEL')

    # ---------------- Physical Probabilities ----------------
    delta_diag          = np.diag(delta_min)

    #TODO - FIX
    if np.linalg.det(delta_diag) == 0:
        print("SKIP")
        continue

    delta_diag_invers   = np.linalg.inv(delta_diag)

    P_init              = delta_diag_invers.dot(pi).dot(np.diag( B.dot(teta)))

    P                   = np.zeros((len(maturities_all), len(states)))
    # normalize P to have row sums of one
    row_sums = P_init.sum(axis=1)[:, None]

    for row in range (0, len(row_sums)):
        if row_sums[row] != 0:
            P[row] = P_init[row] / row_sums[row]
        else:
            P[row] = 0

    if plot:
        plt.figure('PHYSICAL PROBABILITY DISTRIBUTION')
        plt.suptitle('PHYSICAL PROBABILITY DISTRIBUTION')
        for i in range(0, P.shape[0]):
            plt.plot(states, P[i], label='Days to maturity %s' % (days_to_maturity[i]), color=cmap(i))

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
            plt.plot(days_to_maturity[i], exp_r[i]*100, 'ro', label='Days to maturity %s' % (days_to_maturity[i]), color=cmap(i))

        plt.legend(days_to_maturity_all)
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
            plt.plot(days_to_maturity[i], vol[i], 'ro', color=cmap(i), label='Days to maturity %s' % (days_to_maturity[i]))

        plt.legend(days_to_maturity_all)
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
            plt.plot(days_to_maturity[i], skewness[i], 'ro', color=cmap(i), label='Days to maturity %s' % (days_to_maturity[i]))

        plt.legend(days_to_maturity_all)
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
            plt.plot(days_to_maturity[i], kurtosis[i], 'ro', color=cmap(i), label='Days to maturity %s' % (days_to_maturity[i]))

        plt.legend(days_to_maturity_all)
        plt.xlabel('Maturities')
        plt.ylabel('Kurtosis')

    # ---------------- Expected values ----------------
    for i in range(0, len(maturities_all)):
        expectations.loc[len(expectations)] = [days_to_maturity_all[i], date, exp_r[i], vol[i], skewness[i], kurtosis[i]]

    del data_qvar
    del data_rnd
    del data_rf

#
# =========================================================================
#                           Excess Returns
# =========================================================================
#

# ---------------- Calculate Conditional EXPECTED 30-days Excess-Returns ----------------

for day in days_to_maturity_all:
    data_rf     = data_rf_all.loc[(data_rf_all['daystomaturity'] == day)].copy()
    data        = expectations.loc[(expectations['daystomaturity'] == day), ['loctimestamp', 'exp_r']].copy()

    data        = pd.merge(data, data_rf[['loctimestamp', 'riskfree']], on='loctimestamp')
    data        = data.set_index('loctimestamp')

    monthly_exp_ex_r = data.resample('30D').mean()
    monthly_exp_ex_r = monthly_exp_ex_r.pct_change()
    monthly_exp_ex_r['returns'] = monthly_exp_ex_r['exp_r'] - (monthly_exp_ex_r['riskfree'] * (365 / 30))

    monthly_exp_excess_return[day] = monthly_exp_ex_r['returns']

    del data_rf
    del data

plt.figure('Conditional EXPECTED 30-days Excess-Returns')
plt.subplot(311).set_title("Conditional EXPECTED 30-days Excess-Returns")
plt.suptitle('Conditional EXPECTED 30-days Excess-Returns')
for day in days_to_maturity_all:
    plt.plot(monthly_exp_excess_return.index, monthly_exp_excess_return[day], label="%s"%(day))

plt.legend()
plt.xlabel('Date')
plt.ylabel('Conditional Exp Ex R in %')

print("\n")
print("---- Monthly expected excess return ----")
print(monthly_exp_excess_return.head())
print("----------------------------------------")
print("\n")

# ---------------- Calculate REALIZED 30-days Excess-Returns ----------------

for day in days_to_maturity_all:
    data_rf = data_rf_all.loc[(data_rf_all['daystomaturity'] == day)].copy()
    data    = data_qvar_all.loc[(data_qvar_all['daystomaturity'] == day), ['loctimestamp', 'underlyingforwardprice']].copy()

    data    = pd.merge(data, data_rf[['loctimestamp', 'riskfree']], on='loctimestamp')
    data    = data.set_index('loctimestamp')

    monthly_real_ex_r = data.resample('30D').mean()
    monthly_real_ex_r = monthly_real_ex_r.pct_change()
    monthly_real_ex_r['returns']  = monthly_real_ex_r['underlyingforwardprice'] - (monthly_real_ex_r['riskfree']* (365/30))

    monthly_real_excess_return[day] = monthly_real_ex_r['returns']

    del data_rf
    del data

plt.subplot(312).set_title('Realized 30-days Excess-Returns')
plt.suptitle('Realized 30-days Excess-Returns')
for day in days_to_maturity_all:
    plt.plot(monthly_exp_excess_return.index, monthly_exp_excess_return[day], label="%s"%(day))

plt.legend()
plt.xlabel('Date')
plt.ylabel('Realized Ex R in %')

print("\n")
print("---- Monthly realized excess return ----")
print(monthly_real_excess_return.head())
print("----------------------------------------")
print("\n")

# ---------------- Calculate Innovation ----------------
innovation              = pd.DataFrame(columns=[days_to_maturity_all])

for day in days_to_maturity_all:
    innovation[day] = monthly_real_excess_return[day] - monthly_exp_excess_return[day]

plt.subplot(313).set_title('Unpredictabel Innovation')
plt.suptitle('Unpredictabel Innovation')
for day in days_to_maturity_all:
    plt.plot(innovation[day], label="%s"%(day))

plt.legend()
plt.xlabel('Date')
plt.ylabel('Innovation')

print("\n")
print("--------------- Innovation -------------")
print(innovation.head())
print("----------------------------------------")
print("\n")

#
# =========================================================================
#                           Autoregression
# =========================================================================
#

# ---------------- AR(1) ----------------

monthly_real_excess_return  = monthly_real_excess_return.fillna(0)
monthly_exp_excess_return   = monthly_exp_excess_return.fillna(0)

for day in days_to_maturity_all:
    Y = np.array(monthly_real_excess_return[day])[1:len(monthly_real_excess_return)]
    X = np.array(monthly_real_excess_return[day])[0:len(monthly_real_excess_return)-1]
    X = sm.add_constant(X)

    print()
    model   = sm.OLS(Y, X)
    results = model.fit()
    print("AR(1)- Modell for maturity: ", day, ' -> a0 = ', results.params[0], " and a1 = ", results.params[0])

#
# =========================================================================
#                           Results
# =========================================================================
#

# ---------------- KPIS ----------------
#average conditional expected monthly excess return
for day in days_to_maturity_all:
    print("Annualized - Average conditional expected monthly excess return: ", monthly_exp_excess_return[day].mean()*np.sqrt(12))

#cor between recovered vol and VIX

plt.show()