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
from scipy.interpolate import interp1d
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

format = '%Y-%m-%d'
data_rnd_all['loctimestamp']    = pd.to_datetime(data_rnd_all['loctimestamp'], format=format)
data_rf_all['loctimestamp']     = pd.to_datetime(data_rf_all['loctimestamp'], format=format)
data_qvar_all['loctimestamp']   = pd.to_datetime(data_qvar_all['loctimestamp'], format=format)

maturities_all                  = np.sort(np.array(data_rnd_all['maturity'].unique()))
days_to_maturity_all            = np.sort(np.array(data_rnd_all['daystomaturity'].unique()))

# ---------------- Declare Variables ----------------
teta                            = np.zeros(11)

expectations                    = pd.DataFrame(columns=['loctimestamp', 'daystomaturity', 'exp_r', 'exp_vol', 'exp_skew', 'exp_kur'])
monthly_exp_excess_return       = pd.DataFrame(columns=days_to_maturity_all)

plot                            = False
plot_final                      = True
cmap                            = plt.cm.get_cmap("hsv", 12)

dates                           = data_rnd_all['loctimestamp'].unique()
#dates                           = dates[0:60]

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

    # TODO: Create it more flexible, to pass various maturities through it
    if len(days_to_maturity_all) != len(days_to_maturity):
        print("-- continue --")
        continue

    # ---------------- State-Space ----------------
    variances               = np.array(data_qvar['Q_variance'].unique())
    variances               = np.sort(variances)
    current_VIX             = np.sqrt(variances[-1] * (365/30))

    # as state we use the current return, which is 1
    lowest_State            = 1 - (2.5 * current_VIX)
    highest_State           = 1 + (4 * current_VIX)

    # ---------------- Arrow-Debreu-Prices ----------------
    # get all states as moneyness
    data = pd.DataFrame()
    for day in days_to_maturity:
        data = pd.concat([data, data_rnd.loc[(data_rnd['daystomaturity'] == day) & (data_rnd['moneyness'] >= lowest_State) & (data_rnd['moneyness'] <= highest_State)]['moneyness']])

    states = np.sort(np.array(data[0].unique()))

    # get arrow-debreu prices for all states
    # interpolate the results for all not defined states
    pi = np.zeros((len(maturities), len(states)))
    for day in range(0, len(days_to_maturity)):
        values  = data_rnd.loc[(data_rnd['daystomaturity'] == days_to_maturity[day])]

        for state in range(0, len(states)):
            ad = values[data_rnd['moneyness'] == states[state]]['ad'].values
            if ad:
                pi[day, state] = ad[0]
            elif day > 0 and state + 1 < len(states) and values['moneyness'].iloc[0] < states[state] and values['moneyness'].iloc[-1] > states[state]:
                f = interp1d(values['moneyness'].values, values['ad'].values)
                pi[day, state] = f(states[state])
                print("Interpolate for ", pi[day, state])

    if plot:
        plt.figure('ARROW-DEBREU-PRICES')
        plt.suptitle('ARROW-DEBREU-PRICES')
        for i in range(0, pi.shape[0]):
            plt.plot(states, pi[i, :], label='Days to maturity %s' % (days_to_maturity[i]), color=cmap(i))

        plt.legend()
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
    b1              = (0, None)
    b2              = (0, 1.0)
    bnds            = [b1,b1,b1,b1,b1,b1,b1,b1,b1,b1,b1,b2]

    teta_seed       = 0.1
    rf_seed         = np.sum(delta) / len(delta)

    x0_seed         = [0.5, teta_seed, teta_seed, teta_seed, teta_seed, teta_seed, teta_seed, teta_seed, teta_seed, teta_seed, teta_seed, rf_seed]

    def objective(x0):
        x1 = x0[0:len(teta)]
        x2 = x0[len(teta):len(teta)+1]

        x = (pi.dot(B)).dot(x1) - (alpha + beta*x2)

        return np.linalg.norm(x)

    res         = minimize(objective, x0_seed, method='L-BFGS-B', bounds=bnds)

    teta        = res.x[0:len(teta)]
    delta_min   = res.x[len(teta):len(teta)+1]

    inv_pricing_kernel = B.dot(teta)

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
    delta_diag          = np.diagflat(np.matrix([delta_min**1,delta_min**2,delta_min**3,delta_min**4,delta_min**5,delta_min**6]))

    if np.linalg.det(delta_diag) == 0:
        print("-- SKIP --")
        continue

    delta_diag_invers   = np.linalg.inv(delta_diag)

    P_init              = delta_diag_invers.dot(pi).dot(np.diag( B.dot(teta)))

    # normalize P to have row sums of one
    P = np.asarray(P_init / P_init.sum(axis=1))

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
    exp_r = np.zeros(P.shape[0])
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
    exp_r_2 = np.zeros(P.shape[0])
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
    exp_r_3 = np.zeros(P.shape[0])
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
    for i in range(0, P.shape[0]):
        num = 1 / P.shape[1] * sum((P[i] - np.mean(P[i])) ** 4)
        den = (1 / P.shape[1] * sum((P[i] - np.mean(P[i])) ** 2)) ** 2
        kurtosis[i] = num / den

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
        expectations.loc[len(expectations)] = [date, days_to_maturity_all[i], exp_r[i], vol[i], skewness[i], kurtosis[i]]

    del data_qvar
    del data_rnd
    del data_rf

#
# =========================================================================
#                           Calculate Results
# =========================================================================
#

# ---------------- Calculate Conditional EXPECTED 30-days Excess-Returns ----------------

for day in days_to_maturity_all:
    data_rf     = data_rf_all.loc[(data_rf_all['daystomaturity'] == day)].copy()
    data        = expectations.loc[(expectations['daystomaturity'] == day), ['loctimestamp', 'exp_r']].copy()

    data        = pd.merge(data, data_rf[['loctimestamp', 'riskfree']], on='loctimestamp')
    data        = data.set_index('loctimestamp')

    monthly_exp_ex_r = data.resample('30D').last()
    monthly_exp_ex_r = monthly_exp_ex_r.pct_change()
    monthly_exp_ex_r['returns'] = monthly_exp_ex_r['exp_r'] - (monthly_exp_ex_r['riskfree'] * (365 / 30))

    monthly_exp_excess_return[day] = monthly_exp_ex_r['returns']

    del data_rf
    del data

if plot_final:
    plt.figure('Conditional EXPECTED 30-days Excess-Returns')
    plt.suptitle('Conditional EXPECTED 30-days Excess-Returns')
    for day in days_to_maturity_all:
        plt.plot(monthly_exp_excess_return.index.to_pydatetime(), monthly_exp_excess_return[day], label="%s"%(day))

    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('Conditional Exp Ex R in %')

print("\n")
print("---- Monthly expected excess return ----")
print(monthly_exp_excess_return.head())
print("----------------------------------------")
print("\n")

#
# =========================================================================
#                           Innovation
# =========================================================================
#
#

es50_P_values_daily           = pd.read_csv("data/es50_P_values_daily.csv", sep = ';')
es50_P_values_daily['Date']   = pd.to_datetime(es50_P_values_daily['Date'])
es50_P_values_daily           = es50_P_values_daily.set_index('Date')

es50_P_values_30_days         = pd.read_csv("data/es50_P_values_30_days.csv", sep = ';')
es50_P_values_30_days['Date'] = pd.to_datetime(es50_P_values_30_days['Date'])
es50_P_values_30_days         = es50_P_values_30_days.set_index('Date')

# ---------------- Innovation Daily ----------------
# daily innovation for: excess return, volatility, skewness, kurtosis

innovation_daily              = pd.DataFrame(columns=['date', 'maturity', 'in_ex_return', 'in_volatility', 'in_skewness', 'in_kurtosis'])
innovation_30_days            = pd.DataFrame(columns=['date', 'maturity', 'in_ex_return'])

for index, expectation in expectations.iterrows():

    realized_P_values   = es50_P_values_daily.loc[es50_P_values_daily.index == expectation['loctimestamp']]
    date                = expectation['loctimestamp']
    maturity            = expectation['daystomaturity']
    in_ex_return        = realized_P_values['P_LN_Returns'].values[0] - expectation['exp_r']
    in_volatility       = realized_P_values['P_variance'].values[0]   - expectation['exp_vol']
    in_skewness         = realized_P_values['P_skewness'].values[0] -  expectation['exp_skew']
    in_kurtosis         = realized_P_values['P_kurtosis'].values[0] - expectation['exp_kur']
    innovation_daily.loc[len(innovation_daily)] = [date, maturity, in_ex_return, in_volatility, in_skewness, in_kurtosis]

if plot_final:
    plt.subplot(211).set_title('The ex post realized excess return on the ex ante recovered expected return')
    plt.plot(es50_P_values_daily.index.to_pydatetime(), es50_P_values_daily['P_LN_Returns'])
    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('Innovation')

    plt.subplot(212).set_title("Conditional EXPECTED 30-days Excess-Returns")
    plt.suptitle('Conditional EXPECTED 30-days Excess-Returns')
    expectations['loctimestamp'] = pd.to_datetime(expectations['loctimestamp'])
    for day in days_to_maturity_all:
        plt.plot(expectations.loc[(expectations['daystomaturity'] == day)]['loctimestamp'].dt.to_pydatetime(), expectations.loc[(expectations['daystomaturity'] == day), ['exp_r']], label="%s" % (day))

    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('Conditional Exp Ex R in %')

print("\n")
print("--------------- Daily Innovation --------")
print(innovation_daily.head())
print("-----------------------------------------")
print("\n")

# ---------------- Innovation for 30 days excess return ----------------
# innovation on 30 days excess return

for index, expectation in monthly_exp_excess_return.iterrows():
    realized_P_value_30_days    = es50_P_values_30_days.loc[es50_P_values_30_days.index == np.datetime64(index.date())]

    for maturity in days_to_maturity_all:
        in_ex_return        = realized_P_value_30_days['30_days_P_LN_Return'].values[0] - expectation[maturity]
        innovation_30_days.loc[len(innovation_30_days)] = [np.datetime64(index.date()), maturity, in_ex_return]

if plot_final:
    plt.figure('Innovation')
    plt.suptitle('Innovation between 30 days realized and expected excess return')
    for day in days_to_maturity_all:
        plt.plot(innovation_30_days['date'].dt.to_pydatetime(), innovation_30_days['in_ex_return'], label="%s"%(day))

    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('Innovation')

print("\n")
print("--------------- 30 days innovation on excess return -------------")
print(innovation_30_days.head())
print("-----------------------------------------------------------------")
print("\n")

#
# =========================================================================
#                           Regression
# =========================================================================
#
# regressing the ex post realized excess return on the ex ante recovered expected return mu_t,
# the ex post innovation in expected return, delta_mu_t+1, and, as controls,
# the ex ante recovered volatility sigma_t and the ex ante VIX
#

# realized_stat_t->t+h = alpha + beta * E_t[stat_t->t+h] + error

# ---------------- Regression ----------------

es50_P_values_30_days       = es50_P_values_30_days.fillna(0)
monthly_exp_excess_return   = monthly_exp_excess_return.fillna(0)
ols_estimates       = pd.DataFrame(columns=['maturity', 'ß0', 'ß1'])

for day in days_to_maturity_all:
    Y = np.array(monthly_exp_excess_return[day])[1:len(es50_P_values_30_days)]
    X = np.array(monthly_exp_excess_return[day])[0:len(monthly_exp_excess_return)-1]
    X = sm.add_constant(X)

    model   = sm.OLS(Y, X)
    results = model.fit()
    ols_estimates.loc[len(ols_estimates)] = [day, results.params[0], results.params[1]]
    print("AR(1)- Modell for maturity: ", day, ' -> a0 = ', results.params[0], " and a1 = ", results.params[1])
    print(results.summary())

# Interested in:
# Does the recovered probability give rise to reasonable expected returns,
# that is time-varying risk premia.
# ß1 > 0 -> higher ex ante expected return ist associated with higher ex post realized return
# looking for ß0 = 0 and ß1 = 1
# =========================================================================
#                           Results
# =========================================================================
#

plt.show()