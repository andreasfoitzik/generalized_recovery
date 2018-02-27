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

import sys

import scipy.stats as stats
import scipy.integrate as integrate

from scipy.optimize import minimize
from scipy.interpolate import interp1d

import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')
import statsmodels.api as sm
from statsmodels.graphics.regressionplots import abline_plot


#
#====================================================
#                       DATA
#====================================================
#

# ---------------- variable declarations ----------------
MATURITY_365                = 365
TETAS_AMOUNT                = 11
STATE_SPACES_AMOUNT         = 10

# ---------------- Load Data ----------------
data_qvar_all                   = pd.read_csv("data/FiglewskiStandardizationEOD_DE0009652396D1_Qmoments.csv", sep = ';')
data_rnd_all                    = pd.read_csv("data/FiglewskiStandardizationEOD_DE0009652396D1_rnd.csv",  sep = ';')
data_rf_all                     = pd.read_csv("data/riskfree_rate.csv", sep = ';')

# ---------------- Reduce Data ----------------
data_qvar_all                   = data_qvar_all.loc[data_qvar_all['daystomaturity'] <= MATURITY_365]
data_rnd_all                    = data_rnd_all.loc[data_rnd_all['daystomaturity'] <= MATURITY_365]
data_rf_all                     = data_rf_all.loc[data_rf_all['daystomaturity'] <= MATURITY_365]

# ---------------- Process Data ----------------
data_qvar_all                   = pd.merge(data_qvar_all, data_rf_all, how='left', left_on=['loctimestamp', 'daystomaturity'], right_on=['loctimestamp', 'daystomaturity'])
data_rnd_all                    = pd.merge(data_rnd_all, data_rf_all, how='left', left_on=['loctimestamp', 'daystomaturity'], right_on=['loctimestamp', 'daystomaturity'])

data_rnd_all['moneyness']       = (data_rnd_all['implStrike'] / data_rnd_all['underlyingforwardprice']).round(decimals=3)
data_rnd_all['maturity']        = data_rnd_all['daystomaturity'] / MATURITY_365
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
average_exp_excess_return       = []

plot                            = False
plot_final                      = True
cmap                            = plt.cm.get_cmap("hsv", 12)

dates                           = data_rnd_all['loctimestamp'].unique()
dates                           = dates[2000:2200]

#
#====================================================
#              GENERALIZED RECOVERY
#====================================================
#

# ---------------- For each date ----------------

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
    # get highest variance, for largest state space
    current_VIX             = data_qvar.loc[(data_qvar['daystomaturity'] == 365)]['Q_variance'].values[0]

    # as state we use the current return, which is 1
    lowest_State            = 1 - (2.5 * current_VIX)
    highest_State           = 1 + (4 * current_VIX)

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
            elif index > 0 and state + 1 < len(states) and values['moneyness'].iloc[0] < states[state] and values['moneyness'].iloc[-1] > states[state]:
                # interpolate results of all non defined states
                f = interp1d(x, y)
                pi[index, state] = f(states[state])
                print("Interpolate for ", pi[index, state])

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
    state_spaces = []
    state_space = np.zeros((STATE_SPACES_AMOUNT, 2))

    state_space[0, 0] = 1 - 2.5 * current_VIX
    state_space[0, 1] = 1 - 2 * current_VIX
    state_spaces.append(np.matrix(states[(states >= state_space[0, 0]) & (states < state_space[0, 1])]))

    state_space_equal_portion_from = 1 - 2 * current_VIX
    state_space_equal_portion_to = 1 + 2 * current_VIX
    portion = (state_space_equal_portion_to - state_space_equal_portion_from) / 8

    for i in range(1, 9):
        state_space[i, 0] = state_space[i - 1, 1]
        state_space[i, 1] = state_space[i, 0] + portion
        state_spaces.append(np.matrix(states[(states >= state_space[i, 0]) & (states < state_space[i, 1])]))

    state_space[9, 0] = 1 + 2 * current_VIX
    state_space[9, 1] = 1 + 4 * current_VIX
    state_spaces.append(np.matrix(states[(states >= state_space[9, 0]) & (states < state_space[9, 1])]))

    # ---------------- Design-Matrix B ----------------
    B                       = np.zeros((len(states), len(teta)))
    # counting the rank of the matrix
    amount_of_states_set    = 0

    # Design Matrix Level
    for i in range (0, len(states)):
        B[i, 0]     = 1.00

    # Design Matrix State Space 1
    for i in range(0, state_spaces[0].shape[1]):
        B[i, 1] = (i) / state_spaces[0].shape[1]

    amount_of_states_set += state_spaces[0].shape[1]
    for i in range(amount_of_states_set, len(states)):
        B[i, 1] = 1

    # Design Matrix for State Space 2 till 10
    for a in range(1, len(state_spaces)):
        for i in range(0, amount_of_states_set):
            B[i, a + 1] = 0
        counter = 0
        amount_of_states_putting = (amount_of_states_set + state_spaces[a].shape[1])
        for i in range(amount_of_states_set, amount_of_states_putting):
            counter += 1
            B[i, a + 1] = counter / state_spaces[a].shape[1]
        amount_of_states_set += state_spaces[a].shape[1]
        for i in range(amount_of_states_putting, len(states)):
            B[i, a + 1] = 1

    if plot:
        plt.figure('DESIGN-MATRIX')
        plt.suptitle('Design-Matrix')
        for i in range(0, B.shape[1]):
            plt.plot(states, B[:,i], label='Column %s'% (i+1))

        plt.legend()
        plt.xlabel('States')
        plt.ylabel('Column')

    # ---------------- Minimization-Problem ----------------
    # boundaries
    b1 = (0, None)
    b2 = (0, 1.0)
    bnds = [b1, b1, b1, b1, b1, b1, b1, b1, b1, b1, b1, b2]

    teta_seed = 1
    rf_seed = np.sum(delta) / len(delta)

    x0_seed = [1, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.60, 0.55, 0.50, rf_seed]

    def objective(x0):
        x1 = x0[0:len(teta)]
        x2 = x0[len(teta):len(teta) + 1]

        x = pi.dot(B).dot(x1) - (alpha + beta * x2)

        return np.linalg.norm(x)

    res = minimize(objective, x0_seed, method='L-BFGS-B', bounds=bnds)

    teta = res.x[0:len(teta)]
    delta_min = res.x[len(teta):len(teta) + 1]

    # ---------------- Pricing-Kernel ----------------

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

    pricing_kernel = (1 / inv_pricing_kernel)

    if plot:
        plt.figure('PRICING-KERNEL')
        plt.suptitle('PRICING-KERNEL')
        plt.plot(states, pricing_kernel, color='r')
        for i in range(0, len(state_space)):
            plt.axvline(x=state_space[i, 0], color='b')

        plt.axvline(x=state_space[len(state_space) - 1, 1], color='b')
        plt.legend()
        plt.xlabel('States')
        plt.ylabel('Inverse PRICING-KERNEL')

    # ---------------- Physical Probabilities ----------------
    delta_diag          = np.diagflat(np.matrix([delta_min**1,delta_min**2,delta_min**3,delta_min**4,delta_min**5,delta_min**6]))

    if np.linalg.det(delta_diag) != 0:
        delta_diag_invers   = np.linalg.inv(delta_diag)
    else:
        print("Singular-Matrix - based on poor data!")

    P_init              = delta_diag_invers.dot(pi).dot(np.diag( B.dot(teta)))

    # normalize P to have row sums of one
    kernel              = integrate.trapz(states * pricing_kernel, states)
    P                   = np.asarray(P_init / kernel)

    # TODO - normalize Kernel to have an expected value of 1/rf

    if plot:
        plt.figure('PHYSICAL PROBABILITY DISTRIBUTION')
        plt.subplot(211).set_title('PHYSICAL PROBABILITY DISTRIBUTION')
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

    # ---------------- calculating moments under P-density ----------------
    exp_r   = np.zeros(len(P))
    exp_r_2 = np.zeros(len(P))
    exp_r_3 = np.zeros(len(P))
    exp_r_4 = np.zeros(len(P))

    for i in range (0, len(P)):
        p_list     = np.array(P[i, :]).tolist()
        exp_r[i]   = integrate.trapz((states) * p_list, states)
        exp_r_2[i] = integrate.trapz((states ** 2) * p_list, states)
        exp_r_3[i] = integrate.trapz((states ** 3) * p_list, states)
        exp_r_4[i] = integrate.trapz((states ** 4) * p_list, states)

    # ---------------- conditional Phy. Exp. for next day ----------------
    if plot:
        plt.subplot(212)
        for i in range(0, len(exp_r)):
            plt.plot(days_to_maturity[i], exp_r[i], 'ro', label='Days to maturity %s' % (days_to_maturity[i]), color=cmap(i))

        plt.legend(days_to_maturity_all)
        plt.xlabel('Maturities')
        plt.ylabel('Expected Return in %')

    # ---------------- conditional volatility ----------------
    var = exp_r_2 - np.power(exp_r, 2)
    vol = np.sqrt(var)

    if plot:
        plt.figure('CONDITIONAL EXPECTATIONS')
        plt.subplot(311).set_title('EXPECTED VOLATILITY')
        for i in range(0, len(vol)):
            plt.plot(days_to_maturity[i], vol[i], 'ro', color=cmap(i), label='Days to maturity %s' % (days_to_maturity[i]))

        plt.legend(days_to_maturity_all)
        plt.xlabel('Maturities')
        plt.ylabel('Expected Volatility')

    # ---------------- conditional Skewness ----------------
    skewness = (exp_r_3 - (3 * exp_r.dot(exp_r_2)) + 2 * np.power(exp_r, 3)) / np.power((exp_r_2 - np.power(exp_r, 2)), 3 / 2)

    if plot:
        plt.subplot(312).set_title('EXPECTED SKEWNESS')
        for i in range(0, len(skewness)):
            plt.plot(days_to_maturity[i], skewness[i], 'ro', color=cmap(i), label='Days to maturity %s' % (days_to_maturity[i]))

        plt.legend(days_to_maturity_all)
        plt.xlabel('Maturities')
        plt.ylabel('Skewness')

    # ---------------- conditional Kurtosis ----------------
    kurtosis = np.zeros(len(P))
    for i in range(0, P.shape[0]):
        num = 1 / P.shape[1] * sum((P[i] - np.mean(P[i])) ** 4)
        den = (1 / P.shape[1] * sum((P[i] - np.mean(P[i])) ** 2)) ** 2
        kurtosis[i] = num / den

    if plot:
        plt.subplot(313).set_title('EXPECTED KURTOSIS')
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
#                            Expectations
# =========================================================================
#

plt.figure('EXPECTED RETURNS')
plt.suptitle('EXPECTED RETURNS')
for i in range(0, len(days_to_maturity)):
    plt.plot(expectations.loc[expectations['daystomaturity'] == days_to_maturity[i]]['loctimestamp'].dt.date, expectations.loc[expectations['daystomaturity'] == days_to_maturity[i]]['exp_r'], 'ro', color=cmap(i), label='Days to maturity %s' % (days_to_maturity[i]))

plt.legend(days_to_maturity_all)
plt.xlabel('Date')
plt.ylabel('Return in %')

#
# =========================================================================
#                           Calculate Results
# =========================================================================
#

# ---------------- Calculate Conditional EXPECTED 30-days Excess-Returns ----------------
# get expected value for each maturity
# merge both dataframes to calculate the excess return for the expected values in the past
# TODO - annualize monthly return

for day in days_to_maturity_all:
    data                        = expectations.loc[(expectations['daystomaturity'] == day), ['loctimestamp', 'exp_r']].copy()
    data                        = data.set_index('loctimestamp')

    monthly_recovered_return    = data.resample('30D').last()
    monthly_recovered_return['returns']    = monthly_recovered_return.pct_change()

    average_exp_excess_return.append(np.mean(monthly_exp_excess_return[day]))

    del data

# annualize averaged monthly expected returns
average_exp_excess_return = [np.sqrt(12)*x for x in average_exp_excess_return]

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
#                           REALIZED P-VALUES
# =========================================================================
#
#

# values calculated by P_moments_calculation.py
es50_P_values_daily                     = pd.read_csv("data/es50_P_values_daily.csv", sep = ';')
es50_P_values_daily['loctimestamp']     = pd.to_datetime(es50_P_values_daily['loctimestamp'])

es50_P_values_30_days                   = pd.read_csv("data/es50_P_values_30_days.csv", sep = ';')
es50_P_values_30_days['loctimestamp']   = pd.to_datetime(es50_P_values_30_days['loctimestamp'])

avg_daily_ln_return                     = np.mean(es50_P_values_daily['P_LN_Returns'])
avg_30_day_ln_return                    = np.mean(es50_P_values_30_days['30_days_P_LN_Return'])

#
# =========================================================================
#                           Regression
# =========================================================================
#
# regressing the ex post realized excess return on the ex ante recovered expected return mu_t,
# the ex post innovation in expected return, delta_mu_t+1, and, as controls,
# the ex ante recovered volatility sigma_t and the ex ante VIX
#
# Interested in:
# Does the recovered probability give rise to reasonable expected returns,
# that is time-varying risk premia!
# ß1 > 0 -> higher ex ante expected return ist associated with higher ex post realized return

# ---------------- Regression  ----------------

ols         = pd.DataFrame(columns=['maturity','ß_ret_0','ß_ret_1','ß_var_0','ß_var_1','ß_ske_0','ß_ske_1','ß_kur_0','ß_kur_1'])

for day in days_to_maturity_all:
    values  = pd.merge(es50_P_values_daily, expectations.loc[expectations['daystomaturity'] == day], how='inner', left_on=['loctimestamp'], right_on=['loctimestamp'])

    # ---------------- Regression Returns ----------------
    Y_ret = values['P_LN_Returns'].values
    X_ret = np.reshape(values['exp_vol'].values, (1, len(values))).T
    X_ret = sm.add_constant(X_ret)

    model_ret = sm.OLS(Y_ret, X_ret)
    results_ret = model_ret.fit()

    # ---------------- Regression Volatility ----------------
    Y_var = values['P_variance'].values
    X_var = np.reshape(values['exp_vol'].values, (1,len(values))).T
    X_var = sm.add_constant(X_var)

    model_var = sm.OLS(Y_var, X_var)
    results_var = model_var.fit()

    # ---------------- Regression Skewness ----------------
    Y_ske = values['P_skewness'].values
    X_ske = np.reshape(values['exp_skew'].values, (1, len(values))).T
    X_ske = sm.add_constant(X_ske)

    model_ske = sm.OLS(Y_ske, X_ske)
    results_ske = model_ske.fit()

    # ---------------- Regression Kurtosis ----------------
    Y_kur = values['P_kurtosis'].values
    X_kur = np.reshape(values['exp_kur'].values, (1,len(values))).T
    X_kur = sm.add_constant(X_kur)

    model_kur = sm.OLS(Y_kur, X_kur)
    results_kur = model_kur.fit()

    ols.loc[len(ols)] = [day, results_ret.params[0], results_ret.params[1],
                              results_var.params[0], results_var.params[1],
                              results_ske.params[0], results_ske.params[1],
                              results_kur.params[0], results_kur.params[1]]

    print("Regression - Modell: ")
    print(ols)

# scatter-plot data
ax = df.plot(x='motifScore', y='expression', kind='scatter')

# plot regression line
abline_plot(model_results=model.fit(), ax=ax)

#
# =========================================================================
#                           Results
# =========================================================================
#

plt.show()