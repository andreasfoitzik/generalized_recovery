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

#---------------- Variables Declaration ----------------

es50_daily                  = pd.DataFrame(columns=['loctimestamp', 'P_LN_Returns', 'P_variance', 'P_skewness', 'P_kurtosis'])
es50_daily['loctimestamp']  = pd.to_datetime(es50_daily['loctimestamp'], format="%Y-%m-%d")
es50_30_days                = pd.DataFrame(columns=['loctimestamp', 'daily_P_LN_Returns', '30_days_P_LN_Return'])

begin                       = np.datetime64('2002-01-03', 'D')
end                         = np.datetime64('2015-07-01', 'D')

#
#====================================================
#                       DATA
#====================================================
#

# ---------------- Load Data ----------------
es50_5min                   = pd.read_csv("data/eurostoxx50_prices_5m.csv", sep = ';')
es50_5min['loctimestamp']   = pd.to_datetime(es50_5min['loctimestamp'], format='%d.%m.%y %H:%M')
es50_5min                   = es50_5min.set_index('loctimestamp')
es50_5min                   = es50_5min[begin:end]

rf                          = pd.read_csv("data/riskfree_rate.csv", sep = ';')
rf['loctimestamp']          = pd.to_datetime(rf['loctimestamp'], format="%Y-%m-%d")
rf                          = rf.loc[(rf['daystomaturity'] == 365)]

#
# =========================================================================
#                               Intraday
# =========================================================================
#

# intraday log returns
es50_5min['P_LN_Returns']     = np.log(es50_5min['price'] / es50_5min['price'].shift(1))*100

plt.figure('Intraday Realized Returns')
plt.suptitle('Intraday Realized Returns under P-Density')
plt.plot(es50_5min.index.to_pydatetime(), es50_5min['P_LN_Returns'])
plt.xlabel('Date')
plt.ylabel('P_LN_Returns')

#
# =========================================================================
#                               Daily
# =========================================================================
#

es50            = es50_5min.copy()

for date in es50.index.map(lambda t: t.date()).unique():
    print("Date: ", date)

    values              = es50[date: (np.datetime64(date) + np.timedelta64(1, 'D'))]
    ln_return           = np.log(values.iloc[-1]['price'] / values.iloc[0]['price'])*100
    excess_ln_return    = ln_return - rf.loc[(rf['loctimestamp'].dt.date == date)]['riskfree']
    variance            = np.sum(values['P_LN_Returns']**2)
    skewness            = np.sqrt(values.shape[0])*np.sum(values['P_LN_Returns']**3) / variance**(3/2)
    kurtosis            = np.sqrt(values.shape[0])*np.sum(values['P_LN_Returns']**4) / variance**(2)
    es50_daily.loc[len(es50_daily)] = [date, ln_return, variance, skewness, kurtosis]
    es50                = es50.drop(es50[date: (np.datetime64(date) + np.timedelta64(1, 'D'))].index)

del es50

plt.figure('Daily Realized Returns')
plt.suptitle('Daily Realized Returns under P-Density')
plt.plot(es50_daily['loctimestamp'], es50_daily['P_LN_Returns'])
plt.xlabel('Date')
plt.ylabel('P_LN_Returns')

plt.figure('Daily Realized Variance')
plt.suptitle('Daily Realized Variance under P-Density')
plt.plot(es50_daily['loctimestamp'], es50_daily['P_variance'])
plt.xlabel('Date')
plt.ylabel('P_variance')

plt.figure('Daily Realized Skewness')
plt.suptitle('Daily Realized Skewness under P-Density')
plt.plot(es50_daily['loctimestamp'], es50_daily['P_skewness'])
plt.xlabel('Date')
plt.ylabel('P_skewness')

plt.figure('Daily Realized Kurtosis')
plt.suptitle('Daily Realized Kurtosis under P-Density')
plt.plot(es50_daily['loctimestamp'], es50_daily['P_kurtosis'])
plt.xlabel('Date')
plt.ylabel('P_kurtosis')

#
# =========================================================================
#                               30-days
# =========================================================================
#
# 30 days returns

es50_daily                  = es50_daily.set_index('loctimestamp')
es50_daily.index            = pd.to_datetime(es50_daily.index)

es50_30_days                = es50_30_days.set_index('loctimestamp')
es50_30_days.index          = pd.to_datetime(es50_30_days.index)

es50_30_days['daily_P_LN_Returns']      = es50_daily['P_LN_Returns'].resample('30D').last()
es50_30_days['30_days_P_LN_Return']     = es50_30_days['daily_P_LN_Returns'].pct_change() / 100
es50_30_days                            = es50_30_days.drop('daily_P_LN_Returns', 1)

plt.figure('30 days Realized Returns')
plt.suptitle('30 days Realized Returns under P-Density')

plt.plot(es50_30_days.index.to_pydatetime(), es50_30_days['30_days_P_LN_Return'])
plt.xlabel('Date')
plt.ylabel('P_LN_Return')

#
# =========================================================================
#                               Results to .csv
# =========================================================================
#

es50_daily.to_csv("data/es50_P_values_daily.csv", sep=';')
es50_30_days.to_csv("data/es50_P_values_30_days.csv", sep=';')

print("Successfully created -> data/es50_P_values_daily.csv ")
print("Successfully created -> data/es50_P_values_30_days.csv ")

plt.show()