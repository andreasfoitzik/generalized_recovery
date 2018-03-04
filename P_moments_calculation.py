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

maturities      = [7,30,60,91,182,365]

es50_daily                  = pd.DataFrame(columns=['loctimestamp', 'Opening_Price', 'Closing_Price', 'P_Returns', 'P_variance', 'P_skewness', 'P_kurtosis'])
es50_7_days                 = pd.DataFrame(columns=['loctimestamp', 'Closing_Price', 'P_Returns', 'P_variance', 'P_skewness', 'P_kurtosis'])
es50_30_days                = pd.DataFrame(columns=['loctimestamp', 'Closing_Price', 'P_Returns', 'P_variance', 'P_skewness', 'P_kurtosis'])
es50_60_days                = pd.DataFrame(columns=['loctimestamp', 'Closing_Price', 'P_Returns', 'P_variance', 'P_skewness', 'P_kurtosis'])
es50_91_days                = pd.DataFrame(columns=['loctimestamp', 'Closing_Price', 'P_Returns', 'P_variance', 'P_skewness', 'P_kurtosis'])
es50_182_days               = pd.DataFrame(columns=['loctimestamp', 'Closing_Price', 'P_Returns', 'P_variance', 'P_skewness', 'P_kurtosis'])
es50_365_days               = pd.DataFrame(columns=['loctimestamp', 'Closing_Price', 'P_Returns', 'P_variance', 'P_skewness', 'P_kurtosis'])

begin                       = np.datetime64('2002-01-03', 'D')
end                         = np.datetime64('2015-07-01', 'D')

#---------------- TXT Declaration ----------------

TXT_DATE                = 'Date'
TXT_RETURN_IN_PCT       = 'Return in %'
TXT_CREATED_SUCCESS     = "Successfully created ->"
FILE_PATH               = "data/P_values"

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

#
# =========================================================================
#                               Intraday
# =========================================================================
#

# intraday returns in percentage
es50_5min['P_Returns']     = (es50_5min['price'] / es50_5min['price'].shift(1))

plt.figure('Intraday Realized Returns')
plt.suptitle('Intraday Realized Returns under P-Density')
plt.plot(es50_5min.index.to_pydatetime(), es50_5min['P_Returns'])
plt.xlabel(TXT_DATE)
plt.ylabel(TXT_RETURN_IN_PCT)

#
# =========================================================================
#                               Daily
# =========================================================================
#

es50            = es50_5min.copy()

for date in es50.index.map(lambda t: t.date()).unique():
    print("Date: ", date)

    values                          = es50[date: (np.datetime64(date) + np.timedelta64(1, 'D'))]
    opening_price                   = values.iloc[0]['price']
    closing_price                   = values.iloc[-1]['price']
    returns                         = (closing_price / opening_price)
    variance                        = np.sum(values['P_Returns']**2)
    skewness                        = np.sqrt(values.shape[0])*np.sum(values['P_Returns']**3) / variance**(3/2)
    kurtosis                        = np.sqrt(values.shape[0])*np.sum(values['P_Returns']**4) / variance**(2)
    es50_daily.loc[len(es50_daily)] = [date, opening_price, closing_price, returns, variance, skewness, kurtosis]
    es50                            = es50.drop(es50[date: (np.datetime64(date) + np.timedelta64(1, 'D'))].index)

del es50

plt.figure('Daily Realized Returns')
plt.suptitle('Daily Realized Returns under P-Density')
plt.plot(es50_daily['loctimestamp'], es50_daily['P_Returns']*100)
plt.xlabel(TXT_DATE)
plt.ylabel(TXT_RETURN_IN_PCT)

plt.figure('Daily Realized Variance')
plt.suptitle('Daily Realized Variance under P-Density')
plt.plot(es50_daily['loctimestamp'], es50_daily['P_variance'])
plt.xlabel(TXT_DATE)
plt.ylabel('P_variance')

plt.figure('Daily Realized Skewness')
plt.suptitle('Daily Realized Skewness under P-Density')
plt.plot(es50_daily['loctimestamp'], es50_daily['P_skewness'])
plt.xlabel(TXT_DATE)
plt.ylabel('P_skewness')

plt.figure('Daily Realized Kurtosis')
plt.suptitle('Daily Realized Kurtosis under P-Density')
plt.plot(es50_daily['loctimestamp'], es50_daily['P_kurtosis'])
plt.xlabel(TXT_DATE)
plt.ylabel('P_kurtosis')

#
# =========================================================================
#                  Calculating Moments for each Maturity
# =========================================================================
#

es50_daily['loctimestamp']  = pd.to_datetime(es50_daily['loctimestamp'], format="%Y-%m-%d")
es50_daily                  = es50_daily.set_index('loctimestamp')

es50            = es50_daily.copy()

for date in es50.index.map(lambda t: t.date()).unique():
    print(TXT_DATE,": ", date)

    for maturity in maturities:
        t_maturity          = (np.datetime64(date) + np.timedelta64(maturity, 'D'))
        values              = es50[date: t_maturity]
        returns             = (values.iloc[-1]['Closing_Price'] / values.iloc[0]['Closing_Price'])
        variance            = np.sum(values['P_Returns']**2)
        skewness            = np.sqrt(values.shape[0])*np.sum(values['P_Returns']**3) / variance**(3/2)
        kurtosis            = np.sqrt(values.shape[0])*np.sum(values['P_Returns']**4) / variance**(2)

        if maturity == 7:
            es50_7_days.loc[len(es50_7_days)] = [t_maturity, values.iloc[-1]['Closing_Price'], returns, variance, skewness, kurtosis]
        elif maturity == 30:
            es50_30_days.loc[len(es50_30_days)] = [t_maturity, values.iloc[-1]['Closing_Price'], returns, variance, skewness, kurtosis]
        elif maturity == 60:
            es50_60_days.loc[len(es50_60_days)] = [t_maturity, values.iloc[-1]['Closing_Price'], returns, variance, skewness, kurtosis]
        elif maturity == 91:
            es50_91_days.loc[len(es50_91_days)] = [t_maturity, values.iloc[-1]['Closing_Price'], returns, variance, skewness, kurtosis]
        elif maturity == 182:
            es50_182_days.loc[len(es50_182_days)] = [t_maturity, values.iloc[-1]['Closing_Price'], returns, variance, skewness, kurtosis]
        elif maturity == 365:
            es50_365_days.loc[len(es50_365_days)] = [t_maturity, values.iloc[-1]['Closing_Price'], returns, variance, skewness, kurtosis]

del es50

es50_7_days                 = es50_7_days.set_index('loctimestamp')
es50_7_days.index           = pd.to_datetime(es50_7_days.index)

es50_30_days                = es50_30_days.set_index('loctimestamp')
es50_30_days.index          = pd.to_datetime(es50_30_days.index)

es50_60_days                = es50_60_days.set_index('loctimestamp')
es50_60_days.index          = pd.to_datetime(es50_60_days.index)

es50_91_days                = es50_91_days.set_index('loctimestamp')
es50_91_days.index          = pd.to_datetime(es50_91_days.index)

es50_182_days               = es50_182_days.set_index('loctimestamp')
es50_182_days.index         = pd.to_datetime(es50_182_days.index)

es50_365_days               = es50_365_days.set_index('loctimestamp')
es50_365_days.index         = pd.to_datetime(es50_365_days.index)

plt.figure('7 days Realized Returns')
plt.suptitle('7 days Realized Returns under P-Density')
plt.plot(es50_7_days.index.to_pydatetime(), es50_7_days['P_Returns']*100)
plt.xlabel(TXT_DATE)
plt.ylabel(TXT_RETURN_IN_PCT)

plt.figure('30 days Realized Returns')
plt.suptitle('30 days Realized Returns under P-Density')
plt.plot(es50_30_days.index.to_pydatetime(), es50_30_days['P_Returns']*100)
plt.xlabel(TXT_DATE)
plt.ylabel(TXT_RETURN_IN_PCT)

plt.figure('60 days Realized Returns')
plt.suptitle('60 days Realized Returns under P-Density')
plt.plot(es50_60_days.index.to_pydatetime(), es50_60_days['P_Returns']*100)
plt.xlabel(TXT_DATE)
plt.ylabel(TXT_RETURN_IN_PCT)

plt.figure('91 days Realized Returns')
plt.suptitle('91 days Realized Returns under P-Density')
plt.plot(es50_91_days.index.to_pydatetime(), es50_91_days['P_Returns']*100)
plt.xlabel(TXT_DATE)
plt.ylabel(TXT_RETURN_IN_PCT)

plt.figure('182 days Realized Returns')
plt.suptitle('182 days Realized Returns under P-Density')
plt.plot(es50_182_days.index.to_pydatetime(), es50_182_days['P_Returns']*100)
plt.xlabel(TXT_DATE)
plt.ylabel(TXT_RETURN_IN_PCT)

plt.figure('365 days Realized Returns')
plt.suptitle('365 days Realized Returns under P-Density')
plt.plot(es50_365_days.index.to_pydatetime(), es50_365_days['P_Returns']*100)
plt.xlabel(TXT_DATE)
plt.ylabel(TXT_RETURN_IN_PCT)

#
# =========================================================================
#                               Results to .csv
# =========================================================================
#

es50_daily.to_csv("%s/es50_P_values_daily.csv" % (FILE_PATH), sep=';')
es50_7_days.to_csv("%s/es50_P_values_7_days.csv" % (FILE_PATH), sep=';')
es50_30_days.to_csv("%s/es50_P_values_30_days.csv" % (FILE_PATH), sep=';')
es50_60_days.to_csv("%s/es50_P_values_60_days.csv" % (FILE_PATH), sep=';')
es50_91_days.to_csv("%s/es50_P_values_91_days.csv" % (FILE_PATH), sep=';')
es50_182_days.to_csv("%s/es50_P_values_182_days.csv" % (FILE_PATH), sep=';')
es50_365_days.to_csv("%s/es50_P_values_365_days.csv" % (FILE_PATH), sep=';')

print(TXT_CREATED_SUCCESS," %s/es50_P_values_daily.csv "  % (FILE_PATH))
print(TXT_CREATED_SUCCESS," %s/es50_P_values_7_days.csv "  % (FILE_PATH))
print(TXT_CREATED_SUCCESS," %s/es50_P_values_30_days.csv "  % (FILE_PATH))
print(TXT_CREATED_SUCCESS," %s/es50_P_values_60_days.csv "  % (FILE_PATH))
print(TXT_CREATED_SUCCESS," %s/es50_P_values_91_days.csv "  % (FILE_PATH))
print(TXT_CREATED_SUCCESS," %s/es50_P_values_182_days.csv "  % (FILE_PATH))
print(TXT_CREATED_SUCCESS," %s/es50_P_values_364_days.csv "  % (FILE_PATH))

plt.show()